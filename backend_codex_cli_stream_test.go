package llmwrapper

import (
	"context"
	"errors"
	"strings"
	"testing"
)

const (
	codexStreamThread    = `{"type":"thread.started","thread_id":"thread-123"}`
	codexStreamCompleted = `{"type":"item.completed","item":{"id":"item-1","type":"agent_message","text":"Hello, world"}}`
	codexStreamTurn      = `{"type":"turn.completed","usage":{"input_tokens":11,"cached_input_tokens":5,` +
		`"cache_write_input_tokens":3,"output_tokens":7,"reasoning_output_tokens":2}}`
)

// codexStreamDeltaLines exercises the CLI versions that publish an agent
// message incrementally: each update carries the cumulative item text, and the
// completion repeats it.
func codexStreamDeltaLines() []string {
	return []string{
		codexStreamThread,
		`{"type":"item.started","item":{"id":"item-1","type":"agent_message","text":""}}`,
		`{"type":"item.updated","item":{"id":"item-1","type":"agent_message","text":"Hello, "}}`,
		`{"type":"item.updated","item":{"id":"item-1","type":"agent_message","text":"Hello, world"}}`,
		codexStreamCompleted,
		codexStreamTurn,
	}
}

// codexStreamCompletedOnlyLines exercises the CLI versions that publish an
// agent message only once it is finished.
func codexStreamCompletedOnlyLines() []string {
	return []string{codexStreamThread, codexStreamCompleted, codexStreamTurn}
}

func newCodexStreamClient(runner streamingCommandRunner) *Client {
	return newCLIStreamClient(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorCLI,
		Model:    "codex-test",
	}, runner)
}

func TestCodexCLIStreamCommandMatchesBatch(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{lines: codexStreamCompletedOnlyLines()}
	client := newCLIStreamClient(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorCLI,
		Model:    "codex-test",
		CLIPath:  "/opt/bin/codex-custom",
		CLIArgs:  []string{"--full-auto"},
	}, runner.run)

	stream, err := client.GenerateStream(context.Background(), Request{
		System:    "Be concise.",
		MaxTokens: 32,
		Messages:  []Message{User("First"), Assistant("Second"), User("Third")},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()
	if _, err, _ := collectStream(t, stream); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	runner.assertCall(t, "/opt/bin/codex-custom", []string{
		"exec", "--json", "--sandbox", "read-only", "--skip-git-repo-check", "--model", "codex-test",
		"--config", `developer_instructions="Be concise."`,
		"--full-auto",
		"-",
	}, "User: First\n\nAssistant: Second\n\nUser: Third")
}

func TestCodexCLIStreamChunkSources(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		lines      []string
		wantChunks string
	}{
		{name: "cumulative item updates", lines: codexStreamDeltaLines(), wantChunks: "Hello, |world"},
		{name: "completed item only", lines: codexStreamCompletedOnlyLines(), wantChunks: "Hello, world"},
		{
			name: "explicit delta field",
			lines: []string{
				codexStreamThread,
				`{"type":"item.updated","item":{"id":"item-1","type":"agent_message"},"delta":"Hello, "}`,
				`{"type":"item.updated","item":{"id":"item-1","type":"agent_message"},"delta":"world"}`,
				codexStreamCompleted,
				codexStreamTurn,
			},
			wantChunks: "Hello, |world",
		},
		{
			name: "unknown event types are ignored",
			lines: []string{
				codexStreamThread,
				`{"type":"item.completed","item":{"id":"reason-1","type":"reasoning","text":"thinking"}}`,
				`{"type":"future.event","payload":{"unknown":true}}`,
				codexStreamCompleted,
				codexStreamTurn,
			},
			wantChunks: "Hello, world",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			runner := &fakeStreamingRunner{lines: test.lines}
			client := newCodexStreamClient(runner.run)
			stream, err := client.GenerateStream(context.Background(), Request{
				MaxTokens: 32,
				Messages:  []Message{User("Hello")},
			})
			if err != nil {
				t.Fatalf("GenerateStream(): %v", err)
			}
			defer stream.Close()

			chunks, err, response := collectStream(t, stream)
			if err != nil {
				t.Fatalf("stream error: %v", err)
			}
			if strings.Join(chunks, "|") != test.wantChunks {
				t.Fatalf("chunks = %#v, want %q", chunks, test.wantChunks)
			}
			if response.Text != "Hello, world" {
				t.Fatalf("response text = %q", response.Text)
			}
			if response.ID != "thread-123" || response.Model != "codex-test" {
				t.Fatalf("response = %+v", response)
			}
			if response.Provider != ProviderOpenAI || response.Flavor != FlavorCLI {
				t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
			}
			wantUsage := Usage{
				InputTokens:              11,
				OutputTokens:             7,
				CachedInputTokens:        5,
				CacheCreationInputTokens: 3,
				ReasoningTokens:          2,
			}
			if response.Usage != wantUsage {
				t.Fatalf("usage = %+v, want %+v", response.Usage, wantUsage)
			}
		})
	}
}

func TestCodexCLIStreamDeliversChunksBeforeExit(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{
		lines:    codexStreamDeltaLines(),
		holdFrom: 3,
		gate:     make(chan struct{}),
	}
	client := newCodexStreamClient(runner.run)
	stream, err := client.GenerateStream(context.Background(), Request{
		MaxTokens: 32,
		Messages:  []Message{User("Hello")},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	if !stream.Next() {
		t.Fatalf("Next() = false before the process finished: %v", stream.Err())
	}
	if stream.Current().Text != "Hello, " {
		t.Fatalf("first chunk = %q", stream.Current().Text)
	}

	close(runner.gate)
	if _, err, _ := collectStream(t, stream); err != nil {
		t.Fatalf("stream error: %v", err)
	}
}

func TestCodexCLIStreamErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		runner     *fakeStreamingRunner
		wantStatus int
		wantCode   string
		startFails bool
	}{
		{
			name:       "missing executable",
			runner:     &fakeStreamingRunner{startErr: ErrExecutableNotFound},
			startFails: true,
		},
		{
			name: "turn failed",
			runner: &fakeStreamingRunner{lines: []string{
				codexStreamThread,
				`{"type":"turn.failed","error":{"message":"model unavailable"}}`,
			}},
			wantCode: "turn_failed",
		},
		{
			name: "turn failed with nonzero exit",
			runner: &fakeStreamingRunner{
				lines: []string{
					codexStreamThread,
					`{"type":"turn.failed","error":{"message":"model unavailable"}}`,
				},
				exitCode: 3,
			},
			wantCode:   "turn_failed",
			wantStatus: 3,
		},
		{
			name: "stream error event",
			runner: &fakeStreamingRunner{lines: []string{
				codexStreamThread,
				`{"type":"error","message":"stream aborted"}`,
			}},
			wantCode: "error",
		},
		{
			name:     "missing turn completion",
			runner:   &fakeStreamingRunner{lines: []string{codexStreamThread, codexStreamCompleted}},
			wantCode: "missing_turn_completion",
		},
		{
			name:     "missing final message",
			runner:   &fakeStreamingRunner{lines: []string{codexStreamThread, codexStreamTurn}},
			wantCode: "missing_final_message",
		},
		{
			name:   "malformed line",
			runner: &fakeStreamingRunner{lines: []string{codexStreamThread, "not-json"}},
		},
		{
			name: "nonzero exit without diagnostic",
			runner: &fakeStreamingRunner{
				lines:    []string{codexStreamThread},
				stderr:   []byte("authentication failed\n"),
				exitCode: 17,
			},
			wantStatus: 17,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client := newCodexStreamClient(test.runner.run)
			stream, err := client.GenerateStream(context.Background(), Request{
				MaxTokens: 32,
				Messages:  []Message{User("Hello")},
			})
			if test.startFails {
				if stream != nil {
					t.Fatal("GenerateStream() returned a stream for a failed start")
				}
			} else {
				if err != nil {
					t.Fatalf("GenerateStream(): %v", err)
				}
				defer stream.Close()
				_, err, _ = collectStream(t, stream)
			}

			var target *Error
			if !errors.As(err, &target) {
				t.Fatalf("stream error = %T %v, want *Error", err, err)
			}
			if target.Provider != ProviderOpenAI || target.Flavor != FlavorCLI || target.Operation != "generate_stream" {
				t.Fatalf("error context = %+v", target)
			}
			if target.Status != test.wantStatus || target.Code != test.wantCode {
				t.Fatalf("status/code = %d/%q, want %d/%q", target.Status, target.Code, test.wantStatus, test.wantCode)
			}
		})
	}
}

func TestCodexCLIStreamFailureWithoutMessage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		line     string
		wantCode string
	}{
		{name: "turn failed", line: `{"type":"turn.failed"}`, wantCode: "turn_failed"},
		{name: "error event", line: `{"type":"error"}`, wantCode: "error"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			runner := &fakeStreamingRunner{lines: []string{codexStreamThread, test.line}}
			client := newCodexStreamClient(runner.run)
			stream, err := client.GenerateStream(context.Background(), streamRequest())
			if err != nil {
				t.Fatalf("GenerateStream(): %v", err)
			}
			defer stream.Close()

			_, err, _ = collectStream(t, stream)
			var target *Error
			if !errors.As(err, &target) {
				t.Fatalf("stream error = %T %v, want *Error", err, err)
			}
			if target.Code != test.wantCode {
				t.Fatalf("code = %q, want %q", target.Code, test.wantCode)
			}
			// A failure the CLI reports without text still has to explain itself.
			if target.Message == "" {
				t.Fatal("message is empty for a failure event without a message")
			}
		})
	}
}
