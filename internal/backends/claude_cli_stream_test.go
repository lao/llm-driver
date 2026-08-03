package backends

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lao/llm-driver/internal/core"
)

// claudeStreamJSONLines mirrors `claude -p --output-format stream-json
// --include-partial-messages`: partial-message lines wrap raw Anthropic SSE
// events, and the final result line is identical to batch mode.
func claudeStreamJSONLines() []string {
	return []string{
		`{"type":"system","subtype":"init","session_id":"session-123"}`,
		`{"type":"stream_event","event":{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}}`,
		`{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}}`,
		`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello, "}}}`,
		`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}}`,
		`{"type":"stream_event","event":{"type":"content_block_stop","index":0}}`,
		`{"type":"assistant","message":{"id":"msg_1","content":[{"type":"text","text":"Hello, world"}]}}`,
		`{"type":"result","subtype":"success","is_error":false,"result":"Hello, world","session_id":"session-123",` +
			`"usage":{"input_tokens":11,"output_tokens":7,"cache_read_input_tokens":5,"cache_creation_input_tokens":3}}`,
	}
}

func TestClaudeCLIStreamCommand(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		config     core.Config
		request    core.Request
		wantPath   string
		wantArgs   []string
		wantPrompt string
	}{
		{
			name:     "single user message",
			config:   core.Config{Provider: core.ProviderClaude, Flavor: core.FlavorCLI, Model: "claude-test"},
			request:  core.Request{MaxTokens: 32, Messages: []core.Message{core.User("Hello from stdin")}},
			wantPath: "claude",
			wantArgs: []string{
				"-p", "--output-format", "stream-json", "--include-partial-messages", "--verbose",
				"--permission-mode", "default", "--model", "claude-test",
			},
			wantPrompt: "Hello from stdin",
		},
		{
			name: "system and multi turn with custom CLI settings",
			config: core.Config{
				Provider: core.ProviderClaude,
				Flavor:   core.FlavorCLI,
				Model:    "claude-test",
				CLIPath:  "/opt/bin/claude-custom",
				CLIArgs:  []string{"--permission-mode", "plan"},
			},
			request: core.Request{
				System:    "Be concise.",
				MaxTokens: 32,
				Messages:  []core.Message{core.User("First"), core.Assistant("Second"), core.User("Third")},
			},
			wantPath: "/opt/bin/claude-custom",
			wantArgs: []string{
				"-p", "--output-format", "stream-json", "--include-partial-messages", "--verbose",
				"--permission-mode", "default", "--model", "claude-test",
				"--append-system-prompt", "Be concise.",
				"--permission-mode", "plan",
			},
			wantPrompt: "User: First\n\nAssistant: Second\n\nUser: Third",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			runner := &fakeStreamingRunner{lines: claudeStreamJSONLines()}
			client := newCLIStreamClient(test.config, runner.run)
			stream, err := client.GenerateStream(context.Background(), test.request)
			if err != nil {
				t.Fatalf("GenerateStream(): %v", err)
			}
			defer stream.Close()

			if _, err, _ := collectStream(t, stream); err != nil {
				t.Fatalf("stream error: %v", err)
			}
			runner.assertCall(t, test.wantPath, test.wantArgs, test.wantPrompt)
		})
	}
}

func TestClaudeCLIStreamSuccess(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{lines: claudeStreamJSONLines()}
	client := newCLIStreamClient(core.Config{
		Provider: core.ProviderClaude,
		Flavor:   core.FlavorCLI,
		Model:    "claude-test",
	}, runner.run)

	stream, err := client.GenerateStream(context.Background(), core.Request{
		MaxTokens: 32,
		Messages:  []core.Message{core.User("Hello")},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	chunks, err, response := collectStream(t, stream)
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if strings.Join(chunks, "|") != "Hello, |world" {
		t.Fatalf("chunks = %#v", chunks)
	}
	if response.ID != "session-123" || response.Model != "claude-test" || response.Text != "Hello, world" {
		t.Fatalf("response = %+v", response)
	}
	if response.Provider != core.ProviderClaude || response.Flavor != core.FlavorCLI {
		t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
	}
	wantUsage := core.Usage{InputTokens: 11, OutputTokens: 7, CachedInputTokens: 5, CacheCreationInputTokens: 3}
	if response.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", response.Usage, wantUsage)
	}
}

func TestClaudeCLIStreamDeliversChunksBeforeExit(t *testing.T) {
	t.Parallel()

	// The runner holds every line after the first delta, so reaching a chunk
	// proves the adapter reads stdout while the process is still running.
	runner := &fakeStreamingRunner{
		lines:    claudeStreamJSONLines(),
		holdFrom: 4,
		gate:     make(chan struct{}),
	}
	client := newCLIStreamClient(core.Config{
		Provider: core.ProviderClaude,
		Flavor:   core.FlavorCLI,
		Model:    "claude-test",
	}, runner.run)

	stream, err := client.GenerateStream(context.Background(), core.Request{
		MaxTokens: 32,
		Messages:  []core.Message{core.User("Hello")},
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
	chunks, err, response := collectStream(t, stream)
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if strings.Join(chunks, "|") != "world" || response.Text != "Hello, world" {
		t.Fatalf("chunks = %#v, response = %+v", chunks, response)
	}
}

func TestClaudeCLIStreamErrors(t *testing.T) {
	t.Parallel()

	launchFailure := errors.New("launch failed")
	tests := []struct {
		name       string
		runner     *fakeStreamingRunner
		wantIs     error
		wantStatus int
		wantCode   string
		startFails bool
	}{
		{
			name:       "missing executable",
			runner:     &fakeStreamingRunner{startErr: core.ErrExecutableNotFound},
			wantIs:     core.ErrExecutableNotFound,
			startFails: true,
		},
		{
			name:       "launch failure",
			runner:     &fakeStreamingRunner{startErr: launchFailure},
			wantIs:     launchFailure,
			startFails: true,
		},
		{
			name: "nonzero exit",
			runner: &fakeStreamingRunner{
				lines:    claudeStreamJSONLines()[:6],
				stderr:   []byte("authentication failed\n"),
				exitCode: 17,
			},
			wantStatus: 17,
		},
		{
			name:   "malformed line",
			runner: &fakeStreamingRunner{lines: append(claudeStreamJSONLines()[:4], "not-json")},
		},
		{
			name: "error result",
			runner: &fakeStreamingRunner{lines: append(claudeStreamJSONLines()[:4],
				`{"type":"result","subtype":"error_max_turns","result":"turn limit reached"}`)},
			wantCode: "error_max_turns",
		},
		{
			name:     "missing result line",
			runner:   &fakeStreamingRunner{lines: claudeStreamJSONLines()[:7]},
			wantCode: "missing_result",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client := newCLIStreamClient(core.Config{
				Provider: core.ProviderClaude,
				Flavor:   core.FlavorCLI,
				Model:    "claude-test",
			}, test.runner.run)

			stream, err := client.GenerateStream(context.Background(), core.Request{
				MaxTokens: 32,
				Messages:  []core.Message{core.User("Hello")},
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

			var target *core.Error
			if !errors.As(err, &target) {
				t.Fatalf("stream error = %T %v, want *core.Error", err, err)
			}
			if target.Provider != core.ProviderClaude || target.Flavor != core.FlavorCLI || target.Operation != "generate_stream" {
				t.Fatalf("error context = %+v", target)
			}
			if target.Status != test.wantStatus || target.Code != test.wantCode {
				t.Fatalf("status/code = %d/%q, want %d/%q", target.Status, target.Code, test.wantStatus, test.wantCode)
			}
			if test.wantIs != nil && !errors.Is(err, test.wantIs) {
				t.Fatalf("stream error = %v, want errors.Is(%v)", err, test.wantIs)
			}
		})
	}
}

func TestClaudeCLIStreamCloseTerminatesRunner(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{
		lines:    claudeStreamJSONLines(),
		holdFrom: 4,
		gate:     make(chan struct{}),
	}
	client := newCLIStreamClient(core.Config{
		Provider: core.ProviderClaude,
		Flavor:   core.FlavorCLI,
		Model:    "claude-test",
	}, runner.run)

	stream, err := client.GenerateStream(context.Background(), core.Request{
		MaxTokens: 32,
		Messages:  []core.Message{core.User("Hello")},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	if !stream.Next() {
		t.Fatalf("Next() = false: %v", stream.Err())
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close(): %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("second Close(): %v", err)
	}
	if runner.closes != 1 {
		t.Fatalf("process closes = %d, want 1", runner.closes)
	}
	if stream.Next() {
		t.Fatal("Next() = true after Close()")
	}
}

func TestClaudeCLIStreamMalformedResultLine(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		result      string
		wantCode    string
		wantMessage string
		wantDecode  bool
	}{
		{
			// The envelope decodes but the result payload does not, so the failure
			// has to come from the second decode rather than being silently ignored.
			name:       "result payload with wrong field types",
			result:     `{"type":"result","subtype":"success","usage":"none"}`,
			wantDecode: true,
		},
		{
			name:        "result without a subtype",
			result:      `{"type":"result"}`,
			wantCode:    "unsuccessful_result",
			wantMessage: "Claude CLI returned an unsuccessful result",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			runner := &fakeStreamingRunner{lines: append(claudeStreamJSONLines()[:5], test.result)}
			client := newCLIStreamClient(core.Config{
				Provider: core.ProviderClaude,
				Flavor:   core.FlavorCLI,
				Model:    "claude-test",
			}, runner.run)

			stream, err := client.GenerateStream(context.Background(), streamRequest())
			if err != nil {
				t.Fatalf("GenerateStream(): %v", err)
			}
			defer stream.Close()

			_, err, _ = collectStream(t, stream)
			var target *core.Error
			if !errors.As(err, &target) {
				t.Fatalf("stream error = %T %v, want *core.Error", err, err)
			}
			if target.Provider != core.ProviderClaude || target.Flavor != core.FlavorCLI || target.Operation != "generate_stream" {
				t.Fatalf("error context = %+v", target)
			}
			if target.Code != test.wantCode {
				t.Fatalf("code = %q, want %q", target.Code, test.wantCode)
			}
			if test.wantMessage != "" && target.Message != test.wantMessage {
				t.Fatalf("message = %q, want %q", target.Message, test.wantMessage)
			}
			if test.wantDecode && target.Err == nil {
				t.Fatalf("error = %+v, want a wrapped decode failure", target)
			}
		})
	}
}
