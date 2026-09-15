package backends

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lao/llm-driver/internal/core"
)

// openCodeStreamLines mirrors `opencode run --format json`: one JSON event per
// line, text parts carrying the answer and step_finish the terminal state.
func openCodeStreamLines() []string {
	return []string{
		`{"type":"step_start","sessionID":"ses_123","part":{"id":"prt_1","type":"step-start"}}`,
		`{"type":"text","sessionID":"ses_123","part":{"id":"prt_2","type":"text","text":"Hello, "}}`,
		`{"type":"text","sessionID":"ses_123","part":{"id":"prt_3","type":"text","text":"world"}}`,
		`{"type":"step_finish","sessionID":"ses_123","part":{"id":"prt_4","type":"step-finish","reason":"stop",` +
			`"tokens":{"input":11,"output":7,"reasoning":2,"cache":{"read":5,"write":3}}}}`,
	}
}

func TestOpenCodeCLIStreamCommand(t *testing.T) {
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
			name:       "single user message",
			config:     core.Config{Provider: core.ProviderOpenCode, Flavor: core.FlavorCLI, Model: "anthropic/claude-sonnet-4-5"},
			request:    core.Request{MaxTokens: 32, Messages: []core.Message{core.User("Hello from stdin")}},
			wantPath:   "opencode",
			wantArgs:   []string{"run", "--format", "json", "--model", "anthropic/claude-sonnet-4-5"},
			wantPrompt: "Hello from stdin",
		},
		{
			name: "system and multi turn with custom CLI settings",
			config: core.Config{
				Provider: core.ProviderOpenCode,
				Flavor:   core.FlavorCLI,
				Model:    "anthropic/claude-sonnet-4-5",
				CLIPath:  "/opt/bin/opencode-custom",
				CLIArgs:  []string{"--agent", "build"},
			},
			request: core.Request{
				System:    "Be concise.",
				MaxTokens: 32,
				Messages:  []core.Message{core.User("First"), core.Assistant("Second"), core.User("Third")},
			},
			wantPath:   "/opt/bin/opencode-custom",
			wantArgs:   []string{"run", "--format", "json", "--model", "anthropic/claude-sonnet-4-5", "--agent", "build"},
			wantPrompt: "Be concise.\n\nUser: First\n\nAssistant: Second\n\nUser: Third",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			runner := &fakeStreamingRunner{lines: openCodeStreamLines()}
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

func TestOpenCodeCLIStreamSuccess(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{lines: openCodeStreamLines()}
	client := newCLIStreamClient(core.Config{
		Provider: core.ProviderOpenCode,
		Flavor:   core.FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
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
	if response.ID != "ses_123" || response.Model != "anthropic/claude-sonnet-4-5" || response.Text != "Hello, world" {
		t.Fatalf("response = %+v", response)
	}
	if response.Provider != core.ProviderOpenCode || response.Flavor != core.FlavorCLI {
		t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
	}
	wantUsage := core.Usage{InputTokens: 11, OutputTokens: 7, CachedInputTokens: 5, CacheCreationInputTokens: 3, ReasoningTokens: 2}
	if response.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", response.Usage, wantUsage)
	}
}

func TestOpenCodeCLIStreamStructuredOutput(t *testing.T) {
	t.Parallel()

	lines := []string{
		`{"type":"text","sessionID":"ses_123","part":{"id":"prt_1","type":"text","text":"{\"answer\":"}}`,
		`{"type":"text","sessionID":"ses_123","part":{"id":"prt_2","type":"text","text":"\"42\"}"}}`,
		`{"type":"step_finish","sessionID":"ses_123","part":{"id":"prt_3","type":"step-finish","reason":"stop","tokens":{}}}`,
	}
	runner := &fakeStreamingRunner{lines: lines}
	client := newCLIStreamClient(core.Config{
		Provider: core.ProviderOpenCode,
		Flavor:   core.FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)

	stream, err := client.GenerateStream(context.Background(), core.Request{
		MaxTokens:    32,
		Messages:     []core.Message{core.User("q")},
		OutputSchema: &core.OutputSchema{Name: "answer", Schema: testSchema},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	chunks, err, response := collectStream(t, stream)
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if !strings.Contains(string(runner.stdin), string(testSchema)) {
		t.Fatalf("stdin prompt %q does not carry the schema", runner.stdin)
	}
	if strings.Join(chunks, "") != testJSON {
		t.Fatalf("chunks = %#v, want the JSON document", chunks)
	}
	if response.Text != testJSON || string(response.StructuredOutput) != testJSON {
		t.Fatalf("response Text=%q StructuredOutput=%q, want %q", response.Text, response.StructuredOutput, testJSON)
	}
}

func TestOpenCodeCLIStreamErrors(t *testing.T) {
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
				lines:    openCodeStreamLines()[:3],
				stderr:   []byte("authentication failed\n"),
				exitCode: 17,
			},
			wantStatus: 17,
		},
		{
			name:   "malformed line",
			runner: &fakeStreamingRunner{lines: append(openCodeStreamLines()[:2], "not-json")},
		},
		{
			name: "error event",
			runner: &fakeStreamingRunner{lines: []string{
				`{"type":"error","sessionID":"ses_1","error":{"name":"ProviderError","data":{"message":"model unavailable"}}}`,
			}},
			wantCode: "ProviderError",
		},
		{
			name:     "missing step finish",
			runner:   &fakeStreamingRunner{lines: openCodeStreamLines()[:3]},
			wantCode: "missing_step_finish",
		},
		{
			name:     "missing text",
			runner:   &fakeStreamingRunner{lines: openCodeStreamLines()[3:]},
			wantCode: "missing_text",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client := newCLIStreamClient(core.Config{
				Provider: core.ProviderOpenCode,
				Flavor:   core.FlavorCLI,
				Model:    "anthropic/claude-sonnet-4-5",
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
			if target.Provider != core.ProviderOpenCode || target.Flavor != core.FlavorCLI || target.Operation != "generate_stream" {
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

func TestOpenCodeCLIStreamDeliversChunksBeforeExit(t *testing.T) {
	t.Parallel()

	// The runner holds every line after the first text event, so reaching a chunk
	// proves the adapter reads stdout while the process is still running.
	runner := &fakeStreamingRunner{
		lines:    openCodeStreamLines(),
		holdFrom: 3,
		gate:     make(chan struct{}),
	}
	client := newCLIStreamClient(core.Config{
		Provider: core.ProviderOpenCode,
		Flavor:   core.FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
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
