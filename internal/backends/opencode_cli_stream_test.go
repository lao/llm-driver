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
			name:       "reasoning effort is encoded in the model id",
			config:     core.Config{Provider: core.ProviderOpenCode, Flavor: core.FlavorCLI, Model: "anthropic/claude-sonnet-4-5"},
			request:    core.Request{MaxTokens: 32, Messages: []core.Message{core.User("q")}, Effort: core.EffortHigh},
			wantPath:   "opencode",
			wantArgs:   []string{"run", "--format", "json", "--model", "anthropic/claude-sonnet-4-5#high"},
			wantPrompt: "q",
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
	if response.CompletionReason != core.CompletionReasonStop {
		t.Fatalf("completion reason = %q, want %q", response.CompletionReason, core.CompletionReasonStop)
	}
}

// TestOpenCodeCLIStreamTextOnlyCompletes covers the opencode 2.x stream: the
// process ends after the text part with no terminal step_finish. The stream
// must return the text rather than a missing_step_finish error.
func TestOpenCodeCLIStreamTextOnlyCompletes(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{lines: []string{
		`{"type":"step_start","sessionID":"ses_123","part":{"id":"prt_1","type":"step-start"}}`,
		`{"type":"text","sessionID":"ses_123","part":{"id":"prt_2","type":"text","text":"finished answer"}}`,
	}}
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

	_, err, response := collectStream(t, stream)
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if response.Text != "finished answer" {
		t.Fatalf("Text = %q, want %q", response.Text, "finished answer")
	}
	if response.CompletionReason != core.CompletionReasonUnverified || response.Usage != (core.Usage{}) {
		t.Fatalf("response = %+v, want unverified reason and zero usage without step_finish", response)
	}
}

// A clean exit can still end on a result error (a headless run that
// auto-rejected a tool); the stderr explanation must reach the error.
func TestOpenCodeCLIStreamCleanExitCarriesStderr(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{
		lines: []string{
			`{"type":"tool_use","sessionID":"ses_1","part":{"type":"tool","tool":"read"}}`,
			`{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_1","type":"step-finish","reason":"tool-calls"}}`,
		},
		stderr: []byte("permission denied\n"),
	}
	client := newCLIStreamClient(core.Config{
		Provider: core.ProviderOpenCode,
		Flavor:   core.FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	stream, err := client.GenerateStream(context.Background(), core.Request{MaxTokens: 32, Messages: []core.Message{core.User("q")}})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()
	_, err, _ = collectStream(t, stream)
	var contextual *core.Error
	if !errors.As(err, &contextual) {
		t.Fatalf("stream error = %T %v, want *core.Error", err, err)
	}
	if contextual.Code != core.CodeToolCalls {
		t.Fatalf("code = %q, want %q", contextual.Code, core.CodeToolCalls)
	}
	if !strings.Contains(contextual.Message, "permission denied") {
		t.Fatalf("message = %q, want the stderr diagnostic", contextual.Message)
	}
}

// Finish is valid only after the stream is drained. A caller that finalizes
// early must not get partial text presented as a completed answer.
func TestOpenCodeCLIStreamFinishBeforeExit(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{lines: []string{
		`{"type":"text","sessionID":"ses_1","part":{"id":"prt_1","type":"text","text":"partial"}}`,
	}}
	client := newCLIStreamClient(core.Config{
		Provider: core.ProviderOpenCode,
		Flavor:   core.FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	stream, err := client.GenerateStream(context.Background(), core.Request{MaxTokens: 32, Messages: []core.Message{core.User("q")}})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()
	_, err = stream.source.Finish()
	var contextual *core.Error
	if !errors.As(err, &contextual) {
		t.Fatalf("Finish() error = %T %v, want *core.Error", err, err)
	}
	if contextual.Code != "incomplete_stream" {
		t.Fatalf("code = %q, want incomplete_stream", contextual.Code)
	}
}

func TestOpenCodeCompletionReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		reason string
		want   core.CompletionReason
	}{
		{reason: "stop", want: core.CompletionReasonStop},
		{reason: "tool-calls", want: ""},
		{reason: "length", want: core.CompletionReasonMaxTokens},
		{reason: "content-filter", want: core.CompletionReasonContentFilter},
		{reason: "", want: ""},
		{reason: "future-reason", want: ""},
	}
	for _, test := range tests {
		if got := openCodeCompletionReason(test.reason); got != test.want {
			t.Fatalf("openCodeCompletionReason(%q) = %q, want %q", test.reason, got, test.want)
		}
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
		wantReason string
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
			name:     "missing step finish and text",
			runner:   &fakeStreamingRunner{lines: []string{`{"type":"step_start","sessionID":"ses_1"}`}},
			wantCode: "missing_step_finish",
		},
		{
			name:       "missing text",
			runner:     &fakeStreamingRunner{lines: openCodeStreamLines()[3:]},
			wantCode:   "missing_text",
			wantReason: "stop",
		},
		{
			name: "tool call ended without text",
			runner: &fakeStreamingRunner{lines: []string{
				`{"type":"tool_use","sessionID":"ses_1","part":{"type":"tool","tool":"glob","state":{"status":"completed"}}}`,
				`{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_1","type":"step-finish","reason":"tool-calls","tokens":{}}}`,
			}},
			wantCode:   core.CodeToolCalls,
			wantReason: "tool-calls",
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
			if target.Reason != test.wantReason {
				t.Fatalf("reason = %q, want %q", target.Reason, test.wantReason)
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

func TestOpenCodeParserDedupesStepFinishAndRejectsTerminalToolCalls(t *testing.T) {
	t.Parallel()

	finish := `{"type":"step_finish","part":{"id":"prt_f","reason":"stop","tokens":{"input":4,"output":2}}}`
	parser := newOpenCodeParser("m", nil, "generate")
	for _, line := range []string{
		`{"type":"text","part":{"id":"prt_t","text":"hi"}}`, finish, finish,
	} {
		if _, err := parser.event([]byte(line)); err != nil {
			t.Fatal(err)
		}
	}
	response, err := parser.response()
	if err != nil {
		t.Fatal(err)
	}
	if response.Usage.InputTokens != 4 || response.Usage.OutputTokens != 2 {
		t.Fatalf("usage double counted: %+v", response.Usage)
	}

	parser = newOpenCodeParser("m", nil, "generate")
	for _, line := range []string{
		`{"type":"text","part":{"id":"prt_t","text":"let me check"}}`,
		`{"type":"step_finish","part":{"id":"prt_f","reason":"tool-calls"}}`,
	} {
		if _, err := parser.event([]byte(line)); err != nil {
			t.Fatal(err)
		}
	}
	_, err = parser.response()
	var coreErr *core.Error
	if !errors.As(err, &coreErr) || coreErr.Code != core.CodeToolCalls {
		t.Fatalf("want tool_calls error, got %v", err)
	}
}

func TestOpenCodeFailureErrorFallsBackWhenMessageSanitizesEmpty(t *testing.T) {
	t.Parallel()

	parser := newOpenCodeParser("m", nil, "generate")
	_, err := parser.event([]byte(`{"type":"error","error":{"name":"X","data":{"message":"\u001b[2J\u0007"},"message":"real reason"}}`))
	var coreErr *core.Error
	if !errors.As(err, &coreErr) || coreErr.Message != "real reason" {
		t.Fatalf("want fallback to error.message, got %v", err)
	}
}

func TestOpenCodeParserRedeliveredToolFinishAfterTextKeepsAnswer(t *testing.T) {
	t.Parallel()

	toolFinish := `{"type":"step_finish","part":{"id":"prt_f1","reason":"tool-calls"}}`
	parser := newOpenCodeParser("m", nil, "generate")
	for _, line := range []string{
		`{"type":"text","part":{"id":"prt_t1","text":"checking"}}`, toolFinish,
		`{"type":"text","part":{"id":"prt_t2","text":" done"}}`, toolFinish,
	} {
		if _, err := parser.event([]byte(line)); err != nil {
			t.Fatal(err)
		}
	}
	response, err := parser.response()
	if err != nil || response.Text != "checking done" {
		t.Fatalf("response = %+v, err = %v", response, err)
	}
}
