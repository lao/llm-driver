package llmdriver

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"testing"
)

func TestOpenCodeCLICommand(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		config     Config
		request    Request
		wantPath   string
		wantArgs   []string
		wantPrompt string
	}{
		{
			name: "single user message",
			config: Config{
				Provider: ProviderOpenCode,
				Flavor:   FlavorCLI,
				Model:    "anthropic/claude-sonnet-4-5",
			},
			request:    Request{MaxTokens: 32, Messages: []Message{User("Hello from stdin")}},
			wantPath:   "opencode",
			wantArgs:   []string{"run", "--format", "json", "--model", "anthropic/claude-sonnet-4-5"},
			wantPrompt: "Hello from stdin",
		},
		{
			name: "system and multi turn with custom CLI settings",
			config: Config{
				Provider: ProviderOpenCode,
				Flavor:   FlavorCLI,
				Model:    "anthropic/claude-sonnet-4-5",
				CLIPath:  "/opt/bin/opencode-custom",
				CLIArgs:  []string{"--agent", "build"},
			},
			request: Request{
				System:    "Be concise.",
				MaxTokens: 32,
				Messages:  []Message{User("First"), Assistant("Second"), User("Third")},
			},
			wantPath:   "/opt/bin/opencode-custom",
			wantArgs:   []string{"run", "--format", "json", "--model", "anthropic/claude-sonnet-4-5", "--agent", "build"},
			wantPrompt: "Be concise.\n\nUser: First\n\nAssistant: Second\n\nUser: Third",
		},
		{
			name: "reasoning effort is encoded in the model id",
			config: Config{
				Provider: ProviderOpenCode,
				Flavor:   FlavorCLI,
				Model:    "anthropic/claude-sonnet-4-5",
			},
			request:    Request{MaxTokens: 32, Messages: []Message{User("q")}, Effort: EffortHigh},
			wantPath:   "opencode",
			wantArgs:   []string{"run", "--format", "json", "--model", "anthropic/claude-sonnet-4-5#high"},
			wantPrompt: "q",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			runner := &fakeCommandRunner{stdout: []byte(openCodeSuccessOutput("ok"))}
			client, err := newClient(test.config, runner.run)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			if _, err := client.Generate(context.Background(), test.request); err != nil {
				t.Fatalf("Generate(): %v", err)
			}
			runner.assertCall(t, test.wantPath, test.wantArgs, test.wantPrompt)
		})
	}
}

// opencode 2.x can only express an effort through a named model's #variant.
// Both shapes that cannot work are rejected, matching the library's
// unsupported-feature contract instead of silently dropping the level.
func TestOpenCodeCLIEffortRequiresModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		model string
		want  string
	}{
		{name: "empty model", model: "", want: "requires Config.Model"},
		{name: "inline variant", model: "p/m#low", want: "conflicts with the variant"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			runner := &fakeCommandRunner{stdout: []byte(openCodeSuccessOutput("ok"))}
			client, err := newClient(Config{
				Provider: ProviderOpenCode,
				Flavor:   FlavorCLI,
				Model:    test.model,
			}, runner.run)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}
			_, err = client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}, Effort: EffortHigh})
			if !errors.Is(err, ErrUnsupportedFeature) {
				t.Fatalf("Generate() error = %v, want ErrUnsupportedFeature", err)
			}
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %q, want %q", err, test.want)
			}
			if runner.calls != 0 {
				t.Fatalf("the CLI was spawned %d times for an unsupported effort", runner.calls)
			}
			stream, err := client.GenerateStream(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}, Effort: EffortHigh})
			if !errors.Is(err, ErrUnsupportedFeature) {
				t.Fatalf("GenerateStream() error = %v, want ErrUnsupportedFeature", err)
			}
			if stream != nil {
				t.Fatal("GenerateStream() returned a stream for an unsupported effort")
			}
		})
	}
}

// A provider/CLI-controlled error message is sanitized and capped before it
// reaches caller logs, just like stderr.
func TestOpenCodeCLIErrorEventIsSanitized(t *testing.T) {
	t.Parallel()

	message := strings.Repeat("x", 1000) + `\u001b]52;c;SGVscG8=\u0007`
	code := strings.Repeat("y", 1000) + `\u001b[?25l`
	runner := &fakeCommandRunner{stdout: []byte(
		`{"type":"error","sessionID":"ses_1","error":{"type":"` + code + `","data":{"message":"` + message + `"}}}` + "\n",
	)}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}})
	if err == nil {
		t.Fatal("Generate() succeeded, want an error")
	}
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Generate() error = %T %v, want *Error", err, err)
	}
	if len(target.Message) > 500 {
		t.Fatalf("message length = %d, want <= 500", len(target.Message))
	}
	if strings.ContainsAny(target.Message, "\x1b\x07") {
		t.Fatalf("message carries an escape sequence: %q", target.Message)
	}
	if len(target.Code) > 500 {
		t.Fatalf("code length = %d, want <= 500", len(target.Code))
	}
	if strings.ContainsAny(target.Code, "\x1b\x07") {
		t.Fatalf("code carries an escape sequence: %q", target.Code)
	}
}

// A provider-controlled step_finish reason is sanitized before it reaches the
// error's Reason field, which callers log.
func TestOpenCodeCLIErrorReasonIsSanitized(t *testing.T) {
	t.Parallel()

	reason := strings.Repeat("z", 1000) + `\u001b]52;c;SGVscG8=\u0007`
	runner := &fakeCommandRunner{stdout: []byte(
		`{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_1","type":"step-finish","reason":"` + reason + `"}}` + "\n",
	)}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}})
	if err == nil {
		t.Fatal("Generate() succeeded, want an error")
	}
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Generate() error = %T %v, want *Error", err, err)
	}
	if len(target.Reason) > 500 {
		t.Fatalf("reason length = %d, want <= 500", len(target.Reason))
	}
	if strings.ContainsAny(target.Reason, "\x1b\x07") {
		t.Fatalf("reason carries an escape sequence: %q", target.Reason)
	}
}

func TestOpenCodeCLISuccessConcatenatesTextPartsAndUsage(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(
		`{"type":"step_start","sessionID":"ses_123","part":{"id":"prt_1","type":"step-start"}}` + "\n" +
			`{"type":"text","sessionID":"ses_123","part":{"id":"prt_2","type":"text","text":"Hello, "}}` + "\n" +
			`{"type":"text","sessionID":"ses_123","part":{"id":"prt_3","type":"text","text":"world"}}` + "\n" +
			`{"type":"step_finish","sessionID":"ses_123","part":{"id":"prt_4","type":"step-finish","reason":"stop",` +
			`"tokens":{"total":20,"input":19,"output":7,"reasoning":3,"cache":{"write":5,"read":11}}}}` + "\n",
	)}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	response, err := client.Generate(context.Background(), Request{
		MaxTokens: 32,
		Messages:  []Message{User("Hello")},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	if response.ID != "ses_123" || response.Model != "anthropic/claude-sonnet-4-5" || response.Text != "Hello, world" {
		t.Fatalf("response = %+v", response)
	}
	if response.Provider != ProviderOpenCode || response.Flavor != FlavorCLI {
		t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
	}
	wantUsage := Usage{
		InputTokens:              19,
		OutputTokens:             7,
		CachedInputTokens:        11,
		CacheCreationInputTokens: 5,
		ReasoningTokens:          3,
	}
	if response.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", response.Usage, wantUsage)
	}
	if response.CompletionReason != CompletionReasonStop {
		t.Fatalf("completion reason = %q, want %q", response.CompletionReason, CompletionReasonStop)
	}
}

// A multi-step turn reports one step_finish per step; the response usage is the
// sum, not just the last step's count.
func TestOpenCodeCLIUsageSumsSteps(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(
		`{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_1","type":"step-finish","reason":"tool-calls","tokens":{"input":10,"output":2,"reasoning":1,"cache":{"read":3,"write":4}}}}` + "\n" +
			`{"type":"text","sessionID":"ses_1","part":{"id":"prt_2","type":"text","text":"answer"}}` + "\n" +
			`{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_3","type":"step-finish","reason":"stop","tokens":{"input":20,"output":5,"reasoning":2,"cache":{"read":6,"write":7}}}}` + "\n",
	)}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	response, err := client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	want := Usage{InputTokens: 30, OutputTokens: 7, CachedInputTokens: 9, CacheCreationInputTokens: 11, ReasoningTokens: 3}
	if response.Usage != want {
		t.Fatalf("Usage = %+v, want %+v", response.Usage, want)
	}
}

// TestOpenCodeCLITextOnlyRunWithoutStepFinishCompletes covers the opencode 2.x
// stream: a plain text answer ends after the text part, with no terminal
// step_finish. A completed process with text is the finish signal; usage and
// completion reason are simply unavailable.
func TestOpenCodeCLITextOnlyRunWithoutStepFinishCompletes(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(
		`{"type":"step_start","sessionID":"ses_1","part":{"id":"prt_1","type":"step-start"}}` + "\n" +
			`{"type":"text","sessionID":"ses_1","part":{"id":"prt_2","type":"text","text":"finished answer"}}` + "\n",
	)}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	response, err := client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if response.Text != "finished answer" {
		t.Fatalf("Text = %q, want %q", response.Text, "finished answer")
	}
	if response.CompletionReason != CompletionReasonUnverified {
		t.Fatalf("CompletionReason = %q, want %q when opencode omits step_finish", response.CompletionReason, CompletionReasonUnverified)
	}
	if response.Usage != (Usage{}) {
		t.Fatalf("Usage = %+v, want zero when opencode omits step_finish", response.Usage)
	}
}

// A tool step's finish reason belongs to that step, not to a later text answer.
// opencode 2.x omits the terminal step_finish, so a tool-call-then-text turn
// must not report CompletionReasonToolUse for a complete answer.
func TestOpenCodeCLITextAfterToolStepDoesNotInheritReason(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(
		`{"type":"step_start","sessionID":"ses_1","part":{"id":"prt_1","type":"step-start"}}` + "\n" +
			`{"type":"tool_use","sessionID":"ses_1","part":{"id":"prt_2","type":"tool","tool":"read"}}` + "\n" +
			`{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_3","type":"step-finish","reason":"tool-calls"}}` + "\n" +
			`{"type":"step_start","sessionID":"ses_1","part":{"id":"prt_4","type":"step-start"}}` + "\n" +
			`{"type":"text","sessionID":"ses_1","part":{"id":"prt_5","type":"text","text":"final answer"}}` + "\n",
	)}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	response, err := client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if response.Text != "final answer" {
		t.Fatalf("Text = %q, want the final answer", response.Text)
	}
	if response.CompletionReason != CompletionReasonUnverified {
		t.Fatalf("CompletionReason = %q, want %q; the tool step's reason is stale", response.CompletionReason, CompletionReasonUnverified)
	}
}

// A re-delivered cumulative text part after step_finish emits no new delta and
// must not erase the terminal reason.
func TestOpenCodeCLIDuplicateTextAfterFinishKeepsReason(t *testing.T) {
	t.Parallel()

	text := `{"type":"text","sessionID":"ses_1","part":{"id":"prt_1","type":"text","text":"answer"}}`
	runner := &fakeCommandRunner{stdout: []byte(
		text + "\n" +
			`{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_2","type":"step-finish","reason":"stop","tokens":{"input":1,"output":1}}}` + "\n" +
			text + "\n",
	)}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	response, err := client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if response.CompletionReason != CompletionReasonStop {
		t.Fatalf("CompletionReason = %q, want %q", response.CompletionReason, CompletionReasonStop)
	}
}

// TestOpenCodeCLIEffortAppearsInResponseModel pins that the effective model is
// the same string the CLI ran: the variant suffix belongs to the response model
// too, so callers do not report a bare id for a variant run.
func TestOpenCodeCLIEffortAppearsInResponseModel(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(openCodeSuccessOutput("ok"))}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	response, err := client.Generate(context.Background(), Request{
		MaxTokens: 32,
		Messages:  []Message{User("q")},
		Effort:    EffortHigh,
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if response.Model != "anthropic/claude-sonnet-4-5#high" {
		t.Fatalf("response model = %q, want the effective model with #high", response.Model)
	}
}

// TestOpenCodeCLITruncatedResponseReportsMaxTokens confirms a run that hits the
// provider's output cap is not reported as a clean stop: the step_finish reason
// reaches the response as CompletionReasonMaxTokens, so a caller can tell a
// truncated answer from a finished one even though text was delivered.
func TestOpenCodeCLITruncatedResponseReportsMaxTokens(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(
		`{"type":"text","sessionID":"ses_1","part":{"id":"prt_1","type":"text","text":"partial ans"}}` + "\n" +
			`{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_2","type":"step-finish","reason":"length",` +
			`"tokens":{"input":19,"output":32000}}}` + "\n",
	)}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	response, err := client.Generate(context.Background(), Request{
		MaxTokens: 32,
		Messages:  []Message{User("Hello")},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if response.Text != "partial ans" {
		t.Fatalf("text = %q, want the truncated answer", response.Text)
	}
	if response.CompletionReason != CompletionReasonMaxTokens {
		t.Fatalf("completion reason = %q, want %q", response.CompletionReason, CompletionReasonMaxTokens)
	}
}

// TestOpenCodeCLITextPartsAreCumulative confirms a re-emitted part id is not
// counted twice, matching opencode's part-snapshot events.
func TestOpenCodeCLITextPartsAreCumulative(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(
		`{"type":"text","sessionID":"ses_1","part":{"id":"prt_1","type":"text","text":"Hello"}}` + "\n" +
			`{"type":"text","sessionID":"ses_1","part":{"id":"prt_1","type":"text","text":"Hello, world"}}` + "\n" +
			openCodeStepFinish + "\n",
	)}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	response, err := client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if response.Text != "Hello, world" {
		t.Fatalf("Text = %q, want %q", response.Text, "Hello, world")
	}
}

func TestOpenCodeCLIStructuredOutput(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(openCodeSuccessOutput(structuredJSON))}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	result, err := client.Generate(context.Background(), Request{
		MaxTokens:    32,
		Messages:     []Message{User("q")},
		OutputSchema: &OutputSchema{Name: "answer", Schema: structuredSchema},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	// opencode run has no schema flag: the schema is stated in the stdin prompt.
	stdin := string(runner.stdin)
	if !strings.Contains(stdin, string(structuredSchema)) {
		t.Fatalf("stdin prompt %q does not carry the schema", stdin)
	}
	if !strings.HasSuffix(stdin, "q") {
		t.Fatalf("stdin prompt %q does not end with the transcript", stdin)
	}
	if result.Text != structuredJSON || string(result.StructuredOutput) != structuredJSON {
		t.Fatalf("Text=%q StructuredOutput=%q, want %q", result.Text, result.StructuredOutput, structuredJSON)
	}
}

func TestOpenCodeCLIStructuredOutputRejectsNonJSON(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(openCodeSuccessOutput("not json"))}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = client.Generate(context.Background(), Request{
		MaxTokens:    32,
		Messages:     []Message{User("q")},
		OutputSchema: &OutputSchema{Name: "answer", Schema: structuredSchema},
	})
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Generate() error = %T %v, want *Error", err, err)
	}
	if target.Operation != "generate" || target.Code != "invalid_structured_output" {
		t.Fatalf("error = %+v, want generate/invalid_structured_output", target)
	}
}

func TestOpenCodeCLIErrors(t *testing.T) {
	t.Parallel()

	launchFailure := errors.New("launch failed")
	tests := []struct {
		name        string
		runner      *fakeCommandRunner
		wantIs      error
		wantStatus  int
		wantCode    string
		wantReason  string
		wantMessage string
	}{
		{
			name:   "missing executable",
			runner: &fakeCommandRunner{err: ErrExecutableNotFound},
			wantIs: ErrExecutableNotFound,
		},
		{
			name:   "launch failure",
			runner: &fakeCommandRunner{err: launchFailure},
			wantIs: launchFailure,
		},
		{
			name:       "nonzero exit",
			runner:     &fakeCommandRunner{stderr: []byte("not authenticated\n"), exitCode: 9},
			wantStatus: 9,
		},
		{
			name: "nonzero exit with error diagnostic on stdout",
			runner: &fakeCommandRunner{
				stdout:   []byte(`{"type":"error","sessionID":"ses_1","error":{"name":"UnknownError","data":{"message":"Unexpected server error."}}}` + "\n"),
				exitCode: 1,
			},
			wantStatus:  1,
			wantCode:    "UnknownError",
			wantMessage: "Unexpected server error.",
		},
		{
			name:   "malformed JSONL",
			runner: &fakeCommandRunner{stdout: []byte("{\"type\":\"step_start\"}\nnot-json\n")},
		},
		{
			name:        "error event",
			runner:      &fakeCommandRunner{stdout: []byte(`{"type":"error","sessionID":"ses_1","error":{"name":"ProviderError","data":{"message":"model unavailable"}}}` + "\n")},
			wantCode:    "ProviderError",
			wantMessage: "model unavailable",
		},
		{
			name: "opencode 2.x error event",
			runner: &fakeCommandRunner{stdout: []byte(
				`{"type":"error","sessionID":"ses_1","error":{"type":"provider.no-route","message":"Variant unavailable for p/m: bogus"}}` + "\n",
			)},
			wantCode:    "provider.no-route",
			wantMessage: "Variant unavailable for p/m: bogus",
		},
		{
			name:     "missing step finish and text",
			runner:   &fakeCommandRunner{stdout: []byte(`{"type":"step_start","sessionID":"ses_1"}` + "\n")},
			wantCode: "missing_step_finish",
		},
		{
			name:       "missing text",
			runner:     &fakeCommandRunner{stdout: []byte(openCodeStepFinish + "\n")},
			wantCode:   "missing_text",
			wantReason: "stop",
		},
		{
			name:       "no text before the output cap",
			runner:     &fakeCommandRunner{stdout: []byte(`{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_1","type":"step-finish","reason":"length"}}` + "\n")},
			wantCode:   "max_tokens",
			wantReason: "length",
		},
		{
			name:       "no text before a content filter",
			runner:     &fakeCommandRunner{stdout: []byte(`{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_1","type":"step-finish","reason":"content-filter"}}` + "\n")},
			wantCode:   "content_filter",
			wantReason: "content-filter",
		},
		{
			name: "tool call ended without text",
			runner: &fakeCommandRunner{stdout: []byte(
				`{"type":"tool_use","sessionID":"ses_1","part":{"type":"tool","tool":"glob","state":{"status":"completed"}}}` + "\n" +
					openCodeToolCallStepFinish + "\n",
			)},
			wantCode:   CodeToolCalls,
			wantReason: "tool-calls",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client, err := newClient(Config{
				Provider: ProviderOpenCode,
				Flavor:   FlavorCLI,
				Model:    "anthropic/claude-sonnet-4-5",
			}, test.runner.run)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			_, err = client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("Hello")}})
			var target *Error
			if !errors.As(err, &target) {
				t.Fatalf("Generate() error = %T %v, want *Error", err, err)
			}
			if target.Provider != ProviderOpenCode || target.Flavor != FlavorCLI || target.Operation != "generate" {
				t.Fatalf("error context = %+v", target)
			}
			if target.Status != test.wantStatus || target.Code != test.wantCode {
				t.Fatalf("status/code = %d/%q, want %d/%q", target.Status, target.Code, test.wantStatus, test.wantCode)
			}
			if target.Reason != test.wantReason {
				t.Fatalf("reason = %q, want %q", target.Reason, test.wantReason)
			}
			if test.wantMessage != "" && target.Message != test.wantMessage {
				t.Fatalf("message = %q, want %q", target.Message, test.wantMessage)
			}
			if test.wantIs != nil && !errors.Is(err, test.wantIs) {
				t.Fatalf("Generate() error = %v, want errors.Is(%v)", err, test.wantIs)
			}
		})
	}
}

func TestOpenCodeCLINonzeroExitPreservesTruncatedJSONDiagnostic(t *testing.T) {
	t.Parallel()

	stdout := `{"type":"error","error":{"name":"ProviderError","data":{"message":"upstream unavailable"}`
	runner := &fakeCommandRunner{stdout: []byte(stdout), exitCode: 17}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("Hello")}})
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Generate() error = %T %v, want *Error", err, err)
	}
	if target.Status != 17 {
		t.Fatalf("status = %d, want %d", target.Status, 17)
	}
	if !strings.Contains(target.Message, "upstream unavailable") {
		t.Fatalf("message = %q, want diagnostic from stdout %q", target.Message, stdout)
	}
}

func TestOpenCodeCLIOmitsModelWhenEmpty(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(openCodeSuccessOutput("ok"))}
	client, err := newClient(Config{Provider: ProviderOpenCode, Flavor: FlavorCLI}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	runner.assertCall(t, "opencode", []string{"run", "--format", "json"}, "q")
}

// openCodeStepFinish is a terminal event with no text, for error-path fixtures.
const openCodeStepFinish = `{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_finish","type":"step-finish","reason":"stop","tokens":{}}}`

// openCodeToolCallStepFinish is a terminal event for a turn the agent ended on a
// tool call, as when a headless run auto-rejects the tool permission.
const openCodeToolCallStepFinish = `{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_finish","type":"step-finish","reason":"tool-calls","tokens":{}}}`

// TestOpenCodeCLIStderrDiagnosticIsPreserved confirms the batch path keeps the
// stderr warning opencode prints when it auto-rejects a tool permission, even
// though the process exits 0. Without it the operator sees only "returned no
// text" with no explanation.
func TestOpenCodeCLIStderrDiagnosticIsPreserved(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{
		stdout: []byte(openCodeToolCallStepFinish + "\n"),
		stderr: []byte("! permission requested: external_directory (/tmp/*); auto-rejecting\n"),
	}
	client, err := newClient(Config{
		Provider: ProviderOpenCode,
		Flavor:   FlavorCLI,
		Model:    "anthropic/claude-sonnet-4-5",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}})
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Generate() error = %T %v, want *Error", err, err)
	}
	if target.Code != CodeToolCalls {
		t.Fatalf("code = %q, want %q", target.Code, CodeToolCalls)
	}
	if !strings.Contains(target.Message, "auto-rejecting") {
		t.Fatalf("message = %q, want it to carry the stderr diagnostic", target.Message)
	}
}

// openCodeSuccessOutput renders a text event plus a terminal step_finish.
func openCodeSuccessOutput(text string) string {
	return `{"type":"text","sessionID":"ses_1","part":{"id":"prt_text","type":"text","text":` + strconv.Quote(text) + `}}` + "\n" +
		openCodeStepFinish + "\n"
}
