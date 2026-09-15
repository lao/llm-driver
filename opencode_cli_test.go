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
			name: "reasoning effort maps to --variant",
			config: Config{
				Provider: ProviderOpenCode,
				Flavor:   FlavorCLI,
				Model:    "anthropic/claude-sonnet-4-5",
			},
			request:    Request{MaxTokens: 32, Messages: []Message{User("q")}, Effort: EffortHigh},
			wantPath:   "opencode",
			wantArgs:   []string{"run", "--format", "json", "--model", "anthropic/claude-sonnet-4-5", "--variant", "high"},
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
			name:     "missing step finish",
			runner:   &fakeCommandRunner{stdout: []byte(`{"type":"text","sessionID":"ses_1","part":{"id":"p","type":"text","text":"orphan"}}` + "\n")},
			wantCode: "missing_step_finish",
		},
		{
			name:     "missing text",
			runner:   &fakeCommandRunner{stdout: []byte(openCodeStepFinish + "\n")},
			wantCode: "missing_text",
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
			if test.wantMessage != "" && target.Message != test.wantMessage {
				t.Fatalf("message = %q, want %q", target.Message, test.wantMessage)
			}
			if test.wantIs != nil && !errors.Is(err, test.wantIs) {
				t.Fatalf("Generate() error = %v, want errors.Is(%v)", err, test.wantIs)
			}
		})
	}
}

// openCodeStepFinish is a terminal event with no text, for error-path fixtures.
const openCodeStepFinish = `{"type":"step_finish","sessionID":"ses_1","part":{"id":"prt_finish","type":"step-finish","reason":"stop","tokens":{}}}`

// openCodeSuccessOutput renders a text event plus a terminal step_finish.
func openCodeSuccessOutput(text string) string {
	return `{"type":"text","sessionID":"ses_1","part":{"id":"prt_text","type":"text","text":` + strconv.Quote(text) + `}}` + "\n" +
		openCodeStepFinish + "\n"
}
