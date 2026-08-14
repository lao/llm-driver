package llmdriver

import (
	"context"
	"errors"
	"strconv"
	"testing"
)

func TestClaudeCLICommand(t *testing.T) {
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
				Provider: ProviderClaude,
				Flavor:   FlavorCLI,
				Model:    "claude-test",
			},
			request:    Request{MaxTokens: 32, Messages: []Message{User("Hello from stdin")}},
			wantPath:   "claude",
			wantArgs:   []string{"-p", "--output-format", "json", "--permission-mode", "default", "--model", "claude-test"},
			wantPrompt: "Hello from stdin",
		},
		{
			name: "system and multi turn with custom CLI settings",
			config: Config{
				Provider: ProviderClaude,
				Flavor:   FlavorCLI,
				Model:    "claude-test",
				CLIPath:  "/opt/bin/claude-custom",
				CLIArgs:  []string{"--permission-mode", "plan"},
			},
			request: Request{
				System:    "Be concise.",
				MaxTokens: 32,
				Messages:  []Message{User("First"), Assistant("Second"), User("Third")},
			},
			wantPath: "/opt/bin/claude-custom",
			wantArgs: []string{
				"-p", "--output-format", "json", "--permission-mode", "default", "--model", "claude-test",
				"--append-system-prompt", "Be concise.",
				"--permission-mode", "plan",
			},
			wantPrompt: "User: First\n\nAssistant: Second\n\nUser: Third",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			runner := &fakeCommandRunner{stdout: []byte(`{"type":"result","subtype":"success","result":"ok"}`)}
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

func TestClaudeCLISuccess(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(`{
  "type":"result",
  "subtype":"success",
  "is_error":false,
  "result":"Hello from Claude",
  "session_id":"session-123",
  "usage":{
    "input_tokens":11,
    "output_tokens":7,
    "cache_read_input_tokens":5,
    "cache_creation_input_tokens":3
  }
}`)}
	client, err := newClient(Config{
		Provider: ProviderClaude,
		Flavor:   FlavorCLI,
		Model:    "claude-test",
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

	if response.ID != "session-123" || response.Model != "claude-test" || response.Text != "Hello from Claude" {
		t.Fatalf("response = %+v", response)
	}
	if response.Provider != ProviderClaude || response.Flavor != FlavorCLI {
		t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
	}
	wantUsage := Usage{InputTokens: 11, OutputTokens: 7, CachedInputTokens: 5, CacheCreationInputTokens: 3}
	if response.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", response.Usage, wantUsage)
	}
}

func TestClaudeCLIStructuredOutput(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(
		`{"type":"result","subtype":"success","result":` + strconv.Quote(structuredJSON) + `,"session_id":"s"}`)}
	client, err := newClient(Config{
		Provider: ProviderClaude,
		Flavor:   FlavorCLI,
		Model:    "claude-test",
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

	// The schema is passed inline via --json-schema.
	runner.assertCall(t, "claude", []string{
		"-p", "--output-format", "json", "--permission-mode", "default", "--model", "claude-test",
		"--json-schema", string(structuredSchema),
	}, "q")

	if result.Text != structuredJSON || string(result.StructuredOutput) != structuredJSON {
		t.Fatalf("Text=%q StructuredOutput=%q, want %q", result.Text, result.StructuredOutput, structuredJSON)
	}
}

func TestClaudeCLIStructuredOutputRejectsNonJSON(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(`{"type":"result","subtype":"success","result":"not json","session_id":"s"}`)}
	client, err := newClient(Config{
		Provider: ProviderClaude,
		Flavor:   FlavorCLI,
		Model:    "claude-test",
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

func TestClaudeCLIErrors(t *testing.T) {
	t.Parallel()

	launchFailure := errors.New("launch failed")
	tests := []struct {
		name       string
		runner     *fakeCommandRunner
		wantIs     error
		wantStatus int
		wantCode   string
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
			runner:     &fakeCommandRunner{stderr: []byte("authentication failed\n"), exitCode: 17},
			wantStatus: 17,
		},
		{
			name:   "malformed JSON",
			runner: &fakeCommandRunner{stdout: []byte("not-json")},
		},
		{
			name:     "non-success subtype",
			runner:   &fakeCommandRunner{stdout: []byte(`{"type":"result","subtype":"error_max_turns","result":"turn limit reached"}`)},
			wantCode: "error_max_turns",
		},
		{
			name:     "error result",
			runner:   &fakeCommandRunner{stdout: []byte(`{"type":"result","subtype":"success","is_error":true,"result":"provider rejected request"}`)},
			wantCode: "error",
		},
		{
			name:     "unexpected result type",
			runner:   &fakeCommandRunner{stdout: []byte(`{"type":"unexpected","subtype":"success","result":"wrong payload"}`)},
			wantCode: "unexpected",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client, err := newClient(Config{
				Provider: ProviderClaude,
				Flavor:   FlavorCLI,
				Model:    "claude-test",
			}, test.runner.run)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			_, err = client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("Hello")}})
			var target *Error
			if !errors.As(err, &target) {
				t.Fatalf("Generate() error = %T %v, want *Error", err, err)
			}
			if target.Provider != ProviderClaude || target.Flavor != FlavorCLI || target.Operation != "generate" {
				t.Fatalf("error context = %+v", target)
			}
			if target.Status != test.wantStatus || target.Code != test.wantCode {
				t.Fatalf("status/code = %d/%q, want %d/%q", target.Status, target.Code, test.wantStatus, test.wantCode)
			}
			if test.wantIs != nil && !errors.Is(err, test.wantIs) {
				t.Fatalf("Generate() error = %v, want errors.Is(%v)", err, test.wantIs)
			}
		})
	}
}
