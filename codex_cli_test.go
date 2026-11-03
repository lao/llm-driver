package llmdriver

import (
	"context"
	"errors"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func TestCodexCLICommand(t *testing.T) {
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
				Provider: ProviderOpenAI,
				Flavor:   FlavorCLI,
				Model:    "gpt-test",
			},
			request:    Request{MaxTokens: 32, Messages: []Message{User("Hello from stdin")}},
			wantPath:   "codex",
			wantArgs:   []string{"exec", "--json", "--sandbox", "read-only", "--skip-git-repo-check", "--model", "gpt-test", "-"},
			wantPrompt: "Hello from stdin",
		},
		{
			name: "system and multi turn with custom CLI settings",
			config: Config{
				Provider: ProviderOpenAI,
				Flavor:   FlavorCLI,
				Model:    "gpt-test",
				CLIPath:  "/opt/bin/codex-custom",
				CLIArgs:  []string{"--color", "never"},
			},
			request: Request{
				System:    "Be \"concise\".\nUse x = y.",
				MaxTokens: 32,
				Messages:  []Message{User("First"), Assistant("Second"), User("Third")},
			},
			wantPath: "/opt/bin/codex-custom",
			wantArgs: []string{
				"exec", "--json", "--sandbox", "read-only", "--skip-git-repo-check", "--model", "gpt-test",
				"--config", `developer_instructions="Be \"concise\".\nUse x = y."`,
				"--color", "never", "-",
			},
			wantPrompt: "User: First\n\nAssistant: Second\n\nUser: Third",
		},
		{
			name: "reasoning effort maps to -c model_reasoning_effort",
			config: Config{
				Provider: ProviderOpenAI,
				Flavor:   FlavorCLI,
				Model:    "gpt-test",
			},
			request:  Request{MaxTokens: 32, Messages: []Message{User("q")}, Effort: EffortMax},
			wantPath: "codex",
			wantArgs: []string{
				"exec", "--json", "--sandbox", "read-only", "--skip-git-repo-check", "--model", "gpt-test",
				"-c", "model_reasoning_effort=max", "-",
			},
			wantPrompt: "q",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			runner := &fakeCommandRunner{stdout: []byte(
				"{\"type\":\"thread.started\",\"thread_id\":\"thread-1\"}\n" +
					"{\"type\":\"item.completed\",\"item\":{\"id\":\"item-1\",\"type\":\"agent_message\",\"text\":\"ok\"}}\n" +
					"{\"type\":\"turn.completed\",\"usage\":{}}\n",
			)}
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

func TestCodexCLISuccessSelectsFinalMessageAndUsage(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(`{"type":"thread.started","thread_id":"thread-123"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"reason-1","type":"reasoning","text":"private progress"}}
{"type":"item.completed","item":{"id":"message-1","type":"agent_message","text":"intermediate"}}
{"type":"item.completed","item":{"id":"command-1","type":"command_execution","aggregated_output":"noise"}}
{"type":"item.completed","item":{"id":"message-2","type":"agent_message","text":"Final answer"}}
{"type":"turn.completed","usage":{"input_tokens":19,"cached_input_tokens":11,"cache_write_input_tokens":5,"output_tokens":7,"reasoning_output_tokens":3}}
`)}
	client, err := newClient(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorCLI,
		Model:    "gpt-test",
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

	if response.ID != "thread-123" || response.Model != "gpt-test" || response.Text != "Final answer" {
		t.Fatalf("response = %+v", response)
	}
	if response.Provider != ProviderOpenAI || response.Flavor != FlavorCLI {
		t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
	}
	wantUsage := (Usage{
		InputTokens:              19,
		OutputTokens:             7,
		CachedInputTokens:        11,
		CacheCreationInputTokens: 5,
		ReasoningTokens:          3,
	})
	if response.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", response.Usage, wantUsage)
	}
}

func TestCodexCLIStructuredOutput(t *testing.T) {
	t.Parallel()

	output := `{"type":"thread.started","thread_id":"t"}` + "\n" +
		`{"type":"item.completed","item":{"id":"m","type":"agent_message","text":` + strconv.Quote(structuredJSON) + `}}` + "\n" +
		`{"type":"turn.completed","usage":{}}` + "\n"

	// The runner reads the --output-schema file while the process is "running",
	// so the assertion proves the file exists with the schema at invocation time.
	var schemaPath string
	var schemaOnDisk []byte
	run := func(_ context.Context, _ string, args []string, _ []byte) ([]byte, []byte, int, error) {
		if i := slices.Index(args, "--output-schema"); i >= 0 && i+1 < len(args) {
			schemaPath = args[i+1]
			schemaOnDisk, _ = os.ReadFile(schemaPath)
		}
		return []byte(output), nil, 0, nil
	}

	client, err := newClient(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorCLI,
		Model:    "gpt-test",
	}, run)
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

	if schemaPath == "" {
		t.Fatal("codex was not invoked with --output-schema")
	}
	if string(schemaOnDisk) != string(structuredSchema) {
		t.Fatalf("schema file = %q, want %q", schemaOnDisk, structuredSchema)
	}
	// The temp file is removed once Generate returns.
	if _, statErr := os.Stat(schemaPath); !os.IsNotExist(statErr) {
		t.Fatalf("schema temp file %q still exists after Generate, stat err = %v", schemaPath, statErr)
	}
	if result.Text != structuredJSON || string(result.StructuredOutput) != structuredJSON {
		t.Fatalf("Text=%q StructuredOutput=%q, want %q", result.Text, result.StructuredOutput, structuredJSON)
	}
}

func TestCodexCLIStructuredOutputRejectsNonJSON(t *testing.T) {
	t.Parallel()

	output := `{"type":"thread.started","thread_id":"t"}` + "\n" +
		`{"type":"item.completed","item":{"id":"m","type":"agent_message","text":"not json"}}` + "\n" +
		`{"type":"turn.completed","usage":{}}` + "\n"
	runner := &fakeCommandRunner{stdout: []byte(output)}
	client, err := newClient(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorCLI,
		Model:    "gpt-test",
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

func TestCodexCLIErrors(t *testing.T) {
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
			name:        "nonzero exit with failed turn diagnostic",
			runner:      &fakeCommandRunner{stdout: []byte("{\"type\":\"turn.failed\",\"error\":{\"message\":\"model failed\"}}\n"), exitCode: 9},
			wantStatus:  9,
			wantCode:    "turn_failed",
			wantMessage: "model failed",
		},
		{
			name:   "malformed JSONL",
			runner: &fakeCommandRunner{stdout: []byte("{\"type\":\"turn.started\"}\nnot-json\n")},
		},
		{
			name:     "failed turn",
			runner:   &fakeCommandRunner{stdout: []byte("{\"type\":\"turn.failed\",\"error\":{\"message\":\"model failed\"}}\n")},
			wantCode: "turn_failed",
		},
		{
			name:     "failed turn with blank message",
			runner:   &fakeCommandRunner{stdout: []byte("{\"type\":\"turn.failed\",\"error\":{\"message\":\"   \"}}\n")},
			wantCode: "turn_failed",
		},
		{
			name:     "stream error",
			runner:   &fakeCommandRunner{stdout: []byte("{\"type\":\"error\",\"message\":\"stream disconnected\"}\n")},
			wantCode: "error",
		},
		{
			name:     "stream error with blank message",
			runner:   &fakeCommandRunner{stdout: []byte("{\"type\":\"error\",\"message\":\"   \"}\n")},
			wantCode: "error",
		},
		{
			name: "missing final message",
			runner: &fakeCommandRunner{stdout: []byte(
				"{\"type\":\"item.completed\",\"item\":{\"id\":\"reason-1\",\"type\":\"reasoning\",\"text\":\"thinking\"}}\n" +
					"{\"type\":\"turn.completed\",\"usage\":{}}\n",
			)},
			wantCode: CodeMissingFinalMessage,
		},
		{
			name:     "missing turn completion",
			runner:   &fakeCommandRunner{stdout: []byte("{\"type\":\"item.completed\",\"item\":{\"id\":\"message-1\",\"type\":\"agent_message\",\"text\":\"orphan\"}}\n")},
			wantCode: "missing_turn_completion",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client, err := newClient(Config{
				Provider: ProviderOpenAI,
				Flavor:   FlavorCLI,
				Model:    "gpt-test",
			}, test.runner.run)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			_, err = client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("Hello")}})
			var target *Error
			if !errors.As(err, &target) {
				t.Fatalf("Generate() error = %T %v, want *Error", err, err)
			}
			if target.Provider != ProviderOpenAI || target.Flavor != FlavorCLI || target.Operation != "generate" {
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

func TestCodexCLISandboxOptions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		config   Config
		wantArgs []string
	}{
		{
			name: "sandbox mode override",
			config: Config{
				Provider: ProviderOpenAI,
				Flavor:   FlavorCLI,
				Model:    "gpt-test",
				Codex:    CodexOptions{SandboxMode: "workspace-write"},
			},
			wantArgs: []string{
				"exec", "--json", "--sandbox", "workspace-write", "--skip-git-repo-check", "--model", "gpt-test", "-",
			},
		},
		{
			name: "omit sandbox for a permission profile",
			config: Config{
				Provider: ProviderOpenAI,
				Flavor:   FlavorCLI,
				Model:    "gpt-test",
				CLIArgs:  []string{"--config", "permissions.prbooter_review={}"},
				Codex:    CodexOptions{OmitSandbox: true},
			},
			wantArgs: []string{
				"exec", "--json", "--skip-git-repo-check", "--model", "gpt-test",
				"--config", "permissions.prbooter_review={}", "-",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			runner := &fakeCommandRunner{stdout: []byte(`{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}` + "\n" + `{"type":"turn.completed"}`)}
			client, err := newClient(test.config, runner.run)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}
			if _, err := client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}}); err != nil {
				t.Fatalf("Generate(): %v", err)
			}
			if !slices.Equal(runner.args, test.wantArgs) {
				t.Fatalf("args = %#v, want %#v", runner.args, test.wantArgs)
			}
		})
	}
}

func TestCodexCLIOmitsModelWhenEmpty(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte(`{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}` + "\n" + `{"type":"turn.completed"}`)}
	client, err := newClient(Config{Provider: ProviderOpenAI, Flavor: FlavorCLI}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	runner.assertCall(t, "codex", []string{"exec", "--json", "--sandbox", "read-only", "--skip-git-repo-check", "-"}, "q")
}

// A clean exit can still end on a parse/result error; the CLI's stderr is the
// only explanation and must reach the error.
func TestCodexCLICleanExitCarriesStderr(t *testing.T) {
	t.Parallel()

	runner := &fakeCommandRunner{stdout: []byte("not-json\n"), stderr: []byte("permission denied\n")}
	client, err := newClient(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorCLI,
		Model:    "gpt-test",
	}, runner.run)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = client.Generate(context.Background(), Request{MaxTokens: 32, Messages: []Message{User("q")}})
	if err == nil {
		t.Fatal("Generate() succeeded, want a parse error")
	}
	if !strings.Contains(err.Error(), "permission denied") {
		t.Fatalf("error = %q, want the stderr diagnostic", err)
	}
}
