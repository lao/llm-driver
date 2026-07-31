package llmwrapper

import (
	"context"
	"errors"
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
			wantArgs:   []string{"exec", "--json", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check", "--model", "gpt-test", "-"},
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
				System:    "Be concise.",
				MaxTokens: 32,
				Messages:  []Message{User("First"), Assistant("Second"), User("Third")},
			},
			wantPath: "/opt/bin/codex-custom",
			wantArgs: []string{
				"exec", "--json", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check", "--model", "gpt-test",
				"--color", "never", "-",
			},
			wantPrompt: "System: Be concise.\n\nUser: First\n\nAssistant: Second\n\nUser: Third",
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			runner := &fakeCommandRunner{stdout: []byte(
				"{\"type\":\"thread.started\",\"thread_id\":\"thread-1\"}\n" +
					"{\"type\":\"item.completed\",\"item\":{\"id\":\"item-1\",\"type\":\"agent_message\",\"text\":\"ok\"}}\n" +
					"{\"type\":\"turn.completed\",\"usage\":{}}\n",
			)}
			test.config.runner = runner.run
			client, err := New(test.config)
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
	client, err := New(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorCLI,
		Model:    "gpt-test",
		runner:   runner.run,
	})
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

func TestCodexCLIErrors(t *testing.T) {
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
			runner:     &fakeCommandRunner{stderr: []byte("not authenticated\n"), exitCode: 9},
			wantStatus: 9,
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
			name:     "stream error",
			runner:   &fakeCommandRunner{stdout: []byte("{\"type\":\"error\",\"message\":\"stream disconnected\"}\n")},
			wantCode: "error",
		},
		{
			name: "missing final message",
			runner: &fakeCommandRunner{stdout: []byte(
				"{\"type\":\"item.completed\",\"item\":{\"id\":\"reason-1\",\"type\":\"reasoning\",\"text\":\"thinking\"}}\n" +
					"{\"type\":\"turn.completed\",\"usage\":{}}\n",
			)},
			wantCode: "missing_final_message",
		},
		{
			name:     "missing turn completion",
			runner:   &fakeCommandRunner{stdout: []byte("{\"type\":\"item.completed\",\"item\":{\"id\":\"message-1\",\"type\":\"agent_message\",\"text\":\"orphan\"}}\n")},
			wantCode: "missing_turn_completion",
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client, err := New(Config{
				Provider: ProviderOpenAI,
				Flavor:   FlavorCLI,
				Model:    "gpt-test",
				runner:   test.runner.run,
			})
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
			if test.wantIs != nil && !errors.Is(err, test.wantIs) {
				t.Fatalf("Generate() error = %v, want errors.Is(%v)", err, test.wantIs)
			}
		})
	}
}
