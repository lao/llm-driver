package main

import (
	"errors"
	"flag"
	"reflect"
	"strings"
	"testing"

	llmdriver "github.com/lao/llm-driver"
)

func TestParseOptionsBuildsConfigAndRequest(t *testing.T) {
	t.Parallel()

	parsed, err := parseOptions([]string{
		"--provider", "openai",
		"--flavor", "cli",
		"--model", "gpt-test",
		"--prompt", "Say hello",
		"--max-tokens", "256",
	}, new(strings.Builder))
	if err != nil {
		t.Fatalf("parseOptions(): %v", err)
	}

	wantConfig := (llmdriver.Config{
		Provider: llmdriver.ProviderOpenAI,
		Flavor:   llmdriver.FlavorCLI,
		Model:    "gpt-test",
	})
	if !reflect.DeepEqual(parsed.config, wantConfig) {
		t.Fatalf("config = %+v, want %+v", parsed.config, wantConfig)
	}
	wantRequest := (llmdriver.Request{
		Messages:  []llmdriver.Message{llmdriver.User("Say hello")},
		MaxTokens: 256,
	})
	got := parsed.request.Messages
	if len(got) != 1 || got[0].Role != wantRequest.Messages[0].Role || got[0].Text != wantRequest.Messages[0].Text || parsed.request.MaxTokens != wantRequest.MaxTokens {
		t.Fatalf("request = %+v, want %+v", parsed.request, wantRequest)
	}
}

func TestParseOptionsAcceptsOpenCode(t *testing.T) {
	t.Parallel()

	parsed, err := parseOptions([]string{
		"--provider", "opencode",
		"--flavor", "cli",
		"--model", "anthropic/claude-sonnet-4-5",
		"--prompt", "Say hello",
	}, new(strings.Builder))
	if err != nil {
		t.Fatalf("parseOptions(): %v", err)
	}
	if parsed.config.Provider != llmdriver.ProviderOpenCode || parsed.config.Flavor != llmdriver.FlavorCLI {
		t.Fatalf("config = %+v", parsed.config)
	}
}

func TestParseOptionsRejectsOpenCodeAPIFlavor(t *testing.T) {
	t.Parallel()

	_, err := parseOptions([]string{
		"--provider", "opencode",
		"--flavor", "api",
		"--model", "model",
		"--prompt", "hello",
	}, new(strings.Builder))
	if err == nil || err.Error() != "opencode provider supports only --flavor cli" {
		t.Fatalf("parseOptions() error = %v, want opencode provider supports only --flavor cli", err)
	}
}

func TestParseOptionsUsesDocumentedTokenDefault(t *testing.T) {
	t.Parallel()

	parsed, err := parseOptions([]string{
		"--provider", "claude",
		"--flavor", "api",
		"--model", "claude-test",
		"--prompt", "Say hello",
	}, new(strings.Builder))
	if err != nil {
		t.Fatalf("parseOptions(): %v", err)
	}
	if parsed.request.MaxTokens != defaultMaxTokens {
		t.Fatalf("MaxTokens = %d, want %d", parsed.request.MaxTokens, defaultMaxTokens)
	}
}

func TestParseOptionsRejectsInvalidArguments(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		args []string
	}{
		{name: "missing provider", args: []string{"--flavor", "api", "--model", "model", "--prompt", "hello"}},
		{name: "invalid provider", args: []string{"--provider", "other", "--flavor", "api", "--model", "model", "--prompt", "hello"}},
		{name: "missing flavor", args: []string{"--provider", "claude", "--model", "model", "--prompt", "hello"}},
		{name: "invalid flavor", args: []string{"--provider", "claude", "--flavor", "other", "--model", "model", "--prompt", "hello"}},
		{name: "missing model", args: []string{"--provider", "claude", "--flavor", "api", "--prompt", "hello"}},
		{name: "missing prompt", args: []string{"--provider", "claude", "--flavor", "api", "--model", "model"}},
		{name: "invalid max tokens", args: []string{"--provider", "claude", "--flavor", "api", "--model", "model", "--prompt", "hello", "--max-tokens", "0"}},
		{name: "positional argument", args: []string{"--provider", "claude", "--flavor", "api", "--model", "model", "--prompt", "hello", "extra"}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			_, err := parseOptions(test.args, new(strings.Builder))
			if err == nil {
				t.Fatal("parseOptions() error = nil")
			}
		})
	}
}

func TestParseOptionsHelpDescribesAuthentication(t *testing.T) {
	t.Parallel()

	var output strings.Builder
	_, err := parseOptions([]string{"--help"}, &output)
	if !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("parseOptions() error = %v, want flag.ErrHelp", err)
	}
	for _, want := range []string{"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "local claude, codex, or opencode login", "--max-tokens"} {
		if !strings.Contains(output.String(), want) {
			t.Errorf("help missing %q:\n%s", want, output.String())
		}
	}
}

func TestRunReturnsArgumentErrors(t *testing.T) {
	t.Parallel()

	err := run(t.Context(), nil, new(strings.Builder), new(strings.Builder))
	if err == nil || !strings.Contains(err.Error(), "--provider") {
		t.Fatalf("run() error = %v", err)
	}
}

func TestExitCodeReportsInvalidArguments(t *testing.T) {
	t.Parallel()

	var stderr strings.Builder
	got := exitCode(t.Context(), nil, new(strings.Builder), &stderr)
	if got != 1 {
		t.Fatalf("exitCode() = %d, want 1", got)
	}
	if stderr.String() != "error: --provider is required\n" {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestParseOptionsStreamFlag(t *testing.T) {
	t.Parallel()

	base := []string{"--provider", "claude", "--flavor", "cli", "--model", "claude-test", "--prompt", "hello"}
	parsed, err := parseOptions(base, new(strings.Builder))
	if err != nil {
		t.Fatalf("parseOptions(): %v", err)
	}
	if parsed.stream {
		t.Fatal("stream = true without --stream")
	}

	parsed, err = parseOptions(append(base, "--stream"), new(strings.Builder))
	if err != nil {
		t.Fatalf("parseOptions(--stream): %v", err)
	}
	if !parsed.stream {
		t.Fatal("stream = false with --stream")
	}
}

func TestParseOptionsHelpDescribesStreaming(t *testing.T) {
	t.Parallel()

	var output strings.Builder
	_, _ = parseOptions([]string{"--help"}, &output)
	if !strings.Contains(output.String(), "--stream") {
		t.Fatalf("help missing --stream:\n%s", output.String())
	}
}
