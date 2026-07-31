package main

import (
	"errors"
	"flag"
	"reflect"
	"strings"
	"testing"

	"github.com/lao/llmwrapper"
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

	wantConfig := (llmwrapper.Config{
		Provider: llmwrapper.ProviderOpenAI,
		Flavor:   llmwrapper.FlavorCLI,
		Model:    "gpt-test",
	})
	if !reflect.DeepEqual(parsed.config, wantConfig) {
		t.Fatalf("config = %+v, want %+v", parsed.config, wantConfig)
	}
	wantRequest := (llmwrapper.Request{
		Messages:  []llmwrapper.Message{llmwrapper.User("Say hello")},
		MaxTokens: 256,
	})
	if len(parsed.request.Messages) != 1 || parsed.request.Messages[0] != wantRequest.Messages[0] || parsed.request.MaxTokens != wantRequest.MaxTokens {
		t.Fatalf("request = %+v, want %+v", parsed.request, wantRequest)
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
		test := test
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
	for _, want := range []string{"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "local claude or codex login", "--max-tokens"} {
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
