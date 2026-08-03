//go:build integration

package llmwrapper

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// These smoke tests are excluded from the default offline suite and only run
// under `go test -tags integration`. They launch the real `claude` and `codex`
// binaries so that a renamed, removed, or invalid flag surfaces as a live
// failure instead of silently passing every fake-runner test. Each test skips
// when the binary is missing or when the model environment variable is unset,
// so opting in requires a working local login.
//
//	LLMWRAPPER_CLAUDE_CLI_MODEL   model passed to `claude -p`
//	LLMWRAPPER_CODEX_CLI_MODEL    model passed to `codex exec`

// unknownFlagMarkers are substrings the tested CLIs emit when they reject an
// argument. They let the smoke tests report flag drift explicitly rather than
// as a generic non-zero exit.
var unknownFlagMarkers = []string{
	"unknown flag",
	"unknown option",
	"unexpected argument",
	"unrecognized",
	"invalid option",
	"unknown argument",
}

func TestClaudeCLIIntegration(t *testing.T) {
	runCLIIntegrationSmoke(t, ProviderClaude, "claude", "LLMWRAPPER_CLAUDE_CLI_MODEL")
}

func TestCodexCLIIntegration(t *testing.T) {
	runCLIIntegrationSmoke(t, ProviderOpenAI, "codex", "LLMWRAPPER_CODEX_CLI_MODEL")
}

func runCLIIntegrationSmoke(t *testing.T, provider Provider, binary, modelEnv string) {
	t.Helper()

	if _, err := exec.LookPath(binary); err != nil {
		t.Skipf("%s CLI not installed: %v", binary, err)
	}
	model := os.Getenv(modelEnv)
	if model == "" {
		t.Skipf("set %s to run the %s CLI integration smoke test", modelEnv, binary)
	}

	client, err := New(Config{
		Provider: provider,
		Flavor:   FlavorCLI,
		Model:    model,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	response, err := client.Generate(ctx, Request{
		System:    "Answer with a single short word.",
		MaxTokens: 64,
		Messages:  []Message{User("Reply with the word ready.")},
	})
	if err != nil {
		var cliError *Error
		if errors.As(err, &cliError) && mentionsUnknownFlag(cliError) {
			t.Fatalf("%s CLI rejected a flag (flag drift): %v", binary, err)
		}
		t.Fatalf("%s CLI Generate(): %v", binary, err)
	}
	if strings.TrimSpace(response.Text) == "" {
		t.Fatalf("%s CLI returned empty text", binary)
	}
}

func mentionsUnknownFlag(cliError *Error) bool {
	haystack := strings.ToLower(cliError.Message)
	if cliError.Err != nil {
		haystack += " " + strings.ToLower(cliError.Err.Error())
	}
	for _, marker := range unknownFlagMarkers {
		if strings.Contains(haystack, marker) {
			return true
		}
	}
	return false
}
