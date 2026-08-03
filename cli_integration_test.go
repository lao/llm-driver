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

	request := Request{
		System:    "Answer with a single short word.",
		MaxTokens: 64,
		Messages:  []Message{User("Reply with the word ready.")},
	}

	response, err := client.Generate(ctx, request)
	if err != nil {
		reportCLIFailure(t, binary, "Generate", err)
	}
	if strings.TrimSpace(response.Text) == "" {
		t.Fatalf("%s CLI returned empty text", binary)
	}

	// Streaming adds flags the batch path never passes (stream-json output,
	// partial messages, verbose), so it needs its own live invocation for flag
	// drift to surface here rather than at runtime.
	stream, err := client.GenerateStream(ctx, request)
	if err != nil {
		reportCLIFailure(t, binary, "GenerateStream", err)
	}
	defer stream.Close()

	var streamed strings.Builder
	for stream.Next() {
		streamed.WriteString(stream.Current().Text)
	}
	if err := stream.Err(); err != nil {
		reportCLIFailure(t, binary, "GenerateStream", err)
	}
	streamResponse, err := stream.Response()
	if err != nil {
		reportCLIFailure(t, binary, "GenerateStream", err)
	}
	if strings.TrimSpace(streamed.String()) == "" {
		t.Fatalf("%s CLI streamed no text", binary)
	}
	if streamResponse.Text != streamed.String() {
		t.Fatalf("%s CLI stream response text = %q, want the concatenated chunks %q",
			binary, streamResponse.Text, streamed.String())
	}
}

func reportCLIFailure(t *testing.T, binary, operation string, err error) {
	t.Helper()

	var cliError *Error
	if errors.As(err, &cliError) && mentionsUnknownFlag(cliError) {
		t.Fatalf("%s CLI rejected a flag (flag drift) during %s: %v", binary, operation, err)
	}
	t.Fatalf("%s CLI %s(): %v", binary, operation, err)
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
