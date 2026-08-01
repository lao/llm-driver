package llmwrapper

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type claudeCLIBackend struct {
	model  string
	path   string
	args   []string
	runner commandRunner
}

type claudeCLIResult struct {
	Type      string `json:"type"`
	Subtype   string `json:"subtype"`
	IsError   bool   `json:"is_error"`
	Result    string `json:"result"`
	SessionID string `json:"session_id"`
	Usage     struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	} `json:"usage"`
}

func newClaudeCLIBackend(config cliBackendConfig) *claudeCLIBackend {
	path := config.path
	if path == "" {
		path = "claude"
	}
	runner := config.runner
	if runner == nil {
		runner = defaultCommandRunner
	}
	return &claudeCLIBackend{
		model:  config.model,
		path:   path,
		args:   append([]string(nil), config.args...),
		runner: runner,
	}
}

// These flags target the non-interactive `claude -p` contract described in
// SPEC.md (JSON output, no session persistence, safe mode, tools disabled).
// Default tests inject a fake runner and cannot catch a renamed or removed
// flag, so the opt-in `go test -tags integration` smoke test in
// backend_cli_integration_test.go launches the real binary to surface flag
// drift against the installed `claude` version.
func (b *claudeCLIBackend) generate(ctx context.Context, request Request) (*Response, error) {
	args := []string{
		"-p",
		"--output-format", "json",
		"--no-session-persistence",
		"--safe-mode",
		"--tools", "",
		"--model", b.model,
	}
	if request.System != "" {
		args = append(args, "--append-system-prompt", request.System)
	}
	args = append(args, b.args...)

	stdout, err := executeCLI(
		ctx,
		ProviderClaude,
		b.path,
		args,
		[]byte(renderTranscript(request.Messages)),
		b.runner,
	)
	if err != nil {
		return nil, err
	}
	return parseClaudeCLIOutput(stdout, b.model)
}

func parseClaudeCLIOutput(stdout []byte, model string) (*Response, error) {
	var result claudeCLIResult
	if err := json.Unmarshal(stdout, &result); err != nil {
		return nil, &Error{
			Provider:  ProviderClaude,
			Flavor:    FlavorCLI,
			Operation: "generate",
			Err:       fmt.Errorf("decode Claude CLI output: %w", err),
		}
	}
	if result.Type != "result" || result.Subtype != "success" || result.IsError {
		message := result.Result
		if strings.TrimSpace(message) == "" {
			message = "Claude CLI returned an unsuccessful result"
		}
		return nil, &Error{
			Provider:  ProviderClaude,
			Flavor:    FlavorCLI,
			Operation: "generate",
			Code:      claudeCLIErrorCode(result),
			Message:   message,
		}
	}

	return &Response{
		ID:    result.SessionID,
		Model: model,
		Text:  result.Result,
		Usage: Usage{
			InputTokens:              result.Usage.InputTokens,
			OutputTokens:             result.Usage.OutputTokens,
			CachedInputTokens:        result.Usage.CacheReadInputTokens,
			CacheCreationInputTokens: result.Usage.CacheCreationInputTokens,
		},
	}, nil
}

func claudeCLIErrorCode(result claudeCLIResult) string {
	if result.Subtype != "" && result.Subtype != "success" {
		return result.Subtype
	}
	if result.IsError {
		return "error"
	}
	if result.Type != "" && result.Type != "result" {
		return result.Type
	}
	return "unsuccessful_result"
}
