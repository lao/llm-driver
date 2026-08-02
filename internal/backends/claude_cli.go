package backends

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lao/llmwrapper/internal/core"
)

type claudeCLIBackend struct {
	model  string
	path   string
	args   []string
	runner core.CommandRunner
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
// docs/SPEC.md. Print mode is already single-shot and non-persistent, and the
// default permission mode keeps a headless run from approving agent tool
// actions. Default tests inject a fake runner and cannot catch a renamed or
// removed flag, so the opt-in `go test -tags integration` smoke test in the
// root package's cli_integration_test.go launches the real binary to surface
// flag drift against the installed `claude` version.
func (b *claudeCLIBackend) Generate(ctx context.Context, request core.Request) (*core.Response, error) {
	args := []string{
		"-p",
		"--output-format", "json",
		"--permission-mode", "default",
		"--model", b.model,
	}
	if request.System != "" {
		args = append(args, "--append-system-prompt", request.System)
	}
	args = append(args, b.args...)

	stdout, err := executeCLI(
		ctx,
		core.ProviderClaude,
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

func parseClaudeCLIOutput(stdout []byte, model string) (*core.Response, error) {
	var result claudeCLIResult
	if err := json.Unmarshal(stdout, &result); err != nil {
		return nil, &core.Error{
			Provider:  core.ProviderClaude,
			Flavor:    core.FlavorCLI,
			Operation: "generate",
			Err:       fmt.Errorf("decode Claude CLI output: %w", err),
		}
	}
	if result.Type != "result" || result.Subtype != "success" || result.IsError {
		message := result.Result
		if strings.TrimSpace(message) == "" {
			message = "Claude CLI returned an unsuccessful result"
		}
		return nil, &core.Error{
			Provider:  core.ProviderClaude,
			Flavor:    core.FlavorCLI,
			Operation: "generate",
			Code:      claudeCLIErrorCode(result),
			Message:   message,
		}
	}

	return &core.Response{
		ID:    result.SessionID,
		Model: model,
		Text:  result.Result,
		Usage: core.Usage{
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
