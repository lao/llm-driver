package llmwrapper

import (
	"context"
	"encoding/json"
	"fmt"
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

func newClaudeCLIBackend(config Config) *claudeCLIBackend {
	path := config.CLIPath
	if path == "" {
		path = "claude"
	}
	runner := config.runner
	if runner == nil {
		runner = defaultCommandRunner
	}
	return &claudeCLIBackend{
		model:  config.Model,
		path:   path,
		args:   append([]string(nil), config.CLIArgs...),
		runner: runner,
	}
}

func (b *claudeCLIBackend) generate(ctx context.Context, request Request) (*Response, error) {
	args := []string{
		"-p",
		"--output-format", "json",
		"--no-session-persistence",
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
		if message == "" {
			message = "Claude CLI returned an unsuccessful result"
		}
		code := result.Subtype
		if code == "" {
			code = result.Type
		}
		return nil, &Error{
			Provider:  ProviderClaude,
			Flavor:    FlavorCLI,
			Operation: "generate",
			Code:      code,
			Message:   message,
		}
	}

	return &Response{
		ID:    result.SessionID,
		Model: b.model,
		Text:  result.Result,
		Usage: Usage{
			InputTokens:              result.Usage.InputTokens,
			OutputTokens:             result.Usage.OutputTokens,
			CachedInputTokens:        result.Usage.CacheReadInputTokens,
			CacheCreationInputTokens: result.Usage.CacheCreationInputTokens,
		},
	}, nil
}
