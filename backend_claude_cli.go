package llmwrapper

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type claudeCLIBackend struct {
	model        string
	path         string
	args         []string
	runner       commandRunner
	streamRunner streamingCommandRunner
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
	streamRunner := config.streamRunner
	if streamRunner == nil {
		streamRunner = defaultStreamingCommandRunner
	}
	return &claudeCLIBackend{
		model:        config.model,
		path:         path,
		args:         append([]string(nil), config.args...),
		runner:       runner,
		streamRunner: streamRunner,
	}
}

// These flags target the non-interactive `claude -p` contract described in
// SPEC.md. Print mode is already single-shot and non-persistent, and the
// default permission mode keeps a headless run from approving agent tool
// actions. Default tests inject a fake runner and cannot catch a renamed or
// removed flag, so the opt-in `go test -tags integration` smoke test in
// backend_cli_integration_test.go launches the real binary to surface flag
// drift against the installed `claude` version.
func (b *claudeCLIBackend) generate(ctx context.Context, request Request) (*Response, error) {
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

// generate builds the same transcript and system flags as the batch path but
// asks for incremental stream-json output. Print mode requires --verbose for
// stream-json, and --include-partial-messages adds the raw Anthropic SSE
// events the text deltas come from.
func (b *claudeCLIBackend) generateStream(ctx context.Context, request Request) (chunkStream, error) {
	args := []string{
		"-p",
		"--output-format", "stream-json",
		"--include-partial-messages",
		"--verbose",
		"--permission-mode", "default",
		"--model", b.model,
	}
	if request.System != "" {
		args = append(args, "--append-system-prompt", request.System)
	}
	args = append(args, b.args...)

	process, err := startCLIStream(
		ctx,
		ProviderClaude,
		b.path,
		args,
		[]byte(renderTranscript(request.Messages)),
		b.streamRunner,
	)
	if err != nil {
		return nil, err
	}
	return newCLIChunkStream(ProviderClaude, process, &claudeCLIStreamParser{model: b.model}), nil
}

// claudeCLIStreamLine is the stream-json envelope: partial-message lines carry
// a raw Anthropic SSE event, and the terminal line is the batch result object.
type claudeCLIStreamLine struct {
	Type  string `json:"type"`
	Event struct {
		Type  string `json:"type"`
		Delta struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"delta"`
	} `json:"event"`
}

type claudeCLIStreamParser struct {
	model  string
	result *Response
}

func (p *claudeCLIStreamParser) event(line []byte) (string, error) {
	var envelope claudeCLIStreamLine
	if err := json.Unmarshal(line, &envelope); err != nil {
		return "", claudeCLIDecodeError("generate_stream", err)
	}

	switch envelope.Type {
	case "stream_event":
		if envelope.Event.Type == "content_block_delta" && envelope.Event.Delta.Type == "text_delta" {
			return envelope.Event.Delta.Text, nil
		}
	case "result":
		var result claudeCLIResult
		if err := json.Unmarshal(line, &result); err != nil {
			return "", claudeCLIDecodeError("generate_stream", err)
		}
		response, err := claudeCLIResponse(result, p.model, "generate_stream")
		if err != nil {
			return "", err
		}
		p.result = response
	}
	return "", nil
}

func (p *claudeCLIStreamParser) response() (*Response, error) {
	if p.result == nil {
		return nil, &Error{
			Provider:  ProviderClaude,
			Flavor:    FlavorCLI,
			Operation: "generate_stream",
			Code:      "missing_result",
			Message:   "Claude CLI output did not contain a result",
		}
	}
	return p.result, nil
}

func parseClaudeCLIOutput(stdout []byte, model string) (*Response, error) {
	var result claudeCLIResult
	if err := json.Unmarshal(stdout, &result); err != nil {
		return nil, claudeCLIDecodeError("generate", err)
	}
	return claudeCLIResponse(result, model, "generate")
}

func claudeCLIDecodeError(operation string, err error) error {
	return &Error{
		Provider:  ProviderClaude,
		Flavor:    FlavorCLI,
		Operation: operation,
		Err:       fmt.Errorf("decode Claude CLI output: %w", err),
	}
}

func claudeCLIResponse(result claudeCLIResult, model string, operation string) (*Response, error) {
	if result.Type != "result" || result.Subtype != "success" || result.IsError {
		message := result.Result
		if strings.TrimSpace(message) == "" {
			message = "Claude CLI returned an unsuccessful result"
		}
		return nil, &Error{
			Provider:  ProviderClaude,
			Flavor:    FlavorCLI,
			Operation: operation,
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
