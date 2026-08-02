package backends

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/lao/claudewrap/internal/core"
)

const maxCodexEventSize = 16 << 20

type codexCLIBackend struct {
	model  string
	path   string
	args   []string
	runner core.CommandRunner
}

type codexCLIEvent struct {
	Type     string `json:"type"`
	ThreadID string `json:"thread_id"`
	Message  string `json:"message"`
	Error    struct {
		Message string `json:"message"`
	} `json:"error"`
	Item struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"item"`
	Usage struct {
		InputTokens           int `json:"input_tokens"`
		CachedInputTokens     int `json:"cached_input_tokens"`
		CacheWriteInputTokens int `json:"cache_write_input_tokens"`
		OutputTokens          int `json:"output_tokens"`
		ReasoningOutputTokens int `json:"reasoning_output_tokens"`
	} `json:"usage"`
}

func newCodexCLIBackend(config cliBackendConfig) *codexCLIBackend {
	path := config.path
	if path == "" {
		path = "codex"
	}
	runner := config.runner
	if runner == nil {
		runner = defaultCommandRunner
	}
	return &codexCLIBackend{
		model:  config.model,
		path:   path,
		args:   append([]string(nil), config.args...),
		runner: runner,
	}
}

// These flags target the non-interactive `codex exec` contract described in
// docs/SPEC.md (JSONL output, read-only sandbox, no git-repo check). Default
// tests inject a fake runner and cannot catch a renamed or removed flag, so
// the opt-in `go test -tags integration` smoke test in
// cli_integration_test.go launches the real binary to surface flag
// drift against the installed `codex` version.
func (b *codexCLIBackend) Generate(ctx context.Context, request core.Request) (*core.Response, error) {
	args := []string{
		"exec",
		"--json",
		"--sandbox", "read-only",
		"--skip-git-repo-check",
		"--model", b.model,
	}
	if request.System != "" {
		args = append(args, "--config", codexDeveloperInstructions(request.System))
	}
	args = append(args, b.args...)
	args = append(args, "-")

	stdout, err := executeCLI(
		ctx,
		core.ProviderOpenAI,
		b.path,
		args,
		[]byte(renderTranscript(request.Messages)),
		b.runner,
	)
	if err != nil {
		var processError *core.Error
		if errors.As(err, &processError) && processError.Status != 0 {
			_, diagnosticErr := parseCodexCLIOutput(stdout, b.model)
			var diagnostic *core.Error
			if errors.As(diagnosticErr, &diagnostic) && (diagnostic.Code == "turn_failed" || diagnostic.Code == "error") {
				diagnostic.Status = processError.Status
				return nil, diagnostic
			}
		}
		return nil, err
	}
	return parseCodexCLIOutput(stdout, b.model)
}

func codexDeveloperInstructions(system string) string {
	encoded, _ := json.Marshal(system)
	return "developer_instructions=" + string(encoded)
}

func parseCodexCLIOutput(output []byte, model string) (*core.Response, error) {
	var response core.Response
	response.Model = model
	var completed bool
	var hasFinalMessage bool

	scanner := bufio.NewScanner(bytes.NewReader(output))
	scanner.Buffer(make([]byte, 64*1024), maxCodexEventSize)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := scanner.Bytes()
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}

		var event codexCLIEvent
		if err := json.Unmarshal(line, &event); err != nil {
			return nil, codexCLIParseError(fmt.Errorf("decode Codex CLI event %d: %w", lineNumber, err))
		}

		switch event.Type {
		case "thread.started":
			response.ID = event.ThreadID
		case "item.completed":
			if event.Item.Type == "agent_message" {
				response.Text = event.Item.Text
				hasFinalMessage = true
			}
		case "turn.completed":
			completed = true
			response.Usage = core.Usage{
				InputTokens:              event.Usage.InputTokens,
				OutputTokens:             event.Usage.OutputTokens,
				CachedInputTokens:        event.Usage.CachedInputTokens,
				CacheCreationInputTokens: event.Usage.CacheWriteInputTokens,
				ReasoningTokens:          event.Usage.ReasoningOutputTokens,
			}
		case "turn.failed":
			message := strings.TrimSpace(event.Error.Message)
			if message == "" {
				message = "Codex CLI turn failed"
			}
			return nil, codexCLIResultError("turn_failed", message)
		case "error":
			message := strings.TrimSpace(event.Message)
			if message == "" {
				message = "Codex CLI stream failed"
			}
			return nil, codexCLIResultError("error", message)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, codexCLIParseError(fmt.Errorf("scan Codex CLI output: %w", err))
	}
	if !completed {
		return nil, codexCLIResultError("missing_turn_completion", "Codex CLI output did not complete a turn")
	}
	if !hasFinalMessage {
		return nil, codexCLIResultError("missing_final_message", "Codex CLI output did not contain a completed agent message")
	}
	return &response, nil
}

func codexCLIParseError(err error) error {
	return &core.Error{
		Provider:  core.ProviderOpenAI,
		Flavor:    core.FlavorCLI,
		Operation: "generate",
		Err:       err,
	}
}

func codexCLIResultError(code string, message string) error {
	return &core.Error{
		Provider:  core.ProviderOpenAI,
		Flavor:    core.FlavorCLI,
		Operation: "generate",
		Code:      code,
		Message:   strings.TrimSpace(message),
	}
}
