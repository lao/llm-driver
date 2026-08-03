package backends

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/lao/llm-driver/internal/core"
)

const maxCodexEventSize = 16 << 20

type codexCLIBackend struct {
	cliBackendConfig
}

type codexCLIEvent struct {
	Type     string `json:"type"`
	ThreadID string `json:"thread_id"`
	Message  string `json:"message"`
	Error    struct {
		Message string `json:"message"`
	} `json:"error"`
	Delta string `json:"delta"`
	Item  struct {
		ID   string `json:"id"`
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
	return &codexCLIBackend{withCLIDefaults(config, "codex")}
}

// These flags target the non-interactive `codex exec` contract described in
// docs/SPEC.md (JSONL output, read-only sandbox, no git-repo check). Default
// tests inject a fake runner and cannot catch a renamed or removed flag, so
// the opt-in `go test -tags integration` smoke test in the root package's
// cli_integration_test.go launches the real binary to surface flag drift
// against the installed `codex` version.
func (b *codexCLIBackend) Generate(ctx context.Context, request core.Request) (*core.Response, error) {
	args := b.commandArgs(request)

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

func (b *codexCLIBackend) commandArgs(request core.Request) []string {
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
	return append(args, "-")
}

// GenerateStream reuses the batch invocation and parses the same JSONL events
// as they arrive instead of after the process exits.
func (b *codexCLIBackend) GenerateStream(ctx context.Context, request core.Request) (core.ChunkStream, error) {
	process, err := startCLIStream(
		ctx,
		core.ProviderOpenAI,
		b.path,
		b.commandArgs(request),
		[]byte(renderTranscript(request.Messages)),
		b.streamRunner,
	)
	if err != nil {
		return nil, err
	}
	return newCLIChunkStream(core.ProviderOpenAI, process, &codexCLIStreamParser{model: b.model}), nil
}

// codexCLIStreamParser tolerates both Codex CLI shapes: versions that publish
// an agent message incrementally and versions that publish it only on
// completion. Item text is treated as cumulative and an explicit delta field as
// an increment, so text is never emitted twice.
type codexCLIStreamParser struct {
	model           string
	result          core.Response
	emitted         map[string]int
	completed       bool
	hasFinalMessage bool
}

func (p *codexCLIStreamParser) event(line []byte) (string, error) {
	var event codexCLIEvent
	if err := json.Unmarshal(line, &event); err != nil {
		return "", codexCLIParseError("generate_stream", fmt.Errorf("decode Codex CLI event: %w", err))
	}

	switch event.Type {
	case "thread.started":
		p.result.ID = event.ThreadID
	case "turn.completed":
		p.completed = true
		p.result.Usage = codexCLIUsage(event)
	case "turn.failed", "error":
		return "", codexCLIFailureError(event, "generate_stream")
	default:
		// Unknown event types are ignored, as in the batch parser.
		if !strings.HasPrefix(event.Type, "item.") || event.Item.Type != "agent_message" {
			return "", nil
		}
		if event.Type == "item.completed" {
			p.hasFinalMessage = true
		}
		seen := p.emitted[event.Item.ID]
		if event.Delta != "" {
			p.record(event.Item.ID, seen+len(event.Delta))
			return event.Delta, nil
		}
		if seen >= len(event.Item.Text) {
			return "", nil
		}
		// The cumulative branch assumes prior events are byte-aligned so the
		// already-emitted prefix ends on a rune boundary of the current text.
		// A misaligned event would split a UTF-8 rune; enforce the assumption
		// rather than emit corrupted output.
		if !utf8.RuneStart(event.Item.Text[seen]) {
			return "", codexCLIParseError("generate_stream", fmt.Errorf("Codex CLI agent message %q is not byte-aligned at offset %d", event.Item.ID, seen))
		}
		p.record(event.Item.ID, len(event.Item.Text))
		return event.Item.Text[seen:], nil
	}
	return "", nil
}

// record marks how much of an item's text has been emitted.
func (p *codexCLIStreamParser) record(id string, total int) {
	if p.emitted == nil {
		p.emitted = make(map[string]int)
	}
	p.emitted[id] = total
}

func (p *codexCLIStreamParser) response() (*core.Response, error) {
	if err := codexCLIOutcomeError("generate_stream", p.completed, p.hasFinalMessage); err != nil {
		return nil, err
	}
	response := p.result
	response.Model = p.model
	return &response, nil
}

func codexCLIUsage(event codexCLIEvent) core.Usage {
	return core.Usage{
		InputTokens:              event.Usage.InputTokens,
		OutputTokens:             event.Usage.OutputTokens,
		CachedInputTokens:        event.Usage.CachedInputTokens,
		CacheCreationInputTokens: event.Usage.CacheWriteInputTokens,
		ReasoningTokens:          event.Usage.ReasoningOutputTokens,
	}
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
			return nil, codexCLIParseError("generate", fmt.Errorf("decode Codex CLI event %d: %w", lineNumber, err))
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
			response.Usage = codexCLIUsage(event)
		case "turn.failed", "error":
			return nil, codexCLIFailureError(event, "generate")
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, codexCLIParseError("generate", fmt.Errorf("scan Codex CLI output: %w", err))
	}
	if err := codexCLIOutcomeError("generate", completed, hasFinalMessage); err != nil {
		return nil, err
	}
	return &response, nil
}

func codexCLIParseError(operation string, err error) error {
	return &core.Error{
		Provider:  core.ProviderOpenAI,
		Flavor:    core.FlavorCLI,
		Operation: operation,
		Err:       err,
	}
}

func codexCLIResultError(operation string, code string, message string) error {
	return &core.Error{
		Provider:  core.ProviderOpenAI,
		Flavor:    core.FlavorCLI,
		Operation: operation,
		Code:      code,
		Message:   strings.TrimSpace(message),
	}
}

// codexCLIFailureError normalizes the two failure events both the batch and the
// streaming parser handle. It reports nil for any other event type.
func codexCLIFailureError(event codexCLIEvent, operation string) error {
	var code, message, fallback string
	switch event.Type {
	case "turn.failed":
		code, message, fallback = "turn_failed", event.Error.Message, "Codex CLI turn failed"
	case "error":
		code, message, fallback = "error", event.Message, "Codex CLI stream failed"
	default:
		return nil
	}
	if strings.TrimSpace(message) == "" {
		message = fallback
	}
	return codexCLIResultError(operation, code, message)
}

// codexCLIOutcomeError reports the error for output that ended without a
// completed turn or a final agent message, keeping both parsers in agreement.
func codexCLIOutcomeError(operation string, completed bool, hasFinalMessage bool) error {
	switch {
	case !completed:
		return codexCLIResultError(operation, "missing_turn_completion", "Codex CLI output did not complete a turn")
	case !hasFinalMessage:
		return codexCLIResultError(operation, "missing_final_message", "Codex CLI output did not contain a completed agent message")
	}
	return nil
}
