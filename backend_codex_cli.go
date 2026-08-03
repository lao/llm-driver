package llmwrapper

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const maxCodexEventSize = 16 << 20

type codexCLIBackend struct {
	model        string
	path         string
	args         []string
	runner       commandRunner
	streamRunner streamingCommandRunner
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
	path := config.path
	if path == "" {
		path = "codex"
	}
	runner := config.runner
	if runner == nil {
		runner = defaultCommandRunner
	}
	streamRunner := config.streamRunner
	if streamRunner == nil {
		streamRunner = defaultStreamingCommandRunner
	}
	return &codexCLIBackend{
		model:        config.model,
		path:         path,
		args:         append([]string(nil), config.args...),
		runner:       runner,
		streamRunner: streamRunner,
	}
}

// These flags target the non-interactive `codex exec` contract described in
// SPEC.md (JSONL output, read-only sandbox, no git-repo check). Default tests
// inject a fake runner and cannot catch a renamed or removed flag, so the
// opt-in `go test -tags integration` smoke test in
// backend_cli_integration_test.go launches the real binary to surface flag
// drift against the installed `codex` version.
func (b *codexCLIBackend) generate(ctx context.Context, request Request) (*Response, error) {
	args := b.commandArgs(request)

	stdout, err := executeCLI(
		ctx,
		ProviderOpenAI,
		b.path,
		args,
		[]byte(renderTranscript(request.Messages)),
		b.runner,
	)
	if err != nil {
		var processError *Error
		if errors.As(err, &processError) && processError.Status != 0 {
			_, diagnosticErr := parseCodexCLIOutput(stdout, b.model)
			var diagnostic *Error
			if errors.As(diagnosticErr, &diagnostic) && (diagnostic.Code == "turn_failed" || diagnostic.Code == "error") {
				diagnostic.Status = processError.Status
				return nil, diagnostic
			}
		}
		return nil, err
	}
	return parseCodexCLIOutput(stdout, b.model)
}

func (b *codexCLIBackend) commandArgs(request Request) []string {
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

// generateStream reuses the batch invocation and parses the same JSONL events
// as they arrive instead of after the process exits.
func (b *codexCLIBackend) generateStream(ctx context.Context, request Request) (chunkStream, error) {
	process, err := startCLIStream(
		ctx,
		ProviderOpenAI,
		b.path,
		b.commandArgs(request),
		[]byte(renderTranscript(request.Messages)),
		b.streamRunner,
	)
	if err != nil {
		return nil, err
	}
	return newCLIChunkStream(ProviderOpenAI, process, &codexCLIStreamParser{model: b.model}), nil
}

// codexCLIStreamParser tolerates both Codex CLI shapes: versions that publish
// an agent message incrementally and versions that publish it only on
// completion. Item text is treated as cumulative and an explicit delta field as
// an increment, so text is never emitted twice.
type codexCLIStreamParser struct {
	model           string
	result          Response
	emitted         map[string]int
	completed       bool
	hasFinalMessage bool
}

func (p *codexCLIStreamParser) event(line []byte) (string, error) {
	var event codexCLIEvent
	if err := json.Unmarshal(line, &event); err != nil {
		return "", codexCLIStreamParseError(fmt.Errorf("decode Codex CLI event: %w", err))
	}

	switch event.Type {
	case "thread.started":
		p.result.ID = event.ThreadID
	case "turn.completed":
		p.completed = true
		p.result.Usage = codexCLIUsage(event)
	case "turn.failed":
		message := strings.TrimSpace(event.Error.Message)
		if message == "" {
			message = "Codex CLI turn failed"
		}
		return "", codexCLIStreamResultError("turn_failed", message)
	case "error":
		message := strings.TrimSpace(event.Message)
		if message == "" {
			message = "Codex CLI stream failed"
		}
		return "", codexCLIStreamResultError("error", message)
	default:
		// Unknown event types are ignored, as in the batch parser.
		if !strings.HasPrefix(event.Type, "item.") || event.Item.Type != "agent_message" {
			return "", nil
		}
		if event.Type == "item.completed" {
			p.hasFinalMessage = true
		}
		if event.Delta != "" {
			return p.advance(event.Item.ID, p.seen(event.Item.ID)+len(event.Delta), event.Delta), nil
		}
		seen := p.seen(event.Item.ID)
		if seen >= len(event.Item.Text) {
			return "", nil
		}
		return p.advance(event.Item.ID, len(event.Item.Text), event.Item.Text[seen:]), nil
	}
	return "", nil
}

func (p *codexCLIStreamParser) seen(id string) int {
	return p.emitted[id]
}

func (p *codexCLIStreamParser) advance(id string, total int, text string) string {
	if p.emitted == nil {
		p.emitted = make(map[string]int)
	}
	p.emitted[id] = total
	return text
}

func (p *codexCLIStreamParser) response() (*Response, error) {
	if !p.completed {
		return nil, codexCLIStreamResultError("missing_turn_completion", "Codex CLI output did not complete a turn")
	}
	if !p.hasFinalMessage {
		return nil, codexCLIStreamResultError("missing_final_message", "Codex CLI output did not contain a completed agent message")
	}
	response := p.result
	response.Model = p.model
	return &response, nil
}

func codexCLIUsage(event codexCLIEvent) Usage {
	return Usage{
		InputTokens:              event.Usage.InputTokens,
		OutputTokens:             event.Usage.OutputTokens,
		CachedInputTokens:        event.Usage.CachedInputTokens,
		CacheCreationInputTokens: event.Usage.CacheWriteInputTokens,
		ReasoningTokens:          event.Usage.ReasoningOutputTokens,
	}
}

func codexCLIStreamParseError(err error) error {
	return &Error{
		Provider:  ProviderOpenAI,
		Flavor:    FlavorCLI,
		Operation: "generate_stream",
		Err:       err,
	}
}

func codexCLIStreamResultError(code string, message string) error {
	return &Error{
		Provider:  ProviderOpenAI,
		Flavor:    FlavorCLI,
		Operation: "generate_stream",
		Code:      code,
		Message:   strings.TrimSpace(message),
	}
}

func codexDeveloperInstructions(system string) string {
	encoded, _ := json.Marshal(system)
	return "developer_instructions=" + string(encoded)
}

func parseCodexCLIOutput(output []byte, model string) (*Response, error) {
	var response Response
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
			response.Usage = codexCLIUsage(event)
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
	return &Error{
		Provider:  ProviderOpenAI,
		Flavor:    FlavorCLI,
		Operation: "generate",
		Err:       err,
	}
}

func codexCLIResultError(code string, message string) error {
	return &Error{
		Provider:  ProviderOpenAI,
		Flavor:    FlavorCLI,
		Operation: "generate",
		Code:      code,
		Message:   strings.TrimSpace(message),
	}
}
