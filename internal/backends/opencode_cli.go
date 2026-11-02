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

const maxOpenCodeEventSize = 16 << 20

type openCodeCLIBackend struct {
	cliBackendConfig
}

// openCodeEvent is one `opencode run --format json` line. Text parts carry the
// answer; step_finish carries usage and a completion reason when opencode emits
// one; error carries a failed run. Unknown types are ignored. opencode 2.x
// omits the terminal step_finish for a plain text answer, so its absence after
// text is not an error.
type openCodeEvent struct {
	Type      string `json:"type"`
	SessionID string `json:"sessionID"`
	Part      struct {
		ID     string `json:"id"`
		Text   string `json:"text"`
		Reason string `json:"reason"`
		Tokens struct {
			Input     int `json:"input"`
			Output    int `json:"output"`
			Reasoning int `json:"reasoning"`
			Cache     struct {
				Read  int `json:"read"`
				Write int `json:"write"`
			} `json:"cache"`
		} `json:"tokens"`
	} `json:"part"`
	Error struct {
		// Type is the opencode 2.x error code (for example
		// "provider.no-route"); Name is the 1.x spelling.
		Type    string `json:"type"`
		Name    string `json:"name"`
		Message string `json:"message"`
		Data    struct {
			Message string `json:"message"`
		} `json:"data"`
	} `json:"error"`
}

func newOpenCodeCLIBackend(config cliBackendConfig) *openCodeCLIBackend {
	return &openCodeCLIBackend{withCLIDefaults(config, "opencode")}
}

// These flags target the non-interactive `opencode run --format json` contract:
// one JSON event per line, prompt on stdin, model in provider/model form. The
// opt-in `go test -tags integration` smoke test in the root package launches the
// real binary so a renamed or removed flag surfaces instead of failing only at
// runtime.
func (b *openCodeCLIBackend) Generate(ctx context.Context, request core.Request) (*core.Response, error) {
	if err := b.validateEffort(request); err != nil {
		return nil, err
	}
	stdout, stderr, err := executeCLI(
		ctx,
		core.ProviderOpenCode,
		b.path,
		b.commandArgs(request),
		[]byte(openCodePrompt(request)),
		b.runner,
	)
	if err != nil {
		// A non-zero exit whose stdout still carries an error event holds the
		// real reason (opencode prints it to stdout and leaves stderr empty).
		// Surface that diagnostic instead of the generic process error.
		var processError *core.Error
		if errors.As(err, &processError) && processError.Status != 0 {
			if diagnostic := openCodeDiagnostic(stdout); diagnostic != nil {
				diagnostic.Status = processError.Status
				return nil, diagnostic
			}
		}
		return nil, err
	}
	response, err := parseOpenCodeOutput(stdout, openCodeModel(b.model, request.Effort), request.OutputSchema)
	if err != nil {
		return nil, openCodeStderrError(err, stderr)
	}
	return response, nil
}

// validateEffort rejects an effort 2.x cannot express: the variant is part of
// the model id, so there is no default-model form and no way to override a
// variant already in the model. Rejecting matches the library's contract for
// unsupported request/target combinations; callers that want to degrade
// instead must clear Effort before the call.
func (b *openCodeCLIBackend) validateEffort(request core.Request) error {
	if request.Effort == "" {
		return nil
	}
	if b.model == "" {
		return fmt.Errorf("%w: opencode effort %q requires Config.Model (2.x encodes it as model#variant)", core.ErrUnsupportedFeature, request.Effort)
	}
	if strings.Contains(b.model, "#") {
		return fmt.Errorf("%w: opencode effort %q conflicts with the variant already in model %q; put the desired variant in the model and leave Effort empty", core.ErrUnsupportedFeature, request.Effort, b.model)
	}
	return nil
}

// openCodeModel returns the model id opencode 2.x actually runs: the variant is
// a suffix of the model id. validateEffort has already rejected the two cases
// that cannot be expressed.
func openCodeModel(model string, effort core.Effort) string {
	if effort != "" && model != "" && !strings.Contains(model, "#") {
		return model + "#" + string(effort)
	}
	return model
}

// commandArgs builds the argv shared by the batch and streaming paths.
func (b *openCodeCLIBackend) commandArgs(request core.Request) []string {
	model := openCodeModel(b.model, request.Effort)
	args := []string{"run", "--format", "json"}
	if model != "" {
		args = append(args, "--model", model)
	}
	return append(args, b.args...)
}

// GenerateStream reuses the batch invocation and parses the same JSON events as
// they arrive instead of after the process exits.
func (b *openCodeCLIBackend) GenerateStream(ctx context.Context, request core.Request) (core.ChunkStream, error) {
	if err := b.validateEffort(request); err != nil {
		return nil, err
	}
	process, err := startCLIStream(
		ctx,
		core.ProviderOpenCode,
		b.path,
		b.commandArgs(request),
		[]byte(openCodePrompt(request)),
		b.streamRunner,
	)
	if err != nil {
		return nil, err
	}
	return newCLIChunkStream(
		core.ProviderOpenCode,
		process,
		newOpenCodeParser(openCodeModel(b.model, request.Effort), request.OutputSchema, "generate_stream"),
	), nil
}

// openCodePrompt renders the prompt sent on stdin. opencode run has no system or
// schema flag, so System is prefixed to the transcript and a requested schema is
// stated up front; setStructuredOutput still validates the result and fails
// loudly when the model returns non-JSON.
func openCodePrompt(request core.Request) string {
	transcript := renderTranscript(request.Messages)
	var prompt strings.Builder
	if request.OutputSchema != nil {
		prompt.WriteString("Respond with only a single JSON value that conforms to this JSON Schema, with no markdown fences or other text.\n")
		prompt.Write(compactSchema(request.OutputSchema))
		prompt.WriteString("\n\n")
	}
	if request.System != "" {
		prompt.WriteString(request.System)
		prompt.WriteString("\n\n")
	}
	prompt.WriteString(transcript)
	return prompt.String()
}

// openCodeParser maps `opencode run --format json` events to response text and
// builds the final response. It serves both the batch and streaming paths: the
// per-part emitted map treats an event's text as cumulative for its part id, so
// a repeated part is never emitted twice and separate parts concatenate.
type openCodeParser struct {
	operation string
	schema    *core.OutputSchema
	model     string

	result   core.Response
	emitted  map[string]int
	text     strings.Builder
	finished bool
	reason   string
}

func newOpenCodeParser(model string, schema *core.OutputSchema, operation string) *openCodeParser {
	return &openCodeParser{
		operation: operation,
		schema:    schema,
		model:     model,
		emitted:   make(map[string]int),
	}
}

func (p *openCodeParser) event(line []byte) (string, error) {
	var event openCodeEvent
	if err := json.Unmarshal(line, &event); err != nil {
		return "", openCodeParseError(p.operation, fmt.Errorf("decode opencode event: %w", err))
	}
	if event.SessionID != "" {
		p.result.ID = event.SessionID
	}
	switch event.Type {
	case "text":
		if p.finished && p.emitted[event.Part.ID] < len(event.Part.Text) {
			// A new step answered after the previous step's finish. That
			// reason described the prior step (a tool call, say), not the
			// final text, so it must not become the response's reason. A
			// re-delivered cumulative part emits no delta and keeps the reason.
			p.finished = false
			p.reason = ""
		}
		return p.textDelta(event)
	case "step_finish":
		p.finished = true
		// The reason is provider-controlled and reaches core.Error.Reason on
		// error paths, so sanitize it once here.
		p.reason = clipMessage(event.Part.Reason)
		// Each step_finish reports that step's usage. Sum across steps so a
		// tool-call-then-text turn keeps the tool step's spend; a final text
		// step without a step_finish contributes nothing (2.x reports none).
		p.result.Usage = addOpenCodeUsage(p.result.Usage, openCodeUsage(event))
	case "error":
		return "", openCodeFailureError(event, p.operation)
	}
	return "", nil
}

// textDelta returns the new suffix of the event's text for its part id.
func (p *openCodeParser) textDelta(event openCodeEvent) (string, error) {
	text := event.Part.Text
	seen := p.emitted[event.Part.ID]
	if seen >= len(text) {
		return "", nil
	}
	// A misaligned offset would split a UTF-8 rune; enforce the cumulative
	// assumption rather than emit corrupted output.
	if !utf8.RuneStart(text[seen]) {
		return "", openCodeParseError(p.operation, fmt.Errorf("opencode text part %q is not byte-aligned at offset %d", event.Part.ID, seen))
	}
	p.emitted[event.Part.ID] = len(text)
	delta := text[seen:]
	p.text.WriteString(delta)
	return delta, nil
}

func (p *openCodeParser) response() (*core.Response, error) {
	if p.text.Len() == 0 {
		if !p.finished {
			return nil, openCodeResultError(p.operation, "missing_step_finish", "opencode did not finish a step")
		}
		// A terminal step_finish whose reason is tool-calls means the agent
		// spent the turn on a tool instead of answering — commonly because a
		// headless run auto-rejected the permission. Report that distinctly so
		// it is not confused with an empty completion.
		switch p.reason {
		case openCodeReasonToolCalls:
			err := openCodeResultError(p.operation, core.CodeToolCalls, "opencode ended the turn on a tool call without returning text")
			err.Reason = p.reason
			return nil, err
		case openCodeReasonLength:
			err := openCodeResultError(p.operation, "max_tokens", "opencode stopped at the output cap before returning text")
			err.Reason = p.reason
			return nil, err
		case openCodeReasonContentFilter:
			err := openCodeResultError(p.operation, "content_filter", "opencode was filtered before returning text")
			err.Reason = p.reason
			return nil, err
		}
		err := openCodeResultError(p.operation, "missing_text", "opencode returned no text")
		err.Reason = p.reason
		return nil, err
	}
	// opencode 2.x ends a plain text turn without a terminal step_finish. The
	// completed process plus text is the finish signal in that case; usage and
	// the completion reason are only known when a step_finish was emitted (for
	// example between tool calls).
	response := p.result
	response.Model = p.model
	response.Text = p.text.String()
	response.CompletionReason = openCodeCompletionReason(p.reason)
	if response.CompletionReason == "" && !p.finished {
		// No terminal step_finish: the answer may be complete, but the library
		// cannot distinguish it from a truncated one.
		response.CompletionReason = core.CompletionReasonUnverified
	}
	if err := setStructuredOutput(&response, p.schema, core.ProviderOpenCode, core.FlavorCLI, p.operation); err != nil {
		return nil, err
	}
	return &response, nil
}

// openCodeReasonToolCalls is the step_finish reason opencode reports when the
// agent ended a step to call a tool rather than answer in text.
const openCodeReasonToolCalls = "tool-calls"

// openCodeReasonContentFilter is the AI SDK finish reason opencode reports when
// the provider filtered the response.
const openCodeReasonContentFilter = "content-filter"

// openCodeReasonLength is the step_finish reason opencode reports when the
// provider stopped on its output-token cap. A turn that hits it mid-answer can
// deliver truncated text or, when the cap is reached before the first text
// part, no text at all — the reason is then the only evidence of truncation,
// which is why it is surfaced on the error as well as the response.
const openCodeReasonLength = "length"

// openCodeCompletionReason maps a step_finish reason to the neutral enum,
// following the API adapters: an unrecognized reason degrades to empty rather
// than misclassifying.
func openCodeCompletionReason(reason string) core.CompletionReason {
	switch reason {
	case "stop":
		return core.CompletionReasonStop
	case openCodeReasonToolCalls:
		return core.CompletionReasonToolUse
	case openCodeReasonLength:
		return core.CompletionReasonMaxTokens
	case openCodeReasonContentFilter:
		return core.CompletionReasonContentFilter
	default:
		return ""
	}
}

// openCodeStderrError attaches a stderr diagnostic to a result error when one is
// available. opencode exits 0 after a terminal step_finish even when it
// auto-rejected a tool permission, leaving the explanation on stderr; without
// this the error would carry no reason for the missing text.
func openCodeStderrError(err error, stderr []byte) error {
	return attachCLIStderr(err, stderr)
}

func openCodeUsage(event openCodeEvent) core.Usage {
	return core.Usage{
		InputTokens:              event.Part.Tokens.Input,
		OutputTokens:             event.Part.Tokens.Output,
		CachedInputTokens:        event.Part.Tokens.Cache.Read,
		CacheCreationInputTokens: event.Part.Tokens.Cache.Write,
		ReasoningTokens:          event.Part.Tokens.Reasoning,
	}
}

func addOpenCodeUsage(a, b core.Usage) core.Usage {
	return core.Usage{
		InputTokens:              a.InputTokens + b.InputTokens,
		OutputTokens:             a.OutputTokens + b.OutputTokens,
		CachedInputTokens:        a.CachedInputTokens + b.CachedInputTokens,
		CacheCreationInputTokens: a.CacheCreationInputTokens + b.CacheCreationInputTokens,
		ReasoningTokens:          a.ReasoningTokens + b.ReasoningTokens,
	}
}

func parseOpenCodeOutput(output []byte, model string, schema *core.OutputSchema) (*core.Response, error) {
	parser := newOpenCodeParser(model, schema, "generate")
	scanner := bufio.NewScanner(bytes.NewReader(output))
	scanner.Buffer(make([]byte, 64*1024), maxOpenCodeEventSize)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		if _, err := parser.event(line); err != nil {
			return nil, err
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, openCodeParseError("generate", fmt.Errorf("scan opencode output: %w", err))
	}
	return parser.response()
}

// openCodeDiagnostic returns the error event a failed run printed on stdout, or
// a parse error when stdout contains no recoverable event.
func openCodeDiagnostic(output []byte) *core.Error {
	scanner := bufio.NewScanner(bytes.NewReader(output))
	scanner.Buffer(make([]byte, 64*1024), maxOpenCodeEventSize)
	var decodeErr error
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var event openCodeEvent
		if err := json.Unmarshal(line, &event); err != nil {
			if decodeErr == nil {
				decodeErr = fmt.Errorf("decode opencode diagnostic %q: %w", string(line), err)
			}
			continue
		}
		if event.Type == "error" {
			return openCodeFailureError(event, "generate")
		}
	}
	if err := scanner.Err(); err != nil {
		return openCodeParseError("generate", fmt.Errorf("scan opencode diagnostic: %w", err))
	}
	if decodeErr != nil {
		diagnostic := openCodeParseError("generate", decodeErr)
		diagnostic.Message = fmt.Sprintf("opencode diagnostic output: %q", string(output))
		return diagnostic
	}
	return nil
}

func openCodeParseError(operation string, err error) *core.Error {
	return &core.Error{
		Provider:  core.ProviderOpenCode,
		Flavor:    core.FlavorCLI,
		Operation: operation,
		Err:       err,
	}
}

func openCodeResultError(operation string, code string, message string) *core.Error {
	return &core.Error{
		Provider:  core.ProviderOpenCode,
		Flavor:    core.FlavorCLI,
		Operation: operation,
		Code:      code,
		Message:   message,
	}
}

// openCodeFailureError normalizes an error event. opencode nests the message
// under error.data.message, with error.message used by some failures.
func openCodeFailureError(event openCodeEvent, operation string) *core.Error {
	message := event.Error.Data.Message
	if strings.TrimSpace(message) == "" {
		message = event.Error.Message
	}
	if strings.TrimSpace(message) == "" {
		message = "opencode run failed"
	}
	code := event.Error.Type
	if code == "" {
		code = event.Error.Name
	}
	if code == "" {
		code = "error"
	}
	// The message and code are provider/CLI-controlled and reach caller logs.
	return openCodeResultError(operation, clipMessage(code), clipMessage(message))
}
