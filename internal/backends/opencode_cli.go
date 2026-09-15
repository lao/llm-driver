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
// answer; step_finish carries usage; error carries a failed run. Unknown types
// are ignored.
type openCodeEvent struct {
	Type      string `json:"type"`
	SessionID string `json:"sessionID"`
	Part      struct {
		ID     string `json:"id"`
		Text   string `json:"text"`
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
	stdout, err := executeCLI(
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
	return parseOpenCodeOutput(stdout, b.model, request.OutputSchema)
}

// commandArgs builds the argv shared by the batch and streaming paths.
func (b *openCodeCLIBackend) commandArgs(request core.Request) []string {
	args := []string{"run", "--format", "json", "--model", b.model}
	if request.Effort != "" {
		// Core validated the value; opencode forwards --variant to the provider.
		args = append(args, "--variant", string(request.Effort))
	}
	return append(args, b.args...)
}

// GenerateStream reuses the batch invocation and parses the same JSON events as
// they arrive instead of after the process exits.
func (b *openCodeCLIBackend) GenerateStream(ctx context.Context, request core.Request) (core.ChunkStream, error) {
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
		newOpenCodeParser(b.model, request.OutputSchema, "generate_stream"),
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
		return p.textDelta(event)
	case "step_finish":
		p.finished = true
		p.result.Usage = openCodeUsage(event)
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
	if !p.finished {
		return nil, openCodeResultError(p.operation, "missing_step_finish", "opencode did not finish a step")
	}
	if p.text.Len() == 0 {
		return nil, openCodeResultError(p.operation, "missing_text", "opencode returned no text")
	}
	response := p.result
	response.Model = p.model
	response.Text = p.text.String()
	if err := setStructuredOutput(&response, p.schema, core.ProviderOpenCode, core.FlavorCLI, p.operation); err != nil {
		return nil, err
	}
	return &response, nil
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
// nil when the output carries no error event.
func openCodeDiagnostic(output []byte) *core.Error {
	scanner := bufio.NewScanner(bytes.NewReader(output))
	scanner.Buffer(make([]byte, 64*1024), maxOpenCodeEventSize)
	for scanner.Scan() {
		var event openCodeEvent
		if err := json.Unmarshal(bytes.TrimSpace(scanner.Bytes()), &event); err != nil {
			continue
		}
		if event.Type == "error" {
			return openCodeFailureError(event, "generate")
		}
	}
	return nil
}

func openCodeParseError(operation string, err error) error {
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
	code := event.Error.Name
	if code == "" {
		code = "error"
	}
	return openCodeResultError(operation, code, message)
}
