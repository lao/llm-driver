package backends

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/lao/llm-driver/internal/core"
)

// setStructuredOutput fills response.StructuredOutput from the model's response
// text when the request asked for a schema. Every target returns the JSON
// document as its response text, so the library validates that text is JSON and
// re-exposes the validated bytes; Text keeps the raw output. Non-JSON output is
// a normalized error with the operation preserved. It is a no-op when no schema
// was requested, so callers can invoke it unconditionally.
func setStructuredOutput(response *core.Response, schema *core.OutputSchema, provider core.Provider, flavor core.Flavor, operation string) error {
	if schema == nil {
		return nil
	}
	trimmed := strings.TrimSpace(response.Text)
	if !json.Valid([]byte(trimmed)) {
		return &core.Error{
			Provider:  provider,
			Flavor:    flavor,
			Operation: operation,
			Code:      "invalid_structured_output",
			Message:   "model output is not valid JSON",
		}
	}
	response.StructuredOutput = json.RawMessage(trimmed)
	return nil
}

// compactSchema returns the schema as compact JSON for CLI flags and files. The
// request is already validated, so json.Compact cannot fail here.
func compactSchema(schema *core.OutputSchema) []byte {
	var buffer bytes.Buffer
	_ = json.Compact(&buffer, schema.Schema)
	return buffer.Bytes()
}
