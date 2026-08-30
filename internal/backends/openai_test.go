package backends

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3/responses"

	"github.com/lao/llm-driver/internal/core"
)

// unmarshalOpenAIResponse decodes a Responses API payload into the SDK type the
// adapter reads, failing the test on malformed fixtures.
func unmarshalOpenAIResponse(t *testing.T, raw string) *responses.Response {
	t.Helper()
	var response responses.Response
	if err := json.Unmarshal([]byte(raw), &response); err != nil {
		t.Fatalf("json.Unmarshal(): %v", err)
	}
	return &response
}

// TestOpenAIResponseTextSkipsUnknownParts guards final-text assembly against
// output items and content parts the adapter does not render (reasoning items,
// or any part type a future API adds). Unknown parts must be skipped while the
// recognized output_text/refusal parts are preserved in order — never a panic
// or a dropped answer.
func TestOpenAIResponseTextSkipsUnknownParts(t *testing.T) {
	t.Parallel()

	const raw = `{
  "id":"resp_123",
  "model":"gpt-test",
  "status":"completed",
  "output":[
    {"type":"reasoning","id":"rs_1","summary":[]},
    {"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[
      {"type":"output_text","text":"Hello, ","annotations":[]},
      {"type":"future_part","text":"IGNORED"},
      {"type":"output_text","text":"world","annotations":[]}
    ]}
  ],
  "usage":{"input_tokens":10,"output_tokens":5}
}`

	response := unmarshalOpenAIResponse(t, raw)
	if got := openAIResponseText(response); got != "Hello, world" {
		t.Fatalf("openAIResponseText() = %q, want %q", got, "Hello, world")
	}
	if got := openAICompletionReason(response); got != core.CompletionReasonStop {
		t.Fatalf("openAICompletionReason() = %q, want %q", got, core.CompletionReasonStop)
	}
}

// TestOpenAIResponseRefusalPart confirms a refusal part is surfaced as text and
// flips the completion reason, even when interleaved with an unknown part.
func TestOpenAIResponseRefusalPart(t *testing.T) {
	t.Parallel()

	const raw = `{
  "id":"resp_456",
  "model":"gpt-test",
  "status":"completed",
  "output":[
    {"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[
      {"type":"future_part","text":"IGNORED"},
      {"type":"refusal","refusal":"I can't help with that."}
    ]}
  ],
  "usage":{"input_tokens":3,"output_tokens":7}
}`

	response := unmarshalOpenAIResponse(t, raw)
	if got := openAIResponseText(response); got != "I can't help with that." {
		t.Fatalf("openAIResponseText() = %q, want refusal text", got)
	}
	if got := openAICompletionReason(response); got != core.CompletionReasonRefusal {
		t.Fatalf("openAICompletionReason() = %q, want %q", got, core.CompletionReasonRefusal)
	}
}
