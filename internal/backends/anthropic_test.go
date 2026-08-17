package backends

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/lao/llm-driver/internal/core"
)

func TestNormalizeAnthropicErrorPreservesMessage(t *testing.T) {
	t.Parallel()

	var apiError anthropic.Error
	if err := json.Unmarshal([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`), &apiError); err != nil {
		t.Fatalf("json.Unmarshal(): %v", err)
	}
	apiError.StatusCode = http.StatusBadRequest

	err := normalizeAnthropicError(&apiError)
	var target *core.Error
	if !errors.As(err, &target) {
		t.Fatalf("normalizeAnthropicError() = %T %v, want *Error", err, err)
	}
	if target.Message != "bad request" {
		t.Fatalf("error message = %q, want %q", target.Message, "bad request")
	}
}

func TestNormalizeAnthropicCompletionReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		reason anthropic.StopReason
		want   core.CompletionReason
	}{
		{name: "end turn", reason: anthropic.StopReasonEndTurn, want: core.CompletionReasonStop},
		// A stop-sequence ending is a normal stop, not a distinct reason.
		{name: "stop sequence", reason: anthropic.StopReasonStopSequence, want: core.CompletionReasonStop},
		{name: "maximum tokens", reason: anthropic.StopReasonMaxTokens, want: core.CompletionReasonMaxTokens},
		{name: "refusal", reason: anthropic.StopReasonRefusal, want: core.CompletionReasonRefusal},
		// An unrecognized stop reason (a value added by a future API) must
		// degrade to the empty reason, never panic or misclassify.
		{name: "unknown reason", reason: anthropic.StopReason("future_reason"), want: ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			if got := normalizeAnthropicCompletionReason(test.reason); got != test.want {
				t.Fatalf("normalizeAnthropicCompletionReason(%q) = %q, want %q", test.reason, got, test.want)
			}
		})
	}
}

// TestAnthropicResponseSkipsUnknownContentBlocks guards the assembly of the
// final text against content blocks the adapter does not render (thinking,
// tool_use, or any block type a future API adds). They must be skipped, with
// the surrounding text blocks preserved in order — never a panic or a dropped
// answer.
func TestAnthropicResponseSkipsUnknownContentBlocks(t *testing.T) {
	t.Parallel()

	const raw = `{
  "id":"msg_123",
  "type":"message",
  "role":"assistant",
  "model":"claude-test",
  "content":[
    {"type":"text","text":"Hello, "},
    {"type":"thinking","thinking":"internal reasoning"},
    {"type":"tool_use","id":"tool_1","name":"lookup","input":{}},
    {"type":"future_block","payload":"whatever"},
    {"type":"text","text":"world"}
  ],
  "stop_reason":"end_turn",
  "stop_sequence":null,
  "usage":{"input_tokens":10,"output_tokens":5}
}`

	var message anthropic.Message
	if err := json.Unmarshal([]byte(raw), &message); err != nil {
		t.Fatalf("json.Unmarshal(): %v", err)
	}

	response := anthropicResponse(&message)
	if response.Text != "Hello, world" {
		t.Fatalf("response text = %q, want %q", response.Text, "Hello, world")
	}
	if response.CompletionReason != core.CompletionReasonStop {
		t.Fatalf("completion reason = %q, want %q", response.CompletionReason, core.CompletionReasonStop)
	}
}
