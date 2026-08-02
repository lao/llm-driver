package backends

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/lao/claudewrap/internal/core"
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
		{name: "maximum tokens", reason: anthropic.StopReasonMaxTokens, want: core.CompletionReasonMaxTokens},
		{name: "refusal", reason: anthropic.StopReasonRefusal, want: core.CompletionReasonRefusal},
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
