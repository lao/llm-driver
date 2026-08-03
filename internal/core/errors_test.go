package core

import (
	"errors"
	"strings"
	"testing"
)

func TestErrorIncludesContextAndPreservesCause(t *testing.T) {
	t.Parallel()

	cause := errors.New("network unavailable")
	err := &Error{
		Provider:  ProviderOpenAI,
		Flavor:    FlavorAPI,
		Operation: "generate",
		Status:    429,
		Code:      "rate_limit",
		Message:   "slow down",
		Err:       cause,
	}

	for _, want := range []string{"llmdriver", "openai/api", "generate", "429", "rate_limit", "slow down"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("Error() = %q, missing %q", err.Error(), want)
		}
	}
	if !errors.Is(err, cause) {
		t.Fatal("errors.Is() did not preserve the cause")
	}
}
