package llmwrapper

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestContractGenerateAcrossFourTargets(t *testing.T) {
	t.Parallel()

	targets := []struct {
		provider Provider
		flavor   Flavor
	}{
		{provider: ProviderClaude, flavor: FlavorAPI},
		{provider: ProviderClaude, flavor: FlavorCLI},
		{provider: ProviderOpenAI, flavor: FlavorAPI},
		{provider: ProviderOpenAI, flavor: FlavorCLI},
	}

	for _, target := range targets {
		target := target
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			config, calls := newContractTarget(t, target.provider, target.flavor)
			client, err := New(config)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			response, err := client.Generate(context.Background(), Request{
				System:    "Be concise.",
				MaxTokens: 64,
				Messages:  []Message{User("Hello"), Assistant("Hi"), User("Continue")},
			})
			if err != nil {
				t.Fatalf("Generate(): %v", err)
			}

			// These are the response fields exposed by every target. Provider-specific
			// cache and reasoning counters are covered by each adapter's focused tests.
			if response.ID == "" || response.Model != "contract-model" || response.Text != "contract answer" {
				t.Fatalf("response = %+v", response)
			}
			if response.Usage != (Usage{InputTokens: 12, OutputTokens: 4}) {
				t.Fatalf("usage = %+v", response.Usage)
			}
			if response.Provider != target.provider || response.Flavor != target.flavor {
				t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
			}
			if got := calls(); got != 1 {
				t.Fatalf("transport calls = %d, want 1", got)
			}

			// CLI targets accept MaxTokens to keep the request portable, but their
			// agent CLIs have no reliable equivalent flag. Validation is still shared.
			_, err = client.Generate(context.Background(), Request{})
			if !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("invalid request error = %v, want ErrInvalidRequest", err)
			}
			var contextual *Error
			if !errors.As(err, &contextual) {
				t.Fatalf("invalid request error = %T %v, want *Error", err, err)
			}
			if contextual.Provider != target.provider || contextual.Flavor != target.flavor || contextual.Operation != "generate" {
				t.Fatalf("error context = %+v", contextual)
			}
			if got := calls(); got != 1 {
				t.Fatalf("transport called for invalid request: calls = %d", got)
			}
		})
	}
}

func TestContractTargetMatrix(t *testing.T) {
	t.Parallel()

	providers := []Provider{ProviderClaude, ProviderOpenAI, "other"}
	flavors := []Flavor{FlavorAPI, FlavorCLI, "other"}
	for _, provider := range providers {
		for _, flavor := range flavors {
			provider := provider
			flavor := flavor
			t.Run(string(provider)+"/"+string(flavor), func(t *testing.T) {
				t.Parallel()

				_, err := New(Config{Provider: provider, Flavor: flavor, Model: "contract-model"})
				wantValid := (provider == ProviderClaude || provider == ProviderOpenAI) &&
					(flavor == FlavorAPI || flavor == FlavorCLI)
				if wantValid && err != nil {
					t.Fatalf("New() error = %v", err)
				}
				if !wantValid && !errors.Is(err, ErrInvalidConfig) {
					t.Fatalf("New() error = %v, want ErrInvalidConfig", err)
				}
			})
		}
	}
}

func newContractTarget(t *testing.T, provider Provider, flavor Flavor) (Config, func() int) {
	t.Helper()

	config := Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	var calls atomic.Int32
	callCount := func() int { return int(calls.Load()) }

	if flavor == FlavorCLI {
		runner := &fakeCommandRunner{}
		if provider == ProviderClaude {
			runner.stdout = []byte(`{
  "type":"result",
  "subtype":"success",
  "result":"contract answer",
  "session_id":"contract-session",
  "usage":{"input_tokens":12,"output_tokens":4}
}`)
		} else {
			runner.stdout = []byte(`{"type":"thread.started","thread_id":"contract-thread"}
{"type":"item.completed","item":{"id":"message-1","type":"agent_message","text":"contract answer"}}
{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":4}}
`)
		}
		config.runner = func(ctx context.Context, name string, args []string, stdin []byte) ([]byte, []byte, int, error) {
			calls.Add(1)
			return runner.run(ctx, name, args, stdin)
		}
		return config, callCount
	}

	response := `{
  "id":"contract-message",
  "type":"message",
  "role":"assistant",
  "model":"contract-model",
  "content":[{"type":"text","text":"contract answer"}],
  "stop_reason":"end_turn",
  "stop_sequence":null,
  "usage":{"input_tokens":12,"output_tokens":4}
}`
	if provider == ProviderOpenAI {
		response = `{
  "id":"contract-response",
  "object":"response",
  "status":"completed",
  "model":"contract-model",
  "output":[{
    "id":"message-1",
    "type":"message",
    "status":"completed",
    "role":"assistant",
    "content":[{"type":"output_text","text":"contract answer","annotations":[],"logprobs":[]}]
  }],
  "usage":{"input_tokens":12,"output_tokens":4,"total_tokens":16}
}`
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, response)
	}))
	t.Cleanup(server.Close)
	config.APIKey = "contract-key"
	config.BaseURL = server.URL
	return config, callCount
}
