package llmdriver

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/lao/llm-driver/internal/core"
)

// contractTarget is one of the four provider/flavor combinations every contract
// test runs against.
type contractTarget struct {
	provider Provider
	flavor   Flavor
}

func contractTargets() []contractTarget {
	return []contractTarget{
		{provider: ProviderClaude, flavor: FlavorAPI},
		{provider: ProviderClaude, flavor: FlavorCLI},
		{provider: ProviderOpenAI, flavor: FlavorAPI},
		{provider: ProviderOpenAI, flavor: FlavorCLI},
	}
}

func TestContractGenerateAcrossFourTargets(t *testing.T) {
	t.Parallel()

	for _, target := range contractTargets() {
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			config, runner, calls := newContractTarget(t, target.provider, target.flavor)
			client, err := newClient(config, runner)
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

// TestContractStructuredOutputAcrossFourTargets sends the same OutputSchema
// request to all four targets and asserts an identical normalized shape: Text is
// the raw JSON and StructuredOutput is the same validated bytes, regardless of
// how each target maps the schema.
func TestContractStructuredOutputAcrossFourTargets(t *testing.T) {
	t.Parallel()

	for _, target := range contractTargets() {
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			config, runner := newContractStructuredTarget(t, target.provider, target.flavor)
			client, err := newClient(config, runner)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			response, err := client.Generate(context.Background(), Request{
				MaxTokens:    64,
				Messages:     []Message{User("q")},
				OutputSchema: &OutputSchema{Name: "answer", Schema: structuredSchema},
			})
			if err != nil {
				t.Fatalf("Generate(): %v", err)
			}
			if response.Text != structuredJSON {
				t.Fatalf("Text = %q, want %q", response.Text, structuredJSON)
			}
			if string(response.StructuredOutput) != structuredJSON {
				t.Fatalf("StructuredOutput = %q, want %q", response.StructuredOutput, structuredJSON)
			}
			if !client.Supports(FeatureStructuredOutput) {
				t.Fatal("Supports(FeatureStructuredOutput) = false")
			}
		})
	}
}

// TestContractEffortAcrossFourTargets sends a supported effort level to every
// target and asserts each maps it (the transport is reached and the call
// succeeds), then asserts openai/api rejects an out-of-ceiling value with
// ErrUnsupportedFeature before any transport call — values are never remapped.
func TestContractEffortAcrossFourTargets(t *testing.T) {
	t.Parallel()

	for _, target := range contractTargets() {
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			config, runner, calls := newContractTarget(t, target.provider, target.flavor)
			client, err := newClient(config, runner)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}
			if !client.Supports(FeatureReasoningEffort) {
				t.Fatal("Supports(FeatureReasoningEffort) = false")
			}

			// EffortMedium is inside every target's allowed set, including openai/api.
			if _, err := client.Generate(context.Background(), Request{
				MaxTokens: 64,
				Messages:  []Message{User("q")},
				Effort:    EffortMedium,
			}); err != nil {
				t.Fatalf("Generate() with effort: %v", err)
			}
			if got := calls(); got != 1 {
				t.Fatalf("transport calls = %d, want 1", got)
			}

			// openai/api tops out at high: xhigh and max are rejected loudly and
			// never reach the transport.
			if target.provider == ProviderOpenAI && target.flavor == FlavorAPI {
				for _, value := range []Effort{EffortXHigh, EffortMax} {
					_, err := client.Generate(context.Background(), Request{
						MaxTokens: 64,
						Messages:  []Message{User("q")},
						Effort:    value,
					})
					if !errors.Is(err, ErrUnsupportedFeature) {
						t.Fatalf("effort %q error = %v, want ErrUnsupportedFeature", value, err)
					}
					if got := calls(); got != 1 {
						t.Fatalf("transport called for rejected effort %q: calls = %d", value, got)
					}
				}
			}
		})
	}
}

func newContractStructuredTarget(t *testing.T, provider Provider, flavor Flavor) (Config, core.CommandRunner) {
	t.Helper()

	config := Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	if flavor == FlavorCLI {
		runner := &fakeCommandRunner{}
		if provider == ProviderClaude {
			runner.stdout = []byte(`{"type":"result","subtype":"success","result":` + strconv.Quote(structuredJSON) + `,"session_id":"s"}`)
		} else {
			runner.stdout = []byte(`{"type":"thread.started","thread_id":"t"}` + "\n" +
				`{"type":"item.completed","item":{"id":"m","type":"agent_message","text":` + strconv.Quote(structuredJSON) + `}}` + "\n" +
				`{"type":"turn.completed","usage":{}}` + "\n")
		}
		return config, runner.run
	}

	var response string
	if provider == ProviderClaude {
		response = `{"id":"c","type":"message","role":"assistant","model":"contract-model",` +
			`"content":[{"type":"text","text":` + strconv.Quote(structuredJSON) + `}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	} else {
		response = `{"id":"c","object":"response","status":"completed","model":"contract-model",` +
			`"output":[{"id":"m","type":"message","status":"completed","role":"assistant",` +
			`"content":[{"type":"output_text","text":` + strconv.Quote(structuredJSON) + `,"annotations":[],"logprobs":[]}]}],` +
			`"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, response)
	}))
	t.Cleanup(server.Close)
	config.APIKey = "contract-key"
	config.BaseURL = server.URL
	return config, nil
}

func TestContractTargetMatrix(t *testing.T) {
	t.Parallel()

	providers := []Provider{ProviderClaude, ProviderOpenAI, "other"}
	flavors := []Flavor{FlavorAPI, FlavorCLI, "other"}
	for _, provider := range providers {
		for _, flavor := range flavors {
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

func newContractTarget(t *testing.T, provider Provider, flavor Flavor) (Config, core.CommandRunner, func() int) {
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
		run := func(ctx context.Context, name string, args []string, stdin []byte) ([]byte, []byte, int, error) {
			calls.Add(1)
			return runner.run(ctx, name, args, stdin)
		}
		return config, run, callCount
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
	return config, nil, callCount
}
