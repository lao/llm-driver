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

// contractTarget is one provider/flavor combination every contract test runs
// against.
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
		{provider: ProviderOpenCode, flavor: FlavorCLI},
	}
}

func TestContractGenerateAcrossTargets(t *testing.T) {
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

// TestContractStructuredOutputAcrossTargets sends the same OutputSchema
// request to every target and asserts an identical normalized shape: Text is
// the raw JSON and StructuredOutput is the same validated bytes, regardless of
// how each target maps the schema.
func TestContractStructuredOutputAcrossTargets(t *testing.T) {
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

// TestContractEffortAcrossTargets sends a supported effort level to every
// target and asserts each maps it (the transport is reached and the call
// succeeds), then asserts openai/api rejects an out-of-ceiling value with
// ErrUnsupportedFeature before any transport call — values are never remapped.
func TestContractEffortAcrossTargets(t *testing.T) {
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

// TestContractSamplingParamsAcrossTargets asserts the Phase 4 support map:
// Temperature/TopP/UserID map on both API flavors and are rejected with
// ErrUnsupportedFeature (before any transport call) on both CLI flavors; TopK
// and StopSequences are supported only on claude/api and rejected everywhere
// else, including openai/api (the Responses API lacks both).
func TestContractSamplingParamsAcrossTargets(t *testing.T) {
	t.Parallel()

	temp, topP, topK := 0.5, 0.9, 40
	for _, target := range contractTargets() {
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			config, runner, calls := newContractTarget(t, target.provider, target.flavor)
			client, err := newClient(config, runner)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}
			apiFlavor := target.flavor == FlavorAPI
			claudeAPI := target.provider == ProviderClaude && apiFlavor

			// Temperature/TopP/UserID: mapped on API flavors, rejected on CLI.
			_, err = client.Generate(context.Background(), Request{
				MaxTokens:   64,
				Messages:    []Message{User("q")},
				Temperature: &temp,
				TopP:        &topP,
				UserID:      "u1",
			})
			if apiFlavor {
				if err != nil {
					t.Fatalf("Generate() sampling: %v", err)
				}
				if calls() != 1 {
					t.Fatalf("transport calls = %d, want 1", calls())
				}
				if !client.Supports(FeatureSamplingParams) || !client.Supports(FeatureUserID) {
					t.Fatal("Supports(sampling/user id) = false on API flavor")
				}
			} else {
				if !errors.Is(err, ErrUnsupportedFeature) {
					t.Fatalf("Generate() sampling error = %v, want ErrUnsupportedFeature", err)
				}
				if calls() != 0 {
					t.Fatalf("transport called for rejected sampling: calls = %d", calls())
				}
				if client.Supports(FeatureSamplingParams) || client.Supports(FeatureUserID) {
					t.Fatal("Supports(sampling/user id) = true on CLI flavor")
				}
			}

			// TopK and StopSequences: claude/api only.
			for _, request := range []Request{
				{MaxTokens: 64, Messages: []Message{User("q")}, TopK: &topK},
				{MaxTokens: 64, Messages: []Message{User("q")}, StopSequences: []string{"STOP"}},
			} {
				before := calls()
				_, err := client.Generate(context.Background(), request)
				if claudeAPI {
					if err != nil {
						t.Fatalf("Generate() topk/stop: %v", err)
					}
					if calls() != before+1 {
						t.Fatalf("transport calls = %d, want %d", calls(), before+1)
					}
				} else {
					if !errors.Is(err, ErrUnsupportedFeature) {
						t.Fatalf("Generate() topk/stop error = %v, want ErrUnsupportedFeature", err)
					}
					if calls() != before {
						t.Fatalf("transport called for rejected topk/stop: calls = %d", calls())
					}
				}
			}
			if got := claudeAPI; client.Supports(FeatureTopK) != got || client.Supports(FeatureStopSequences) != got {
				t.Fatalf("Supports(topk/stop) mismatch on %s/%s", target.provider, target.flavor)
			}
		})
	}
}

// TestContractToolCallingAcrossTargets asserts the Phase 5 support map: both
// API flavors map a tool request and return an identical normalized tool call
// with CompletionReasonToolUse, while both CLI flavors reject tools — and tool
// content parts — with ErrUnsupportedFeature before any transport call.
func TestContractToolCallingAcrossTargets(t *testing.T) {
	t.Parallel()

	weatherTool := Tool{Name: "get_weather", InputSchema: structuredSchema}
	for _, target := range contractTargets() {
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			config, runner, calls := newContractToolTarget(t, target.provider, target.flavor)
			client, err := newClient(config, runner)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			if target.flavor == FlavorCLI {
				if client.Supports(FeatureTools) {
					t.Fatal("Supports(FeatureTools) = true on CLI flavor")
				}
				// Tools, tool-result parts, and any Parts usage all trip the same
				// rejection before the transport is ever reached — a CLI flavor
				// cannot render Parts, so it must fail loudly rather than drop the
				// content that only lives there (even a plain TextPart).
				for _, request := range []Request{
					{MaxTokens: 64, Messages: []Message{User("q")}, Tools: []Tool{weatherTool}},
					{MaxTokens: 64, Messages: []Message{ToolResult("call_a", "72F", false)}},
					{MaxTokens: 64, Messages: []Message{{Role: RoleUser, Parts: []Part{TextPart{Text: "hi"}}}}},
				} {
					_, err := client.Generate(context.Background(), request)
					if !errors.Is(err, ErrUnsupportedFeature) {
						t.Fatalf("Generate() tools error = %v, want ErrUnsupportedFeature", err)
					}
				}
				if calls() != 0 {
					t.Fatalf("transport called for rejected tools: calls = %d", calls())
				}
				return
			}

			if !client.Supports(FeatureTools) {
				t.Fatal("Supports(FeatureTools) = false on API flavor")
			}
			response, err := client.Generate(context.Background(), Request{
				MaxTokens:  64,
				Messages:   []Message{User("weather?")},
				Tools:      []Tool{weatherTool},
				ToolChoice: ToolChoice{Mode: ToolChoiceRequired},
			})
			if err != nil {
				t.Fatalf("Generate(): %v", err)
			}
			if response.CompletionReason != CompletionReasonToolUse {
				t.Fatalf("completion reason = %q, want %q", response.CompletionReason, CompletionReasonToolUse)
			}
			if len(response.ToolCalls) != 1 {
				t.Fatalf("tool calls = %#v", response.ToolCalls)
			}
			call := response.ToolCalls[0]
			if call.ID != "call_a" || call.Name != "get_weather" || string(call.Input) != `{"city":"SF"}` {
				t.Fatalf("normalized tool call = %#v", call)
			}
			if calls() != 1 {
				t.Fatalf("transport calls = %d, want 1", calls())
			}
		})
	}
}

// newContractToolTarget returns a target whose transport returns one tool call
// named get_weather with input {"city":"SF"}. CLI flavors need no fixture: tools
// are rejected in core before the runner is called.
func newContractToolTarget(t *testing.T, provider Provider, flavor Flavor) (Config, core.CommandRunner, func() int) {
	t.Helper()

	config := Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	var calls atomic.Int32
	callCount := func() int { return int(calls.Load()) }

	if flavor == FlavorCLI {
		run := func(ctx context.Context, name string, args []string, stdin []byte) ([]byte, []byte, int, error) {
			calls.Add(1)
			return nil, nil, 0, nil
		}
		return config, run, callCount
	}

	var response string
	if provider == ProviderClaude {
		response = `{"id":"c","type":"message","role":"assistant","model":"contract-model",` +
			`"content":[{"type":"tool_use","id":"call_a","name":"get_weather","input":{"city":"SF"}}],` +
			`"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
	} else {
		response = `{"id":"c","object":"response","status":"completed","model":"contract-model",` +
			`"output":[{"id":"fc","type":"function_call","status":"completed","call_id":"call_a","name":"get_weather","arguments":"{\"city\":\"SF\"}"}],` +
			`"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
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

func newContractStructuredTarget(t *testing.T, provider Provider, flavor Flavor) (Config, core.CommandRunner) {
	t.Helper()

	config := Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	if flavor == FlavorCLI {
		runner := &fakeCommandRunner{}
		switch provider {
		case ProviderClaude:
			runner.stdout = []byte(`{"type":"result","subtype":"success","result":` + strconv.Quote(structuredJSON) + `,"session_id":"s"}`)
		case ProviderOpenAI:
			runner.stdout = []byte(`{"type":"thread.started","thread_id":"t"}` + "\n" +
				`{"type":"item.completed","item":{"id":"m","type":"agent_message","text":` + strconv.Quote(structuredJSON) + `}}` + "\n" +
				`{"type":"turn.completed","usage":{}}` + "\n")
		default:
			runner.stdout = []byte(openCodeContractOutput(structuredJSON))
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

	providers := []Provider{ProviderClaude, ProviderOpenAI, ProviderOpenCode, "other"}
	flavors := []Flavor{FlavorAPI, FlavorCLI, "other"}
	for _, provider := range providers {
		for _, flavor := range flavors {
			t.Run(string(provider)+"/"+string(flavor), func(t *testing.T) {
				t.Parallel()

				_, err := New(Config{Provider: provider, Flavor: flavor, Model: "contract-model"})
				wantValid := (provider == ProviderClaude || provider == ProviderOpenAI || provider == ProviderOpenCode) &&
					(flavor == FlavorAPI || flavor == FlavorCLI) &&
					!(provider == ProviderOpenCode && flavor == FlavorAPI)
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

// openCodeContractOutput renders one text event plus a terminal step_finish with
// the usage every contract test expects.
func openCodeContractOutput(text string) string {
	return `{"type":"step_start","sessionID":"contract-session","part":{"id":"prt_1","type":"step-start"}}` + "\n" +
		`{"type":"text","sessionID":"contract-session","part":{"id":"prt_2","type":"text","text":` + strconv.Quote(text) + `}}` + "\n" +
		`{"type":"step_finish","sessionID":"contract-session","part":{"id":"prt_3","type":"step-finish","reason":"stop","tokens":{"input":12,"output":4}}}` + "\n"
}

func newContractTarget(t *testing.T, provider Provider, flavor Flavor) (Config, core.CommandRunner, func() int) {
	t.Helper()

	config := Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	var calls atomic.Int32
	callCount := func() int { return int(calls.Load()) }

	if flavor == FlavorCLI {
		runner := &fakeCommandRunner{}
		switch provider {
		case ProviderClaude:
			runner.stdout = []byte(`{
  "type":"result",
  "subtype":"success",
  "result":"contract answer",
  "session_id":"contract-session",
  "usage":{"input_tokens":12,"output_tokens":4}
}`)
		case ProviderOpenAI:
			runner.stdout = []byte(`{"type":"thread.started","thread_id":"contract-thread"}
{"type":"item.completed","item":{"id":"message-1","type":"agent_message","text":"contract answer"}}
{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":4}}
`)
		default:
			runner.stdout = []byte(openCodeContractOutput("contract answer"))
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
