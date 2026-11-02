package backends

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lao/llm-driver/internal/core"
)

// contractTarget is one provider/flavor combination every streaming contract
// case runs against.
type contractTarget struct {
	provider core.Provider
	flavor   core.Flavor
}

func contractStreamTargets() []contractTarget {
	return []contractTarget{
		{provider: core.ProviderClaude, flavor: core.FlavorAPI},
		{provider: core.ProviderClaude, flavor: core.FlavorCLI},
		{provider: core.ProviderOpenAI, flavor: core.FlavorAPI},
		{provider: core.ProviderOpenAI, flavor: core.FlavorCLI},
		{provider: core.ProviderOpenCode, flavor: core.FlavorCLI},
	}
}

// newSSETestServer serves one text/event-stream response built from raw event
// blocks, flushing each so adapters observe them incrementally.
func newSSETestServer(t *testing.T, events []string) *httptest.Server {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, event := range events {
			_, _ = io.WriteString(w, event)
			w.(http.Flusher).Flush()
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func TestContractGenerateStreamAcrossTargets(t *testing.T) {
	t.Parallel()

	for _, target := range contractStreamTargets() {
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			client := newContractStreamTarget(t, target.provider, target.flavor)
			stream, err := client.GenerateStream(context.Background(), core.Request{
				System:    "Be concise.",
				MaxTokens: 64,
				Messages:  []core.Message{core.User("Hello"), core.Assistant("Hi"), core.User("Continue")},
			})
			if err != nil {
				t.Fatalf("GenerateStream(): %v", err)
			}
			defer stream.Close()

			var accumulated strings.Builder
			var chunks int
			for stream.Next() {
				if stream.Current().Text == "" {
					t.Fatal("Current().Text is empty")
				}
				accumulated.WriteString(stream.Current().Text)
				chunks++
			}
			if err := stream.Err(); err != nil {
				t.Fatalf("Err(): %v", err)
			}
			if chunks < 2 {
				t.Fatalf("chunks = %d, want at least 2", chunks)
			}

			response, err := stream.Response()
			if err != nil {
				t.Fatalf("Response(): %v", err)
			}
			if response.Text != accumulated.String() || response.Text != "contract answer" {
				t.Fatalf("response text = %q, chunks = %q", response.Text, accumulated.String())
			}
			if response.ID == "" || response.Model != "contract-model" {
				t.Fatalf("response = %+v", response)
			}
			if response.Usage != (core.Usage{InputTokens: 12, OutputTokens: 4}) {
				t.Fatalf("usage = %+v", response.Usage)
			}
			if response.Provider != target.provider || response.Flavor != target.flavor {
				t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
			}

			// Every target normalizes rejected requests the same way.
			_, err = client.GenerateStream(context.Background(), core.Request{})
			if !errors.Is(err, core.ErrInvalidRequest) {
				t.Fatalf("invalid request error = %v, want ErrInvalidRequest", err)
			}
			var contextual *core.Error
			if !errors.As(err, &contextual) {
				t.Fatalf("invalid request error = %T %v, want *Error", err, err)
			}
			if contextual.Provider != target.provider ||
				contextual.Flavor != target.flavor ||
				contextual.Operation != "generate_stream" {
				t.Fatalf("error context = %+v", contextual)
			}
		})
	}
}

func TestContractStreamMidStreamFailureAcrossTargets(t *testing.T) {
	t.Parallel()

	for _, target := range contractStreamTargets() {
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			client := newContractFailingStreamTarget(t, target.provider, target.flavor)
			stream, err := client.GenerateStream(context.Background(), core.Request{
				MaxTokens: 64,
				Messages:  []core.Message{core.User("Hello")},
			})
			if err != nil {
				t.Fatalf("GenerateStream(): %v", err)
			}
			defer stream.Close()

			for stream.Next() {
			}
			var contextual *core.Error
			if !errors.As(stream.Err(), &contextual) {
				t.Fatalf("Err() = %T %v, want *Error", stream.Err(), stream.Err())
			}
			if contextual.Provider != target.provider ||
				contextual.Flavor != target.flavor ||
				contextual.Operation != "generate_stream" {
				t.Fatalf("error context = %+v", contextual)
			}
			if _, err := stream.Response(); err == nil {
				t.Fatal("Response() error = nil after a mid-stream failure")
			}
		})
	}
}

func TestContractStreamTruncatedOutputAcrossTargets(t *testing.T) {
	t.Parallel()

	for _, target := range contractStreamTargets() {
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			client := newContractTruncatedStreamTarget(t, target.provider, target.flavor)
			stream, err := client.GenerateStream(context.Background(), core.Request{
				MaxTokens: 64,
				Messages:  []core.Message{core.User("Hello")},
			})
			if err != nil {
				t.Fatalf("GenerateStream(): %v", err)
			}
			defer stream.Close()

			for stream.Next() {
			}
			// opencode 2.x has no terminal event for a plain text answer: the
			// truncated fixture is indistinguishable from a complete one, so it
			// must parse instead of reporting a missing step.
			if target.provider == core.ProviderOpenCode {
				if err := stream.Err(); err != nil {
					t.Fatalf("opencode 2.x text-only stream error = %v, want nil", err)
				}
				response, err := stream.Response()
				if err != nil {
					t.Fatalf("Response(): %v", err)
				}
				if response.Text == "" {
					t.Fatal("Response().Text is empty")
				}
				if response.CompletionReason != core.CompletionReasonUnverified {
					t.Fatalf("CompletionReason = %q, want %q", response.CompletionReason, core.CompletionReasonUnverified)
				}
				return
			}
			// Output that stops before the target's terminal event is incomplete on
			// every other target, so no target may report it as a finished response.
			err = stream.Err()
			if err == nil {
				_, err = stream.Response()
			}
			var contextual *core.Error
			if !errors.As(err, &contextual) {
				t.Fatalf("truncated stream error = %T %v, want *Error", err, err)
			}
			if contextual.Provider != target.provider ||
				contextual.Flavor != target.flavor ||
				contextual.Operation != "generate_stream" {
				t.Fatalf("error context = %+v", contextual)
			}
			if contextual.Code == "" {
				t.Fatalf("error code is empty: %+v", contextual)
			}
		})
	}
}

// TestContractStreamStructuredOutputAcrossTargets streams the same
// OutputSchema request through every target: chunks stay raw JSON-text
// deltas and every target reports the same StructuredOutput at Finish.
func TestContractStreamStructuredOutputAcrossTargets(t *testing.T) {
	t.Parallel()

	for _, target := range contractStreamTargets() {
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			client := newContractStructuredStreamTarget(t, target.provider, target.flavor)
			stream, err := client.GenerateStream(context.Background(), core.Request{
				MaxTokens:    64,
				Messages:     []core.Message{core.User("q")},
				OutputSchema: &core.OutputSchema{Name: "answer", Schema: testSchema},
			})
			if err != nil {
				t.Fatalf("GenerateStream(): %v", err)
			}
			defer stream.Close()

			var accumulated strings.Builder
			for stream.Next() {
				accumulated.WriteString(stream.Current().Text)
			}
			if err := stream.Err(); err != nil {
				t.Fatalf("Err(): %v", err)
			}
			response, err := stream.Response()
			if err != nil {
				t.Fatalf("Response(): %v", err)
			}
			if accumulated.String() != testJSON {
				t.Fatalf("chunks = %q, want %q", accumulated.String(), testJSON)
			}
			if response.Text != testJSON || string(response.StructuredOutput) != testJSON {
				t.Fatalf("response Text=%q StructuredOutput=%q, want %q", response.Text, response.StructuredOutput, testJSON)
			}
		})
	}
}

func newContractStructuredStreamTarget(t *testing.T, provider core.Provider, flavor core.Flavor) *streamClient {
	t.Helper()

	config := core.Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	if flavor == core.FlavorCLI {
		var lines []string
		switch provider {
		case core.ProviderClaude:
			lines = []string{
				`{"type":"system","subtype":"init","session_id":"contract-session"}`,
				`{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}}`,
				`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\"answer\":"}}}`,
				`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\"42\"}"}}}`,
				`{"type":"result","subtype":"success","result":"{\"answer\":\"42\"}","session_id":"contract-session","usage":{"input_tokens":1,"output_tokens":1}}`,
			}
		case core.ProviderOpenCode:
			lines = []string{
				`{"type":"text","sessionID":"contract-session","part":{"id":"prt_1","type":"text","text":"{\"answer\":"}}`,
				`{"type":"text","sessionID":"contract-session","part":{"id":"prt_2","type":"text","text":"\"42\"}"}}`,
				`{"type":"step_finish","sessionID":"contract-session","part":{"id":"prt_3","type":"step-finish","reason":"stop","tokens":{"input":1,"output":1}}}`,
			}
		default:
			lines = []string{
				`{"type":"thread.started","thread_id":"contract-thread"}`,
				`{"type":"item.updated","item":{"id":"m","type":"agent_message","text":"{\"answer\":"}}`,
				`{"type":"item.updated","item":{"id":"m","type":"agent_message","text":"{\"answer\":\"42\"}"}}`,
				`{"type":"item.completed","item":{"id":"m","type":"agent_message","text":"{\"answer\":\"42\"}"}}`,
				`{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}`,
			}
		}
		runner := &fakeStreamingRunner{lines: lines}
		return newCLIStreamClient(config, runner.run)
	}

	var events []string
	if provider == core.ProviderClaude {
		events = []string{
			"event: message_start\n" + `data: {"type":"message_start","message":{"id":"contract-message","type":"message","role":"assistant","model":"contract-model","content":[],"usage":{"input_tokens":1,"output_tokens":1}}}` + "\n\n",
			"event: content_block_start\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n",
			"event: content_block_delta\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\"answer\":"}}` + "\n\n",
			"event: content_block_delta\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\"42\"}"}}` + "\n\n",
			"event: content_block_stop\n" + `data: {"type":"content_block_stop","index":0}` + "\n\n",
			"event: message_delta\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}` + "\n\n",
			"event: message_stop\n" + `data: {"type":"message_stop"}` + "\n\n",
		}
	} else {
		events = []string{
			"event: response.output_text.delta\n" + `data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"m","output_index":0,"content_index":0,"delta":"{\"answer\":","logprobs":[]}` + "\n\n",
			"event: response.output_text.delta\n" + `data: {"type":"response.output_text.delta","sequence_number":2,"item_id":"m","output_index":0,"content_index":0,"delta":"\"42\"}","logprobs":[]}` + "\n\n",
			"event: response.completed\n" + `data: {"type":"response.completed","sequence_number":3,"response":{"id":"contract-response","object":"response","status":"completed","model":"contract-model",` +
				`"output":[{"id":"m","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"{\"answer\":\"42\"}","annotations":[],"logprobs":[]}]}],` +
				`"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}` + "\n\n",
		}
	}
	return newContractAPIStreamTarget(t, config, events)
}

func contractStreamLines(provider core.Provider) []string {
	switch provider {
	case core.ProviderClaude:
		return []string{
			`{"type":"system","subtype":"init","session_id":"contract-session"}`,
			`{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}}`,
			`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"contract "}}}`,
			`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"answer"}}}`,
			`{"type":"result","subtype":"success","result":"contract answer","session_id":"contract-session",` +
				`"usage":{"input_tokens":12,"output_tokens":4}}`,
		}
	case core.ProviderOpenCode:
		return []string{
			`{"type":"step_start","sessionID":"contract-session","part":{"id":"prt_1","type":"step-start"}}`,
			`{"type":"text","sessionID":"contract-session","part":{"id":"prt_2","type":"text","text":"contract "}}`,
			`{"type":"text","sessionID":"contract-session","part":{"id":"prt_3","type":"text","text":"answer"}}`,
			`{"type":"step_finish","sessionID":"contract-session","part":{"id":"prt_4","type":"step-finish","reason":"stop","tokens":{"input":12,"output":4}}}`,
		}
	default:
		return []string{
			`{"type":"thread.started","thread_id":"contract-thread"}`,
			`{"type":"item.updated","item":{"id":"message-1","type":"agent_message","text":"contract "}}`,
			`{"type":"item.updated","item":{"id":"message-1","type":"agent_message","text":"contract answer"}}`,
			`{"type":"item.completed","item":{"id":"message-1","type":"agent_message","text":"contract answer"}}`,
			`{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":4}}`,
		}
	}
}

func contractSSEEvents(provider core.Provider) []string {
	if provider == core.ProviderClaude {
		return []string{
			"event: message_start\n" +
				`data: {"type":"message_start","message":{"id":"contract-message","type":"message","role":"assistant","model":"contract-model","content":[],"usage":{"input_tokens":12,"output_tokens":1}}}` + "\n\n",
			"event: content_block_start\n" +
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n",
			"event: content_block_delta\n" +
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"contract "}}` + "\n\n",
			"event: content_block_delta\n" +
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"answer"}}` + "\n\n",
			"event: content_block_stop\n" + `data: {"type":"content_block_stop","index":0}` + "\n\n",
			"event: message_delta\n" +
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}` + "\n\n",
			"event: message_stop\n" + `data: {"type":"message_stop"}` + "\n\n",
		}
	}
	return []string{
		"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"message-1","output_index":0,"content_index":0,"delta":"contract ","logprobs":[]}` + "\n\n",
		"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","sequence_number":2,"item_id":"message-1","output_index":0,"content_index":0,"delta":"answer","logprobs":[]}` + "\n\n",
		"event: response.completed\n" +
			`data: {"type":"response.completed","sequence_number":3,"response":{"id":"contract-response","object":"response","status":"completed","model":"contract-model",` +
			`"output":[{"id":"message-1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"contract answer","annotations":[],"logprobs":[]}]}],` +
			`"usage":{"input_tokens":12,"output_tokens":4,"total_tokens":16}}}` + "\n\n",
	}
}

// newContractAPIStreamTarget points an API backend at a test server replaying
// the given SSE events.
func newContractAPIStreamTarget(t *testing.T, config core.Config, events []string) *streamClient {
	t.Helper()

	server := newSSETestServer(t, events)
	config.APIKey = "contract-key"
	config.BaseURL = server.URL
	apiConfig := apiBackendConfig{model: config.Model, apiKey: config.APIKey, baseURL: config.BaseURL}
	var backend core.Backend
	if config.Provider == core.ProviderClaude {
		backend = newAnthropicBackend(apiConfig)
	} else {
		backend = newOpenAIBackend(apiConfig)
	}
	return &streamClient{
		provider: config.Provider,
		flavor:   config.Flavor,
		model:    config.Model,
		backend:  backend,
	}
}

func newContractStreamTarget(t *testing.T, provider core.Provider, flavor core.Flavor) *streamClient {
	t.Helper()

	config := core.Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	if flavor == core.FlavorCLI {
		runner := &fakeStreamingRunner{lines: contractStreamLines(provider)}
		return newCLIStreamClient(config, runner.run)
	}

	return newContractAPIStreamTarget(t, config, contractSSEEvents(provider))
}

func newContractFailingStreamTarget(t *testing.T, provider core.Provider, flavor core.Flavor) *streamClient {
	t.Helper()

	config := core.Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	if flavor == core.FlavorCLI {
		runner := &fakeStreamingRunner{lines: append(contractStreamLines(provider)[:3], "not-json")}
		return newCLIStreamClient(config, runner.run)
	}

	events := contractSSEEvents(provider)[:2]
	if provider == core.ProviderClaude {
		events = append(events, "event: error\n"+
			`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`+"\n\n")
	} else {
		events = append(events, "event: error\n"+
			`data: {"type":"error","sequence_number":2,"code":"server_error","message":"upstream failed","param":null}`+"\n\n")
	}
	return newContractAPIStreamTarget(t, config, events)
}

// newContractTruncatedStreamTarget stops each target's output right after the
// text deltas, before the event that marks the response complete.
func newContractTruncatedStreamTarget(t *testing.T, provider core.Provider, flavor core.Flavor) *streamClient {
	t.Helper()

	config := core.Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	if flavor == core.FlavorCLI {
		lines := contractStreamLines(provider)
		if provider == core.ProviderClaude {
			lines = lines[:4] // no result line
		} else {
			lines = lines[:3] // no item.completed, no turn.completed
		}
		runner := &fakeStreamingRunner{lines: lines}
		return newCLIStreamClient(config, runner.run)
	}

	events := contractSSEEvents(provider)
	if provider == core.ProviderClaude {
		events = events[:4] // no content_block_stop, message_delta, or message_stop
	} else {
		events = events[:2] // no response.completed
	}
	return newContractAPIStreamTarget(t, config, events)
}
