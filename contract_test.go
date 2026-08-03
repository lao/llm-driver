package llmwrapper

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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

func newContractTarget(t *testing.T, provider Provider, flavor Flavor) (Config, commandRunner, func() int) {
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

func TestContractGenerateStreamAcrossFourTargets(t *testing.T) {
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
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			client := newContractStreamTarget(t, target.provider, target.flavor)
			stream, err := client.GenerateStream(context.Background(), Request{
				System:    "Be concise.",
				MaxTokens: 64,
				Messages:  []Message{User("Hello"), Assistant("Hi"), User("Continue")},
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
			if response.Usage != (Usage{InputTokens: 12, OutputTokens: 4}) {
				t.Fatalf("usage = %+v", response.Usage)
			}
			if response.Provider != target.provider || response.Flavor != target.flavor {
				t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
			}

			// Every target normalizes rejected requests the same way.
			_, err = client.GenerateStream(context.Background(), Request{})
			if !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("invalid request error = %v, want ErrInvalidRequest", err)
			}
			var contextual *Error
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

func TestContractStreamMidStreamFailureAcrossFourTargets(t *testing.T) {
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
		t.Run(string(target.provider)+"/"+string(target.flavor), func(t *testing.T) {
			t.Parallel()

			client := newContractFailingStreamTarget(t, target.provider, target.flavor)
			stream, err := client.GenerateStream(context.Background(), Request{
				MaxTokens: 64,
				Messages:  []Message{User("Hello")},
			})
			if err != nil {
				t.Fatalf("GenerateStream(): %v", err)
			}
			defer stream.Close()

			for stream.Next() {
			}
			var contextual *Error
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

func contractStreamLines(provider Provider) []string {
	if provider == ProviderClaude {
		return []string{
			`{"type":"system","subtype":"init","session_id":"contract-session"}`,
			`{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}}`,
			`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"contract "}}}`,
			`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"answer"}}}`,
			`{"type":"result","subtype":"success","result":"contract answer","session_id":"contract-session",` +
				`"usage":{"input_tokens":12,"output_tokens":4}}`,
		}
	}
	return []string{
		`{"type":"thread.started","thread_id":"contract-thread"}`,
		`{"type":"item.updated","item":{"id":"message-1","type":"agent_message","text":"contract "}}`,
		`{"type":"item.updated","item":{"id":"message-1","type":"agent_message","text":"contract answer"}}`,
		`{"type":"item.completed","item":{"id":"message-1","type":"agent_message","text":"contract answer"}}`,
		`{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":4}}`,
	}
}

func contractSSEEvents(provider Provider) []string {
	if provider == ProviderClaude {
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

func newContractStreamTarget(t *testing.T, provider Provider, flavor Flavor) *Client {
	t.Helper()

	config := Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	if flavor == FlavorCLI {
		runner := &fakeStreamingRunner{lines: contractStreamLines(provider)}
		return newCLIStreamClient(config, runner.run)
	}

	server, _ := newSSETestServer(t, http.StatusOK, contractSSEEvents(provider))
	config.APIKey = "contract-key"
	config.BaseURL = server.URL
	client, err := New(config)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return client
}

func newContractFailingStreamTarget(t *testing.T, provider Provider, flavor Flavor) *Client {
	t.Helper()

	config := Config{Provider: provider, Flavor: flavor, Model: "contract-model"}
	if flavor == FlavorCLI {
		runner := &fakeStreamingRunner{lines: append(contractStreamLines(provider)[:3], "not-json")}
		return newCLIStreamClient(config, runner.run)
	}

	events := contractSSEEvents(provider)[:2]
	if provider == ProviderClaude {
		events = append(events, "event: error\n"+
			`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`+"\n\n")
	} else {
		events = append(events, "event: error\n"+
			`data: {"type":"error","sequence_number":2,"code":"server_error","message":"upstream failed","param":null}`+"\n\n")
	}
	server, _ := newSSETestServer(t, http.StatusOK, events)
	config.APIKey = "contract-key"
	config.BaseURL = server.URL
	client, err := New(config)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return client
}
