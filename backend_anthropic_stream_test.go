package llmdriver

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func anthropicSSEEvents() []string {
	return []string{
		"event: message_start\n" +
			`data: {"type":"message_start","message":{"id":"msg_stream","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":3,"cache_creation_input_tokens":2}}}` + "\n\n",
		"event: content_block_start\n" +
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n",
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello, "}}` + "\n\n",
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}` + "\n\n",
		"event: content_block_stop\n" +
			`data: {"type":"content_block_stop","index":0}` + "\n\n",
		"event: message_delta\n" +
			`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":5}}` + "\n\n",
		"event: message_stop\n" +
			`data: {"type":"message_stop"}` + "\n\n",
	}
}

func newAnthropicStreamClient(t *testing.T, status int, events []string) (*Client, *capturedHTTPRequest) {
	t.Helper()

	server, captured := newSSETestServer(t, status, events)
	client, err := New(Config{
		Provider: ProviderClaude,
		Flavor:   FlavorAPI,
		Model:    "claude-test",
		APIKey:   "test-key",
		BaseURL:  server.URL,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return client, captured
}

func TestAnthropicStreamMapsDeltasAndResponse(t *testing.T) {
	t.Parallel()

	client, captured := newAnthropicStreamClient(t, http.StatusOK, anthropicSSEEvents())
	stream, err := client.GenerateStream(context.Background(), Request{
		System:    "Be concise.",
		MaxTokens: 64,
		Messages:  []Message{User("hello")},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	chunks, err, response := collectStream(t, stream)
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if strings.Join(chunks, "|") != "Hello, |world" {
		t.Fatalf("chunks = %#v", chunks)
	}

	if captured.method != http.MethodPost || captured.path != "/v1/messages" {
		t.Fatalf("request = %s %s", captured.method, captured.path)
	}
	if captured.body["stream"] != true {
		t.Fatalf("request body = %+v, want stream:true", captured.body)
	}
	if captured.body["model"] != "claude-test" || captured.body["max_tokens"] != float64(64) {
		t.Fatalf("request body = %+v", captured.body)
	}

	if response.ID != "msg_stream" || response.Model != "claude-test" || response.Text != "Hello, world" {
		t.Fatalf("response = %+v", response)
	}
	if response.CompletionReason != CompletionReasonMaxTokens {
		t.Fatalf("completion reason = %q", response.CompletionReason)
	}
	if response.Provider != ProviderClaude || response.Flavor != FlavorAPI {
		t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
	}
	wantUsage := Usage{InputTokens: 10, OutputTokens: 5, CachedInputTokens: 3, CacheCreationInputTokens: 2}
	if response.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", response.Usage, wantUsage)
	}
}

func TestAnthropicStreamErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		status     int
		events     []string
		wantStatus int
		wantCode   string
		startFails bool
	}{
		{
			name:   "mid-stream error event",
			status: http.StatusOK,
			events: append(anthropicSSEEvents()[:3],
				"event: error\n"+
					`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`+"\n\n"),
			wantStatus: http.StatusOK,
			wantCode:   "overloaded_error",
		},
		{
			name:   "malformed SSE payload",
			status: http.StatusOK,
			events: append(anthropicSSEEvents()[:1],
				"event: content_block_delta\ndata: {not json\n\n"),
		},
		{
			name:       "non-200 start",
			status:     http.StatusTooManyRequests,
			events:     []string{`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`},
			wantStatus: http.StatusTooManyRequests,
			wantCode:   "rate_limit_error",
			startFails: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client, _ := newAnthropicStreamClient(t, test.status, test.events)
			stream, err := client.GenerateStream(context.Background(), Request{
				MaxTokens: 64,
				Messages:  []Message{User("hello")},
			})
			if test.startFails {
				if stream != nil {
					t.Fatal("GenerateStream() returned a stream for a failed start")
				}
			} else {
				if err != nil {
					t.Fatalf("GenerateStream(): %v", err)
				}
				defer stream.Close()
				_, err, _ = collectStream(t, stream)
			}

			var target *Error
			if !errors.As(err, &target) {
				t.Fatalf("stream error = %T %v, want *Error", err, err)
			}
			if target.Provider != ProviderClaude || target.Flavor != FlavorAPI || target.Operation != "generate_stream" {
				t.Fatalf("error context = %+v", target)
			}
			if target.Status != test.wantStatus || target.Code != test.wantCode {
				t.Fatalf("status/code = %d/%q, want %d/%q", target.Status, target.Code, test.wantStatus, test.wantCode)
			}
		})
	}
}

func TestAnthropicStreamCloseStopsIteration(t *testing.T) {
	t.Parallel()

	client, _ := newAnthropicStreamClient(t, http.StatusOK, anthropicSSEEvents())
	stream, err := client.GenerateStream(context.Background(), Request{
		MaxTokens: 64,
		Messages:  []Message{User("hello")},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}

	if !stream.Next() {
		t.Fatalf("Next() = false, want the first chunk: %v", stream.Err())
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close(): %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("second Close(): %v", err)
	}
	if stream.Next() {
		t.Fatal("Next() = true after Close()")
	}
}

func TestAnthropicStreamContextCancellation(t *testing.T) {
	t.Parallel()

	// The handler holds the response open after the first delta so cancellation
	// interrupts a pending read instead of racing a finished stream.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range anthropicSSEEvents()[:3] {
			_, _ = io.WriteString(w, event)
			w.(http.Flusher).Flush()
		}
		<-request.Context().Done()
	}))
	t.Cleanup(server.Close)

	client, err := New(Config{
		Provider: ProviderClaude,
		Flavor:   FlavorAPI,
		Model:    "claude-test",
		APIKey:   "test-key",
		BaseURL:  server.URL,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.GenerateStream(ctx, Request{
		MaxTokens: 64,
		Messages:  []Message{User("hello")},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	if !stream.Next() {
		t.Fatalf("Next() = false, want the first chunk: %v", stream.Err())
	}
	cancel()
	for stream.Next() {
	}
	if !errors.Is(stream.Err(), context.Canceled) {
		t.Fatalf("Err() = %v, want context.Canceled", stream.Err())
	}
}

func TestAnthropicStreamTruncatedOutputIsAnError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		events []string
	}{
		// A 200 response whose body ends early looks like a clean EOF to the SDK,
		// so the adapter — not the transport — has to notice the missing terminal
		// event instead of reporting a partial answer as a complete one.
		{name: "truncated after deltas", events: anthropicSSEEvents()[:4]},
		{name: "no events at all", events: nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client, _ := newAnthropicStreamClient(t, http.StatusOK, test.events)
			stream, err := client.GenerateStream(context.Background(), Request{
				MaxTokens: 64,
				Messages:  []Message{User("hello")},
			})
			if err != nil {
				t.Fatalf("GenerateStream(): %v", err)
			}
			defer stream.Close()

			_, err, response := collectStream(t, stream)
			if err == nil {
				t.Fatalf("stream error = nil for a truncated stream, response = %+v", response)
			}
			var target *Error
			if !errors.As(err, &target) {
				t.Fatalf("stream error = %T %v, want *Error", err, err)
			}
			if target.Provider != ProviderClaude || target.Flavor != FlavorAPI || target.Operation != "generate_stream" {
				t.Fatalf("error context = %+v", target)
			}
			if target.Code != "missing_message_completion" {
				t.Fatalf("code = %q, want missing_message_completion", target.Code)
			}
		})
	}
}

func TestAnthropicStreamOutOfOrderEvents(t *testing.T) {
	t.Parallel()

	// A text delta with no content block to attach to cannot be accumulated, so
	// the adapter must fail instead of quietly dropping response text.
	client, _ := newAnthropicStreamClient(t, http.StatusOK, []string{
		anthropicSSEEvents()[0],
		anthropicSSEEvents()[2],
	})
	stream, err := client.GenerateStream(context.Background(), Request{
		MaxTokens: 64,
		Messages:  []Message{User("hello")},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	chunks, err, response := collectStream(t, stream)
	if err == nil {
		t.Fatalf("stream error = nil for out-of-order events, chunks = %#v, response = %+v", chunks, response)
	}
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("stream error = %T %v, want *Error", err, err)
	}
	if target.Provider != ProviderClaude || target.Flavor != FlavorAPI || target.Operation != "generate_stream" {
		t.Fatalf("error context = %+v", target)
	}
}
