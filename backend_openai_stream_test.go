package llmdriver

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

// SSE frames are line-oriented, so response fixtures stay on one line.
const openAIStreamCompletedResponse = `{"id":"resp_stream","object":"response","status":"completed","model":"gpt-test",` +
	`"output":[{"id":"message-1","type":"message","status":"completed","role":"assistant",` +
	`"content":[{"type":"output_text","text":"Hello, world","annotations":[],"logprobs":[]}]}],` +
	`"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":3,"cache_write_tokens":2},` +
	`"output_tokens":5,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":15}}`

func openAISSEEvents() []string {
	return []string{
		"event: response.created\n" +
			`data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_stream","object":"response","status":"in_progress","model":"gpt-test","output":[]}}` + "\n\n",
		"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"message-1","output_index":0,"content_index":0,"delta":"Hello, ","logprobs":[]}` + "\n\n",
		"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","sequence_number":2,"item_id":"message-1","output_index":0,"content_index":0,"delta":"world","logprobs":[]}` + "\n\n",
		"event: response.completed\n" +
			`data: {"type":"response.completed","sequence_number":3,"response":` + openAIStreamCompletedResponse + "}\n\n",
	}
}

func newOpenAIStreamClient(t *testing.T, status int, events []string) (*Client, *capturedHTTPRequest) {
	t.Helper()

	server, captured := newSSETestServer(t, status, events)
	client, err := New(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorAPI,
		Model:    "gpt-test",
		APIKey:   "test-key",
		BaseURL:  server.URL,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return client, captured
}

func TestOpenAIStreamMapsDeltasAndResponse(t *testing.T) {
	t.Parallel()

	client, captured := newOpenAIStreamClient(t, http.StatusOK, openAISSEEvents())
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

	if captured.method != http.MethodPost || captured.path != "/responses" {
		t.Fatalf("request = %s %s", captured.method, captured.path)
	}
	if captured.body["stream"] != true {
		t.Fatalf("request body = %+v, want stream:true", captured.body)
	}
	if captured.body["model"] != "gpt-test" || captured.body["instructions"] != "Be concise." {
		t.Fatalf("request body = %+v", captured.body)
	}

	if response.ID != "resp_stream" || response.Model != "gpt-test" || response.Text != "Hello, world" {
		t.Fatalf("response = %+v", response)
	}
	if response.CompletionReason != CompletionReasonStop {
		t.Fatalf("completion reason = %q", response.CompletionReason)
	}
	if response.Provider != ProviderOpenAI || response.Flavor != FlavorAPI {
		t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
	}
	wantUsage := Usage{
		InputTokens:              10,
		OutputTokens:             5,
		CachedInputTokens:        3,
		CacheCreationInputTokens: 2,
		ReasoningTokens:          1,
	}
	if response.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", response.Usage, wantUsage)
	}
}

func TestOpenAIStreamStructuredOutput(t *testing.T) {
	t.Parallel()

	delta := func(seq int, text string) string {
		return "event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","sequence_number":` + strconv.Itoa(seq) +
			`,"item_id":"message-1","output_index":0,"content_index":0,"delta":` + strconv.Quote(text) + `,"logprobs":[]}` + "\n\n"
	}
	completed := `{"id":"resp_stream","object":"response","status":"completed","model":"gpt-test",` +
		`"output":[{"id":"message-1","type":"message","status":"completed","role":"assistant",` +
		`"content":[{"type":"output_text","text":` + strconv.Quote(structuredJSON) + `,"annotations":[],"logprobs":[]}]}],` +
		`"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}`
	events := []string{
		openAISSEEvents()[0],
		delta(1, `{"answer":`),
		delta(2, `"42"}`),
		"event: response.completed\n" + `data: {"type":"response.completed","sequence_number":3,"response":` + completed + "}\n\n",
	}

	client, _ := newOpenAIStreamClient(t, http.StatusOK, events)
	stream, err := client.GenerateStream(context.Background(), Request{
		MaxTokens:    64,
		Messages:     []Message{User("q")},
		OutputSchema: &OutputSchema{Name: "answer", Schema: structuredSchema},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	chunks, err, response := collectStream(t, stream)
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if strings.Join(chunks, "") != structuredJSON {
		t.Fatalf("chunks = %#v, want the JSON document", chunks)
	}
	if response.Text != structuredJSON || string(response.StructuredOutput) != structuredJSON {
		t.Fatalf("response Text=%q StructuredOutput=%q, want %q", response.Text, response.StructuredOutput, structuredJSON)
	}
}

func TestOpenAIStreamErrors(t *testing.T) {
	t.Parallel()

	incomplete := strings.NewReplacer(
		`"status":"completed"`, `"status":"incomplete"`,
	).Replace(openAIStreamCompletedResponse)

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
			events: append(openAISSEEvents()[:2],
				"event: error\n"+
					`data: {"type":"error","sequence_number":2,"code":"server_error","message":"upstream failed","param":null}`+"\n\n"),
			wantCode: "server_error",
		},
		{
			name:   "response.failed event",
			status: http.StatusOK,
			events: append(openAISSEEvents()[:2],
				"event: response.failed\n"+
					`data: {"type":"response.failed","sequence_number":2,"response":{"id":"resp_stream","object":"response","status":"failed","model":"gpt-test","output":[],"error":{"code":"rate_limit_exceeded","message":"slow down"}}}`+"\n\n"),
			wantCode: "rate_limit_exceeded",
		},
		{
			name:     "incomplete response",
			status:   http.StatusOK,
			events:   append(openAISSEEvents()[:2], "event: response.incomplete\n"+`data: {"type":"response.incomplete","sequence_number":2,"response":`+incomplete+"}\n\n"),
			wantCode: "incomplete",
		},
		{
			name:     "missing completion",
			status:   http.StatusOK,
			events:   openAISSEEvents()[:3],
			wantCode: "missing_response_completion",
		},
		{
			name:   "malformed SSE payload",
			status: http.StatusOK,
			events: append(openAISSEEvents()[:1],
				"event: response.output_text.delta\ndata: {not json\n\n"),
		},
		{
			name:       "non-200 start",
			status:     http.StatusTooManyRequests,
			events:     []string{`{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"slow down"}}`},
			wantStatus: http.StatusTooManyRequests,
			wantCode:   "rate_limit_exceeded",
			startFails: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client, _ := newOpenAIStreamClient(t, test.status, test.events)
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
			if target.Provider != ProviderOpenAI || target.Flavor != FlavorAPI || target.Operation != "generate_stream" {
				t.Fatalf("error context = %+v", target)
			}
			if target.Status != test.wantStatus || target.Code != test.wantCode {
				t.Fatalf("status/code = %d/%q, want %d/%q", target.Status, target.Code, test.wantStatus, test.wantCode)
			}
		})
	}
}

func TestOpenAIStreamContextCancellation(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range openAISSEEvents()[:2] {
			_, _ = io.WriteString(w, event)
			w.(http.Flusher).Flush()
		}
		<-request.Context().Done()
	}))
	t.Cleanup(server.Close)

	client, err := New(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorAPI,
		Model:    "gpt-test",
		APIKey:   "test-key",
		BaseURL:  server.URL,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.GenerateStream(ctx, Request{MaxTokens: 64, Messages: []Message{User("hello")}})
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

func TestOpenAIStreamIncompleteResponseDetails(t *testing.T) {
	t.Parallel()

	incompleteWith := func(reason string) string {
		return strings.NewReplacer(
			`"status":"completed"`, `"status":"incomplete"`,
			`"object":"response"`, `"object":"response","incomplete_details":{"reason":"`+reason+`"}`,
		).Replace(openAIStreamCompletedResponse)
	}
	incompleteEvent := func(reason string) []string {
		return append(openAISSEEvents()[:3],
			"event: response.incomplete\n"+
				`data: {"type":"response.incomplete","sequence_number":3,"response":`+incompleteWith(reason)+"}\n\n")
	}

	t.Run("max output tokens is a successful truncation", func(t *testing.T) {
		t.Parallel()

		client, _ := newOpenAIStreamClient(t, http.StatusOK, incompleteEvent("max_output_tokens"))
		stream, err := client.GenerateStream(context.Background(), Request{
			MaxTokens: 4,
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
		if response.Text != "Hello, world" {
			t.Fatalf("response text = %q", response.Text)
		}
		if response.CompletionReason != CompletionReasonMaxTokens {
			t.Fatalf("completion reason = %q, want %q", response.CompletionReason, CompletionReasonMaxTokens)
		}
	})

	t.Run("content filter is a successful truncation", func(t *testing.T) {
		t.Parallel()

		client, _ := newOpenAIStreamClient(t, http.StatusOK, incompleteEvent("content_filter"))
		stream, err := client.GenerateStream(context.Background(), Request{
			MaxTokens: 64,
			Messages:  []Message{User("hello")},
		})
		if err != nil {
			t.Fatalf("GenerateStream(): %v", err)
		}
		defer stream.Close()

		_, err, response := collectStream(t, stream)
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if response.CompletionReason != CompletionReasonContentFilter {
			t.Fatalf("completion reason = %q, want %q", response.CompletionReason, CompletionReasonContentFilter)
		}
	})

	t.Run("other incomplete reasons are errors", func(t *testing.T) {
		t.Parallel()

		client, _ := newOpenAIStreamClient(t, http.StatusOK, incompleteEvent("moderation"))
		stream, err := client.GenerateStream(context.Background(), Request{
			MaxTokens: 64,
			Messages:  []Message{User("hello")},
		})
		if err != nil {
			t.Fatalf("GenerateStream(): %v", err)
		}
		defer stream.Close()

		_, err, _ = collectStream(t, stream)
		var target *Error
		if !errors.As(err, &target) {
			t.Fatalf("stream error = %T %v, want *Error", err, err)
		}
		if target.Provider != ProviderOpenAI || target.Flavor != FlavorAPI || target.Operation != "generate_stream" {
			t.Fatalf("error context = %+v", target)
		}
		if target.Code != "moderation" {
			t.Fatalf("code = %q, want moderation", target.Code)
		}
		if !strings.Contains(target.Message, "moderation") {
			t.Fatalf("message = %q, want it to name the incomplete reason", target.Message)
		}
	})
}

func TestOpenAIStreamErrorEventWithoutDetails(t *testing.T) {
	t.Parallel()

	client, _ := newOpenAIStreamClient(t, http.StatusOK, append(openAISSEEvents()[:2],
		"event: error\n"+`data: {"type":"error","sequence_number":2}`+"\n\n"))
	stream, err := client.GenerateStream(context.Background(), Request{
		MaxTokens: 64,
		Messages:  []Message{User("hello")},
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	_, err, _ = collectStream(t, stream)
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("stream error = %T %v, want *Error", err, err)
	}
	// An error event carrying neither code nor message still has to be reportable.
	if target.Code == "" || target.Message == "" {
		t.Fatalf("error = %+v, want a fallback code and message", target)
	}
}
