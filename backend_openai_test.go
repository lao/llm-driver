package llmwrapper

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func newOpenAITestServer(t *testing.T, status int, response string) (*httptest.Server, *capturedHTTPRequest) {
	t.Helper()

	captured := new(capturedHTTPRequest)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		captured.method = request.Method
		captured.path = request.URL.Path
		captured.header = request.Header.Clone()
		if err := json.Unmarshal(body, &captured.body); err != nil {
			t.Errorf("decode request: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, response)
	}))
	t.Cleanup(server.Close)
	return server, captured
}

func TestOpenAIGenerateNormalizesAPIError(t *testing.T) {
	t.Parallel()

	server, _ := newOpenAITestServer(t, http.StatusBadRequest,
		`{"error":{"message":"bad request","type":"invalid_request_error","param":null,"code":"invalid_model"}}`)
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

	_, err = client.Generate(context.Background(), Request{MaxTokens: 16, Messages: []Message{User("hello")}})
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Generate() error = %T %v, want *Error", err, err)
	}
	if target.Provider != ProviderOpenAI || target.Flavor != FlavorAPI || target.Operation != "generate" {
		t.Fatalf("error context = %+v", target)
	}
	if target.Status != http.StatusBadRequest || target.Code != "invalid_model" {
		t.Fatalf("error status/code = %d/%q", target.Status, target.Code)
	}
}

func TestOpenAIGeneratePreservesTransportError(t *testing.T) {
	t.Parallel()

	client, err := New(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorAPI,
		Model:    "gpt-test",
		APIKey:   "test-key",
		BaseURL:  "https://example.test",
		HTTPClient: &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			return nil, context.Canceled
		})},
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = client.Generate(context.Background(), Request{MaxTokens: 16, Messages: []Message{User("hello")}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate() error = %v, want context.Canceled", err)
	}
	var target *Error
	if !errors.As(err, &target) || target.Provider != ProviderOpenAI || target.Flavor != FlavorAPI {
		t.Fatalf("Generate() error = %#v, want contextual *Error", err)
	}
}

func TestOpenAIGenerateMapsRequestAndResponse(t *testing.T) {
	t.Parallel()

	const response = `{
  "id":"resp_123",
  "object":"response",
  "created_at":1750000000,
  "status":"completed",
  "model":"gpt-test",
  "output":[{
    "id":"msg_123",
    "type":"message",
    "status":"completed",
    "role":"assistant",
    "content":[
      {"type":"output_text","text":"Hello, ","annotations":[],"logprobs":[]},
      {"type":"output_text","text":"world","annotations":[],"logprobs":[]}
    ]
  }],
  "usage":{
    "input_tokens":10,
    "input_tokens_details":{"cached_tokens":3},
    "output_tokens":7,
    "output_tokens_details":{"reasoning_tokens":2},
    "total_tokens":17
  }
}`
	server, captured := newOpenAITestServer(t, http.StatusOK, response)
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

	result, err := client.Generate(context.Background(), Request{
		System:    "Be concise.",
		MaxTokens: 64,
		Messages:  []Message{User("hello"), Assistant("hi"), User("continue")},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	if captured.method != http.MethodPost || captured.path != "/responses" {
		t.Fatalf("request = %s %s", captured.method, captured.path)
	}
	if captured.header.Get("Authorization") != "Bearer test-key" {
		t.Fatalf("Authorization = %q", captured.header.Get("Authorization"))
	}
	if captured.body["model"] != "gpt-test" || captured.body["max_output_tokens"] != float64(64) {
		t.Fatalf("request body = %+v", captured.body)
	}
	if captured.body["instructions"] != "Be concise." {
		t.Fatalf("instructions = %#v", captured.body["instructions"])
	}

	input, ok := captured.body["input"].([]any)
	if !ok || len(input) != 3 {
		t.Fatalf("input = %#v", captured.body["input"])
	}
	for i, want := range []Message{User("hello"), Assistant("hi"), User("continue")} {
		message := input[i].(map[string]any)
		content := message["content"].([]any)[0].(map[string]any)
		if message["role"] != string(want.Role) || content["text"] != want.Text {
			t.Errorf("input[%d] = %#v", i, message)
		}
	}

	if result.ID != "resp_123" || result.Model != "gpt-test" || result.Text != "Hello, world" {
		t.Fatalf("response = %+v", result)
	}
	if result.Provider != ProviderOpenAI || result.Flavor != FlavorAPI {
		t.Fatalf("response target = %s/%s", result.Provider, result.Flavor)
	}
	wantUsage := (Usage{InputTokens: 10, OutputTokens: 7, CachedInputTokens: 3, ReasoningTokens: 2})
	if result.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", result.Usage, wantUsage)
	}
}
