package llmdriver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"testing"
)

// structuredSchema is the JSON Schema every structured-output test sends, and
// structuredJSON is the JSON document the model returns for it. They are shared
// so the contract tests can assert an identical normalized shape across targets.
var structuredSchema = json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)

const structuredJSON = `{"answer":"42"}`

func TestAnthropicGenerateNormalizesAPIError(t *testing.T) {
	t.Parallel()

	server, _ := newAPITestServer(t, http.StatusBadRequest,
		`{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`)
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

	_, err = client.Generate(context.Background(), Request{MaxTokens: 16, Messages: []Message{User("hello")}})
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Generate() error = %T %v, want *Error", err, err)
	}
	if target.Provider != ProviderClaude || target.Flavor != FlavorAPI || target.Operation != "generate" {
		t.Fatalf("error context = %+v", target)
	}
	if target.Status != http.StatusBadRequest || target.Code != "invalid_request_error" {
		t.Fatalf("error status/code = %d/%q", target.Status, target.Code)
	}
	if target.Message != "bad request" {
		t.Fatalf("error message = %q, want %q", target.Message, "bad request")
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func TestAnthropicGeneratePreservesTransportError(t *testing.T) {
	t.Parallel()

	client, err := New(Config{
		Provider: ProviderClaude,
		Flavor:   FlavorAPI,
		Model:    "claude-test",
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
	if !errors.As(err, &target) || target.Provider != ProviderClaude || target.Flavor != FlavorAPI {
		t.Fatalf("Generate() error = %#v, want contextual *Error", err)
	}
}

func TestAnthropicGenerateMapsRequestAndResponse(t *testing.T) {
	t.Parallel()

	const response = `{
  "id":"msg_123",
  "type":"message",
  "role":"assistant",
  "model":"claude-test",
  "content":[
    {"type":"text","text":"Hello, "},
    {"type":"text","text":"world"}
  ],
  "stop_reason":"end_turn",
  "stop_sequence":null,
  "usage":{
    "input_tokens":10,
    "output_tokens":5,
    "cache_read_input_tokens":3,
    "cache_creation_input_tokens":2
  }
}`
	server, captured := newAPITestServer(t, http.StatusOK, response)
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

	result, err := client.Generate(context.Background(), Request{
		System:    "Be concise.",
		MaxTokens: 64,
		Messages:  []Message{User("hello"), Assistant("hi"), User("continue")},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	if captured.method != http.MethodPost || captured.path != "/v1/messages" {
		t.Fatalf("request = %s %s", captured.method, captured.path)
	}
	if captured.header.Get("X-Api-Key") != "test-key" {
		t.Fatalf("X-Api-Key = %q", captured.header.Get("X-Api-Key"))
	}
	if captured.body["model"] != "claude-test" || captured.body["max_tokens"] != float64(64) {
		t.Fatalf("request body = %+v", captured.body)
	}

	messages, ok := captured.body["messages"].([]any)
	if !ok || len(messages) != 3 {
		t.Fatalf("messages = %#v", captured.body["messages"])
	}
	for i, want := range []Message{User("hello"), Assistant("hi"), User("continue")} {
		message := messages[i].(map[string]any)
		content := message["content"].([]any)[0].(map[string]any)
		if message["role"] != string(want.Role) || content["text"] != want.Text {
			t.Errorf("messages[%d] = %#v", i, message)
		}
	}
	system := captured.body["system"].([]any)[0].(map[string]any)
	if system["text"] != "Be concise." {
		t.Fatalf("system = %#v", system)
	}
	// Backward compat: no OutputSchema => no output_config in the request body.
	if _, ok := captured.body["output_config"]; ok {
		t.Fatalf("output_config present without OutputSchema: %#v", captured.body["output_config"])
	}

	if result.ID != "msg_123" || result.Model != "claude-test" || result.Text != "Hello, world" {
		t.Fatalf("response = %+v", result)
	}
	if result.CompletionReason != CompletionReasonStop {
		t.Fatalf("completion reason = %q, want %q", result.CompletionReason, CompletionReasonStop)
	}
	if result.Provider != ProviderClaude || result.Flavor != FlavorAPI {
		t.Fatalf("response target = %s/%s", result.Provider, result.Flavor)
	}
	wantUsage := Usage{InputTokens: 10, OutputTokens: 5, CachedInputTokens: 3, CacheCreationInputTokens: 2}
	if result.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", result.Usage, wantUsage)
	}
}

func TestAnthropicGenerateStructuredOutput(t *testing.T) {
	t.Parallel()

	// The model returns the JSON document as its response text.
	response := `{"id":"msg_s","type":"message","role":"assistant","model":"claude-test",` +
		`"content":[{"type":"text","text":` + strconv.Quote(structuredJSON) + `}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}`
	server, captured := newAPITestServer(t, http.StatusOK, response)
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

	result, err := client.Generate(context.Background(), Request{
		MaxTokens:    64,
		Messages:     []Message{User("q")},
		OutputSchema: &OutputSchema{Name: "answer", Schema: structuredSchema},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	// Request maps to output_config.format json_schema carrying the schema object.
	outputConfig, ok := captured.body["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("output_config = %#v", captured.body["output_config"])
	}
	format, ok := outputConfig["format"].(map[string]any)
	if !ok || format["type"] != "json_schema" {
		t.Fatalf("output_config.format = %#v", outputConfig["format"])
	}
	schema, ok := format["schema"].(map[string]any)
	if !ok || schema["type"] != "object" {
		t.Fatalf("output_config.format.schema = %#v", format["schema"])
	}

	// Text keeps the raw output; StructuredOutput carries the JSON-validated bytes.
	if result.Text != structuredJSON {
		t.Fatalf("Text = %q, want %q", result.Text, structuredJSON)
	}
	if string(result.StructuredOutput) != structuredJSON {
		t.Fatalf("StructuredOutput = %q, want %q", result.StructuredOutput, structuredJSON)
	}
}

func TestAnthropicGenerateStructuredOutputRejectsNonJSON(t *testing.T) {
	t.Parallel()

	response := `{"id":"msg_s","type":"message","role":"assistant","model":"claude-test",` +
		`"content":[{"type":"text","text":"not json"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}`
	server, _ := newAPITestServer(t, http.StatusOK, response)
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

	_, err = client.Generate(context.Background(), Request{
		MaxTokens:    64,
		Messages:     []Message{User("q")},
		OutputSchema: &OutputSchema{Name: "answer", Schema: structuredSchema},
	})
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Generate() error = %T %v, want *Error", err, err)
	}
	if target.Provider != ProviderClaude || target.Flavor != FlavorAPI || target.Operation != "generate" {
		t.Fatalf("error context = %+v", target)
	}
	if target.Code != "invalid_structured_output" {
		t.Fatalf("code = %q, want invalid_structured_output", target.Code)
	}
}
