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
	// Backward compat: no sampling fields => none appear in the request body.
	for _, key := range []string{"temperature", "top_p", "top_k", "stop_sequences", "metadata"} {
		if _, ok := captured.body[key]; ok {
			t.Fatalf("%s present without sampling params: %#v", key, captured.body[key])
		}
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

func TestAnthropicGenerateMapsEffort(t *testing.T) {
	t.Parallel()

	response := `{"id":"msg_e","type":"message","role":"assistant","model":"claude-test",` +
		`"content":[{"type":"text","text":"ok"}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
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
		MaxTokens: 64,
		Messages:  []Message{User("q")},
		Effort:    EffortXHigh,
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	// Effort maps to output_config.effort with the neutral value passed through.
	outputConfig, ok := captured.body["output_config"].(map[string]any)
	if !ok || outputConfig["effort"] != "xhigh" {
		t.Fatalf("output_config = %#v", captured.body["output_config"])
	}
	// tool_use stop reason maps to the new completion reason.
	if result.CompletionReason != CompletionReasonToolUse {
		t.Fatalf("completion reason = %q, want %q", result.CompletionReason, CompletionReasonToolUse)
	}
}

func TestAnthropicGenerateMapsSamplingParams(t *testing.T) {
	t.Parallel()

	response := `{"id":"msg_p","type":"message","role":"assistant","model":"claude-test",` +
		`"content":[{"type":"text","text":"ok"}],"stop_reason":"stop_sequence","usage":{"input_tokens":1,"output_tokens":1}}`
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

	temp, topP, topK := 0.5, 0.9, 40
	result, err := client.Generate(context.Background(), Request{
		MaxTokens:     64,
		Messages:      []Message{User("q")},
		Temperature:   &temp,
		TopP:          &topP,
		TopK:          &topK,
		StopSequences: []string{"STOP"},
		UserID:        "user-1",
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	if captured.body["temperature"] != 0.5 || captured.body["top_p"] != 0.9 || captured.body["top_k"] != float64(40) {
		t.Fatalf("sampling params = %+v", captured.body)
	}
	stops, ok := captured.body["stop_sequences"].([]any)
	if !ok || len(stops) != 1 || stops[0] != "STOP" {
		t.Fatalf("stop_sequences = %#v", captured.body["stop_sequences"])
	}
	metadata, ok := captured.body["metadata"].(map[string]any)
	if !ok || metadata["user_id"] != "user-1" {
		t.Fatalf("metadata = %#v", captured.body["metadata"])
	}
	// A stop-sequence ending maps to the normal stop reason.
	if result.CompletionReason != CompletionReasonStop {
		t.Fatalf("completion reason = %q, want %q", result.CompletionReason, CompletionReasonStop)
	}
}

func TestAnthropicGenerateSurfacesSamplingRejection(t *testing.T) {
	t.Parallel()

	// Newest models reject temperature; the provider 400 surfaces as the
	// normalized API error, not a client-side validation failure.
	server, _ := newAPITestServer(t, http.StatusBadRequest,
		`{"type":"error","error":{"type":"invalid_request_error","message":"temperature is not supported"}}`)
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

	temp := 0.5
	_, err = client.Generate(context.Background(), Request{
		MaxTokens:   16,
		Messages:    []Message{User("hello")},
		Temperature: &temp,
	})
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Generate() error = %T %v, want *Error", err, err)
	}
	if target.Status != http.StatusBadRequest || target.Code != "invalid_request_error" {
		t.Fatalf("error status/code = %d/%q", target.Status, target.Code)
	}
	if target.Operation != "generate" {
		t.Fatalf("error operation = %q, want %q", target.Operation, "generate")
	}
}

func TestAnthropicStreamMapsSamplingParams(t *testing.T) {
	t.Parallel()

	client, captured := newAnthropicStreamClient(t, http.StatusOK, anthropicSSEEvents())
	temp, topP, topK := 0.5, 0.9, 40
	stream, err := client.GenerateStream(context.Background(), Request{
		MaxTokens:     64,
		Messages:      []Message{User("hello")},
		Temperature:   &temp,
		TopP:          &topP,
		TopK:          &topK,
		StopSequences: []string{"STOP"},
		UserID:        "user-1",
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()
	if _, err, _ := collectStream(t, stream); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if captured.body["temperature"] != 0.5 || captured.body["top_p"] != 0.9 || captured.body["top_k"] != float64(40) {
		t.Fatalf("sampling params = %+v", captured.body)
	}
	if metadata, ok := captured.body["metadata"].(map[string]any); !ok || metadata["user_id"] != "user-1" {
		t.Fatalf("metadata = %#v", captured.body["metadata"])
	}
}

func TestAnthropicGenerateToolRoundTrip(t *testing.T) {
	t.Parallel()

	// First turn: the model returns two parallel tool_use blocks.
	response := `{"id":"msg_t","type":"message","role":"assistant","model":"claude-test",` +
		`"content":[` +
		`{"type":"tool_use","id":"call_a","name":"get_weather","input":{"city":"SF"}},` +
		`{"type":"tool_use","id":"call_b","name":"get_time","input":{"tz":"PT"}}` +
		`],"stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":3}}`
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

	weatherSchema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)
	result, err := client.Generate(context.Background(), Request{
		MaxTokens:  64,
		Messages:   []Message{User("weather and time?")},
		Tools:      []Tool{{Name: "get_weather", Description: "weather by city", InputSchema: weatherSchema}, {Name: "get_time", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: ToolChoice{Mode: ToolChoiceRequired},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	// Request maps tools and tool_choice.
	tools, ok := captured.body["tools"].([]any)
	if !ok || len(tools) != 2 {
		t.Fatalf("tools = %#v", captured.body["tools"])
	}
	tool0 := tools[0].(map[string]any)
	if tool0["name"] != "get_weather" || tool0["description"] != "weather by city" {
		t.Fatalf("tools[0] = %#v", tool0)
	}
	schema, ok := tool0["input_schema"].(map[string]any)
	if !ok || schema["type"] != "object" {
		t.Fatalf("tools[0].input_schema = %#v", tool0["input_schema"])
	}
	// The schema must survive the SDK round-trip with full fidelity: keywords
	// beyond "type" (nested "properties" and top-level "required") must not be
	// silently dropped when mapping through the typed input-schema param.
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("tools[0].input_schema.properties = %#v", schema["properties"])
	}
	if city, ok := properties["city"].(map[string]any); !ok || city["type"] != "string" {
		t.Fatalf("tools[0].input_schema.properties.city = %#v", properties["city"])
	}
	required, ok := schema["required"].([]any)
	if !ok || len(required) != 1 || required[0] != "city" {
		t.Fatalf("tools[0].input_schema.required = %#v", schema["required"])
	}
	if choice, ok := captured.body["tool_choice"].(map[string]any); !ok || choice["type"] != "any" {
		t.Fatalf("tool_choice = %#v", captured.body["tool_choice"])
	}

	// Parallel tool calls preserved in order, with tool_use completion reason.
	if result.CompletionReason != CompletionReasonToolUse {
		t.Fatalf("completion reason = %q, want %q", result.CompletionReason, CompletionReasonToolUse)
	}
	if len(result.ToolCalls) != 2 {
		t.Fatalf("tool calls = %#v", result.ToolCalls)
	}
	if result.ToolCalls[0].ID != "call_a" || result.ToolCalls[0].Name != "get_weather" || string(result.ToolCalls[0].Input) != `{"city":"SF"}` {
		t.Fatalf("tool call 0 = %#v", result.ToolCalls[0])
	}
	if result.ToolCalls[1].ID != "call_b" || result.ToolCalls[1].Name != "get_time" {
		t.Fatalf("tool call 1 = %#v", result.ToolCalls[1])
	}

	// Second turn: the caller echoes the tool call and replies with results; the
	// follow-up request must carry the assistant tool_use and user tool_result.
	follow, captured2 := newAPITestServer(t, http.StatusOK,
		`{"id":"msg_f","type":"message","role":"assistant","model":"claude-test",`+
			`"content":[{"type":"text","text":"Sunny, noon."}],"stop_reason":"end_turn","usage":{"input_tokens":9,"output_tokens":4}}`)
	client2, err := New(Config{Provider: ProviderClaude, Flavor: FlavorAPI, Model: "claude-test", APIKey: "test-key", BaseURL: follow.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	final, err := client2.Generate(context.Background(), Request{
		MaxTokens: 64,
		Messages: []Message{
			User("weather and time?"),
			{Role: RoleAssistant, Parts: []Part{
				ToolCallPart{ID: "call_a", Name: "get_weather", Input: result.ToolCalls[0].Input},
				ToolCallPart{ID: "call_b", Name: "get_time", Input: result.ToolCalls[1].Input},
			}},
			ToolResult("call_a", "Sunny", false),
			ToolResult("call_b", "noon", false),
		},
	})
	if err != nil {
		t.Fatalf("follow-up Generate(): %v", err)
	}
	if final.Text != "Sunny, noon." {
		t.Fatalf("final text = %q", final.Text)
	}

	messages := captured2.body["messages"].([]any)
	if len(messages) != 4 {
		t.Fatalf("follow-up messages = %#v", messages)
	}
	assistant := messages[1].(map[string]any)
	blocks := assistant["content"].([]any)
	if assistant["role"] != "assistant" || len(blocks) != 2 {
		t.Fatalf("assistant echo = %#v", assistant)
	}
	toolUse := blocks[0].(map[string]any)
	if toolUse["type"] != "tool_use" || toolUse["id"] != "call_a" || toolUse["name"] != "get_weather" {
		t.Fatalf("assistant tool_use = %#v", toolUse)
	}
	toolResult := messages[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if toolResult["type"] != "tool_result" || toolResult["tool_use_id"] != "call_a" {
		t.Fatalf("user tool_result = %#v", toolResult)
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
