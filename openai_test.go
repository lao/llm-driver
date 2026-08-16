package llmdriver

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
)

func TestOpenAIGenerateNormalizesAPIError(t *testing.T) {
	t.Parallel()

	server, _ := newAPITestServer(t, http.StatusBadRequest,
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
	if target.Message != "bad request" {
		t.Fatalf("error message = %q, want %q", target.Message, "bad request")
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
    "input_tokens_details":{"cached_tokens":3,"cache_write_tokens":4},
    "output_tokens":7,
    "output_tokens_details":{"reasoning_tokens":2},
    "total_tokens":17
  }
}`
	server, captured := newAPITestServer(t, http.StatusOK, response)
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
	// Backward compat: no OutputSchema => no text.format in the request body.
	if _, ok := captured.body["text"]; ok {
		t.Fatalf("text present without OutputSchema: %#v", captured.body["text"])
	}
	// Backward compat: no sampling fields => none appear in the request body.
	for _, key := range []string{"temperature", "top_p", "safety_identifier"} {
		if _, ok := captured.body[key]; ok {
			t.Fatalf("%s present without sampling params: %#v", key, captured.body[key])
		}
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
		wantPhase := ""
		if want.Role == RoleAssistant {
			wantPhase = "final_answer"
		}
		if phase, _ := message["phase"].(string); phase != wantPhase {
			t.Errorf("input[%d] phase = %q, want %q", i, phase, wantPhase)
		}
	}

	if result.ID != "resp_123" || result.Model != "gpt-test" || result.Text != "Hello, world" {
		t.Fatalf("response = %+v", result)
	}
	if result.CompletionReason != CompletionReasonStop {
		t.Fatalf("completion reason = %q, want %q", result.CompletionReason, CompletionReasonStop)
	}
	if result.Provider != ProviderOpenAI || result.Flavor != FlavorAPI {
		t.Fatalf("response target = %s/%s", result.Provider, result.Flavor)
	}
	wantUsage := Usage{InputTokens: 10, OutputTokens: 7, CachedInputTokens: 3, CacheCreationInputTokens: 4, ReasoningTokens: 2}
	if result.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", result.Usage, wantUsage)
	}
}

func TestOpenAIGeneratePreservesMaxOutputTokensResult(t *testing.T) {
	t.Parallel()

	const response = `{
  "id":"resp_incomplete","object":"response","status":"incomplete","model":"gpt-test",
  "output":[{
    "id":"msg_incomplete","type":"message","status":"incomplete","role":"assistant",
    "content":[{"type":"output_text","text":"partial answer","annotations":[],"logprobs":[]}]
  }],
  "incomplete_details":{"reason":"max_output_tokens"},
  "usage":{
    "input_tokens":10,
    "input_tokens_details":{"cached_tokens":3},
    "output_tokens":7,
    "output_tokens_details":{"reasoning_tokens":2},
    "total_tokens":17
  }
}`
	client, err := New(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorAPI,
		Model:    "gpt-test",
		APIKey:   "test-key",
		BaseURL:  "https://example.test",
		HTTPClient: &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(response)),
				Request:    request,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	result, err := client.Generate(context.Background(), Request{MaxTokens: 16, Messages: []Message{User("hello")}})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if result.Text != "partial answer" {
		t.Fatalf("response text = %q, want %q", result.Text, "partial answer")
	}
	if result.CompletionReason != CompletionReasonMaxTokens {
		t.Fatalf("completion reason = %q, want %q", result.CompletionReason, CompletionReasonMaxTokens)
	}
	wantUsage := Usage{InputTokens: 10, OutputTokens: 7, CachedInputTokens: 3, ReasoningTokens: 2}
	if result.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", result.Usage, wantUsage)
	}
}

func TestOpenAIGenerateMapsEffort(t *testing.T) {
	t.Parallel()

	const response = `{
  "id":"resp_e","object":"response","status":"completed","model":"gpt-test",
  "output":[{"id":"m","type":"message","status":"completed","role":"assistant",
    "content":[{"type":"output_text","text":"ok","annotations":[],"logprobs":[]}]}],
  "usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
}`
	server, captured := newAPITestServer(t, http.StatusOK, response)
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

	if _, err := client.Generate(context.Background(), Request{
		MaxTokens: 64,
		Messages:  []Message{User("q")},
		Effort:    EffortHigh,
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	// Effort maps to reasoning.effort with the neutral value passed through.
	reasoning, ok := captured.body["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "high" {
		t.Fatalf("reasoning = %#v", captured.body["reasoning"])
	}
}

func TestOpenAIGenerateMapsSamplingParams(t *testing.T) {
	t.Parallel()

	const response = `{
  "id":"resp_p","object":"response","status":"completed","model":"gpt-test",
  "output":[{"id":"m","type":"message","status":"completed","role":"assistant",
    "content":[{"type":"output_text","text":"ok","annotations":[],"logprobs":[]}]}],
  "usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
}`
	server, captured := newAPITestServer(t, http.StatusOK, response)
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

	temp, topP := 0.5, 0.9
	if _, err := client.Generate(context.Background(), Request{
		MaxTokens:   64,
		Messages:    []Message{User("q")},
		Temperature: &temp,
		TopP:        &topP,
		UserID:      "user-1",
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	// Temperature/TopP pass through; UserID maps to safety_identifier.
	if captured.body["temperature"] != 0.5 || captured.body["top_p"] != 0.9 {
		t.Fatalf("sampling params = %+v", captured.body)
	}
	if captured.body["safety_identifier"] != "user-1" {
		t.Fatalf("safety_identifier = %#v", captured.body["safety_identifier"])
	}
}

func TestOpenAIStreamMapsSamplingParams(t *testing.T) {
	t.Parallel()

	client, captured := newOpenAIStreamClient(t, http.StatusOK, openAISSEEvents())
	temp, topP := 0.5, 0.9
	stream, err := client.GenerateStream(context.Background(), Request{
		MaxTokens:   64,
		Messages:    []Message{User("hello")},
		Temperature: &temp,
		TopP:        &topP,
		UserID:      "user-1",
	})
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()
	if _, err, _ := collectStream(t, stream); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if captured.body["temperature"] != 0.5 || captured.body["top_p"] != 0.9 || captured.body["safety_identifier"] != "user-1" {
		t.Fatalf("sampling params = %+v", captured.body)
	}
}

func TestOpenAIGeneratePreservesContentFilterResult(t *testing.T) {
	t.Parallel()

	const response = `{
  "id":"resp_cf","object":"response","status":"incomplete","model":"gpt-test",
  "output":[{"id":"m","type":"message","status":"incomplete","role":"assistant",
    "content":[{"type":"output_text","text":"partial","annotations":[],"logprobs":[]}]}],
  "incomplete_details":{"reason":"content_filter"},
  "usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
}`
	server, _ := newAPITestServer(t, http.StatusOK, response)
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

	result, err := client.Generate(context.Background(), Request{MaxTokens: 16, Messages: []Message{User("hello")}})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if result.Text != "partial" {
		t.Fatalf("response text = %q, want %q", result.Text, "partial")
	}
	if result.CompletionReason != CompletionReasonContentFilter {
		t.Fatalf("completion reason = %q, want %q", result.CompletionReason, CompletionReasonContentFilter)
	}
}

func TestOpenAIGeneratePreservesRefusal(t *testing.T) {
	t.Parallel()

	const response = `{
  "id":"resp_refusal","object":"response","status":"completed","model":"gpt-test",
  "output":[{
    "id":"msg_refusal","type":"message","status":"completed","role":"assistant",
    "content":[{"type":"refusal","refusal":"I cannot help with that."}]
  }],
  "usage":{"input_tokens":4,"output_tokens":6,"total_tokens":10}
}`
	client, err := New(Config{
		Provider: ProviderOpenAI,
		Flavor:   FlavorAPI,
		Model:    "gpt-test",
		APIKey:   "test-key",
		BaseURL:  "https://example.test",
		HTTPClient: &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(response)),
				Request:    request,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	result, err := client.Generate(context.Background(), Request{MaxTokens: 16, Messages: []Message{User("hello")}})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if result.Text != "I cannot help with that." {
		t.Fatalf("response text = %q, want refusal text", result.Text)
	}
	if result.CompletionReason != CompletionReasonRefusal {
		t.Fatalf("completion reason = %q, want %q", result.CompletionReason, CompletionReasonRefusal)
	}
	if result.Usage != (Usage{InputTokens: 4, OutputTokens: 6}) {
		t.Fatalf("usage = %+v", result.Usage)
	}
}

func TestOpenAIGenerateToolRoundTrip(t *testing.T) {
	t.Parallel()

	// First turn: two parallel function_call output items.
	response := `{"id":"resp_t","object":"response","status":"completed","model":"gpt-test",` +
		`"output":[` +
		`{"id":"fc_a","type":"function_call","status":"completed","call_id":"call_a","name":"get_weather","arguments":"{\"city\":\"SF\"}"},` +
		`{"id":"fc_b","type":"function_call","status":"completed","call_id":"call_b","name":"get_time","arguments":"{\"tz\":\"PT\"}"}` +
		`],"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}`
	server, captured := newAPITestServer(t, http.StatusOK, response)
	client, err := New(Config{Provider: ProviderOpenAI, Flavor: FlavorAPI, Model: "gpt-test", APIKey: "test-key", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	weatherSchema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)
	result, err := client.Generate(context.Background(), Request{
		MaxTokens:  64,
		Messages:   []Message{User("weather and time?")},
		Tools:      []Tool{{Name: "get_weather", Description: "weather by city", InputSchema: weatherSchema}, {Name: "get_time", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: ToolChoice{Mode: ToolChoiceRequired},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	// Request maps function tools and tool_choice.
	tools, ok := captured.body["tools"].([]any)
	if !ok || len(tools) != 2 {
		t.Fatalf("tools = %#v", captured.body["tools"])
	}
	tool0 := tools[0].(map[string]any)
	if tool0["type"] != "function" || tool0["name"] != "get_weather" || tool0["description"] != "weather by city" {
		t.Fatalf("tools[0] = %#v", tool0)
	}
	if params, ok := tool0["parameters"].(map[string]any); !ok || params["type"] != "object" {
		t.Fatalf("tools[0].parameters = %#v", tool0["parameters"])
	}
	if captured.body["tool_choice"] != "required" {
		t.Fatalf("tool_choice = %#v", captured.body["tool_choice"])
	}

	// Parallel calls preserved in order, with tool_use completion reason.
	if result.CompletionReason != CompletionReasonToolUse {
		t.Fatalf("completion reason = %q, want %q", result.CompletionReason, CompletionReasonToolUse)
	}
	if len(result.ToolCalls) != 2 {
		t.Fatalf("tool calls = %#v", result.ToolCalls)
	}
	if result.ToolCalls[0].ID != "call_a" || result.ToolCalls[0].Name != "get_weather" || string(result.ToolCalls[0].Input) != `{"city":"SF"}` {
		t.Fatalf("tool call 0 = %#v", result.ToolCalls[0])
	}
	if result.ToolCalls[1].ID != "call_b" {
		t.Fatalf("tool call 1 = %#v", result.ToolCalls[1])
	}

	// Second turn: echo the function calls and reply with results.
	follow, captured2 := newAPITestServer(t, http.StatusOK,
		`{"id":"resp_f","object":"response","status":"completed","model":"gpt-test",`+
			`"output":[{"id":"m","type":"message","status":"completed","role":"assistant",`+
			`"content":[{"type":"output_text","text":"Sunny, noon.","annotations":[],"logprobs":[]}]}],`+
			`"usage":{"input_tokens":9,"output_tokens":4,"total_tokens":13}}`)
	client2, err := New(Config{Provider: ProviderOpenAI, Flavor: FlavorAPI, Model: "gpt-test", APIKey: "test-key", BaseURL: follow.URL})
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

	// Follow-up input carries function_call items and function_call_output items.
	input := captured2.body["input"].([]any)
	if len(input) != 5 {
		t.Fatalf("follow-up input = %#v", input)
	}
	call := input[1].(map[string]any)
	if call["type"] != "function_call" || call["call_id"] != "call_a" || call["name"] != "get_weather" || call["arguments"] != `{"city":"SF"}` {
		t.Fatalf("function_call item = %#v", call)
	}
	output := input[3].(map[string]any)
	if output["type"] != "function_call_output" || output["call_id"] != "call_a" || output["output"] != "Sunny" {
		t.Fatalf("function_call_output item = %#v", output)
	}
}

func TestOpenAIGenerateToolChoiceNamed(t *testing.T) {
	t.Parallel()

	response := `{"id":"resp_c","object":"response","status":"completed","model":"gpt-test",` +
		`"output":[{"id":"m","type":"message","status":"completed","role":"assistant",` +
		`"content":[{"type":"output_text","text":"ok","annotations":[],"logprobs":[]}]}],` +
		`"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	server, captured := newAPITestServer(t, http.StatusOK, response)
	client, err := New(Config{Provider: ProviderOpenAI, Flavor: FlavorAPI, Model: "gpt-test", APIKey: "test-key", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := client.Generate(context.Background(), Request{
		MaxTokens:  64,
		Messages:   []Message{User("weather?")},
		Tools:      []Tool{{Name: "get_weather", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: ToolChoice{Mode: ToolChoiceTool, Name: "get_weather"},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	choice, ok := captured.body["tool_choice"].(map[string]any)
	if !ok || choice["type"] != "function" || choice["name"] != "get_weather" {
		t.Fatalf("tool_choice = %#v", captured.body["tool_choice"])
	}
}

func TestOpenAIGenerateToolChoiceNone(t *testing.T) {
	t.Parallel()

	response := `{"id":"resp_n","object":"response","status":"completed","model":"gpt-test",` +
		`"output":[{"id":"m","type":"message","status":"completed","role":"assistant",` +
		`"content":[{"type":"output_text","text":"ok","annotations":[],"logprobs":[]}]}],` +
		`"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	server, captured := newAPITestServer(t, http.StatusOK, response)
	client, err := New(Config{Provider: ProviderOpenAI, Flavor: FlavorAPI, Model: "gpt-test", APIKey: "test-key", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := client.Generate(context.Background(), Request{
		MaxTokens:  64,
		Messages:   []Message{User("weather?")},
		Tools:      []Tool{{Name: "get_weather", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: ToolChoice{Mode: ToolChoiceNone},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	if captured.body["tool_choice"] != "none" {
		t.Fatalf("tool_choice = %#v", captured.body["tool_choice"])
	}
}

func TestOpenAIGenerateStructuredOutput(t *testing.T) {
	t.Parallel()

	response := `{"id":"resp_s","object":"response","status":"completed","model":"gpt-test",` +
		`"output":[{"id":"message-1","type":"message","status":"completed","role":"assistant",` +
		`"content":[{"type":"output_text","text":` + strconv.Quote(structuredJSON) + `,"annotations":[],"logprobs":[]}]}],` +
		`"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}`
	server, captured := newAPITestServer(t, http.StatusOK, response)
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
		MaxTokens:    64,
		Messages:     []Message{User("q")},
		OutputSchema: &OutputSchema{Name: "answer", Schema: structuredSchema},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	// Request maps to text.format json_schema with strict:true.
	text, ok := captured.body["text"].(map[string]any)
	if !ok {
		t.Fatalf("text = %#v", captured.body["text"])
	}
	format, ok := text["format"].(map[string]any)
	if !ok || format["type"] != "json_schema" {
		t.Fatalf("text.format = %#v", text["format"])
	}
	if format["name"] != "answer" || format["strict"] != true {
		t.Fatalf("text.format name/strict = %#v", format)
	}
	if schema, ok := format["schema"].(map[string]any); !ok || schema["type"] != "object" {
		t.Fatalf("text.format.schema = %#v", format["schema"])
	}

	if result.Text != structuredJSON {
		t.Fatalf("Text = %q, want %q", result.Text, structuredJSON)
	}
	if string(result.StructuredOutput) != structuredJSON {
		t.Fatalf("StructuredOutput = %q, want %q", result.StructuredOutput, structuredJSON)
	}
}

func TestOpenAIGenerateStructuredOutputDefaultsName(t *testing.T) {
	t.Parallel()

	response := `{"id":"resp_s","object":"response","status":"completed","model":"gpt-test",` +
		`"output":[{"id":"message-1","type":"message","status":"completed","role":"assistant",` +
		`"content":[{"type":"output_text","text":` + strconv.Quote(structuredJSON) + `,"annotations":[],"logprobs":[]}]}],` +
		`"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}`
	server, captured := newAPITestServer(t, http.StatusOK, response)
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

	// No name supplied: the openai mapping requires one, so it is defaulted.
	if _, err := client.Generate(context.Background(), Request{
		MaxTokens:    64,
		Messages:     []Message{User("q")},
		OutputSchema: &OutputSchema{Schema: structuredSchema},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	format := captured.body["text"].(map[string]any)["format"].(map[string]any)
	if name, _ := format["name"].(string); name == "" {
		t.Fatalf("text.format.name = %q, want a defaulted non-empty name", name)
	}
}

func TestOpenAIGenerateStructuredOutputRejectsNonJSON(t *testing.T) {
	t.Parallel()

	response := `{"id":"resp_s","object":"response","status":"completed","model":"gpt-test",` +
		`"output":[{"id":"message-1","type":"message","status":"completed","role":"assistant",` +
		`"content":[{"type":"output_text","text":"not json","annotations":[],"logprobs":[]}]}],` +
		`"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}`
	server, _ := newAPITestServer(t, http.StatusOK, response)
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

	_, err = client.Generate(context.Background(), Request{
		MaxTokens:    64,
		Messages:     []Message{User("q")},
		OutputSchema: &OutputSchema{Name: "answer", Schema: structuredSchema},
	})
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Generate() error = %T %v, want *Error", err, err)
	}
	if target.Operation != "generate" || target.Code != "invalid_structured_output" {
		t.Fatalf("error = %+v, want generate/invalid_structured_output", target)
	}
}

func TestOpenAIGenerateRejectsFailedTerminalResult(t *testing.T) {
	t.Parallel()

	const response = `{
  "id":"resp_failed","object":"response","status":"failed","model":"gpt-test","output":[],
  "error":{"code":"server_error","message":"generation failed"}
}`
	server, _ := newAPITestServer(t, http.StatusOK, response)
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
	if target.Code != "server_error" {
		t.Fatalf("error code = %q, want %q", target.Code, "server_error")
	}
}
