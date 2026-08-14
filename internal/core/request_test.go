package core

import (
	"encoding/json"
	"errors"
	"math"
	"slices"
	"testing"
)

func TestMessageHelpers(t *testing.T) {
	t.Parallel()

	user := User("hello")
	if user.Role != RoleUser || user.Text != "hello" {
		t.Fatalf("User() = %+v", user)
	}

	assistant := Assistant("hi")
	if assistant.Role != RoleAssistant || assistant.Text != "hi" {
		t.Fatalf("Assistant() = %+v", assistant)
	}
}

func TestRequestValidateAcceptsMultiTurnText(t *testing.T) {
	t.Parallel()

	request := Request{
		System:    "be concise",
		MaxTokens: 128,
		Messages:  []Message{User("hello"), Assistant("hi"), User("continue")},
	}
	if err := request.validate(); err != nil {
		t.Fatalf("validate(): %v", err)
	}
}

func TestRequestValidateRejectsInvalidValues(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request Request
	}{
		{name: "max tokens", request: Request{Messages: []Message{User("hello")}}},
		{name: "messages", request: Request{MaxTokens: 1}},
		{name: "role", request: Request{MaxTokens: 1, Messages: []Message{{Role: "system", Text: "hello"}}}},
		{name: "text", request: Request{MaxTokens: 1, Messages: []Message{{Role: RoleUser, Text: "  "}}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			err := test.request.validate()
			if !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("validate() error = %v, want ErrInvalidRequest", err)
			}
		})
	}
}

func TestRequestValidateRejectsNonObjectSchema(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		schema json.RawMessage
	}{
		{name: "malformed json", schema: json.RawMessage(`{not json`)},
		{name: "json array", schema: json.RawMessage(`[1,2,3]`)},
		{name: "json string", schema: json.RawMessage(`"a string"`)},
		{name: "json null", schema: json.RawMessage(`null`)},
		{name: "empty", schema: nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			request := Request{
				MaxTokens:    16,
				Messages:     []Message{User("hello")},
				OutputSchema: &OutputSchema{Schema: test.schema},
			}
			if err := request.validate(); !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("validate() error = %v, want ErrInvalidRequest", err)
			}
		})
	}
}

func TestRequestValidateAcceptsObjectSchema(t *testing.T) {
	t.Parallel()

	request := Request{
		MaxTokens:    16,
		Messages:     []Message{User("hello")},
		OutputSchema: &OutputSchema{Name: "answer", Schema: json.RawMessage(`{"type":"object"}`)},
	}
	if err := request.validate(); err != nil {
		t.Fatalf("validate(): %v", err)
	}
}

func TestFeaturesOfDetectsStructuredOutput(t *testing.T) {
	t.Parallel()

	if got := featuresOf(Request{MaxTokens: 16, Messages: []Message{User("hi")}}); len(got) != 0 {
		t.Fatalf("featuresOf(bare request) = %v, want none", got)
	}
	request := Request{
		MaxTokens:    16,
		Messages:     []Message{User("hi")},
		OutputSchema: &OutputSchema{Schema: json.RawMessage(`{"type":"object"}`)},
	}
	if got := featuresOf(request); !slices.Equal(got, []Feature{FeatureStructuredOutput}) {
		t.Fatalf("featuresOf(schema request) = %v, want [structured output]", got)
	}
}

func TestToolResultHelper(t *testing.T) {
	t.Parallel()

	message := ToolResult("call_1", "72F", true)
	if message.Role != RoleUser || message.Text != "" || len(message.Parts) != 1 {
		t.Fatalf("ToolResult() = %+v", message)
	}
	part, ok := message.Parts[0].(ToolResultPart)
	if !ok || part.ToolCallID != "call_1" || part.Content != "72F" || !part.IsError {
		t.Fatalf("ToolResult() part = %#v", message.Parts[0])
	}
}

func TestRequestValidateAcceptsToolParts(t *testing.T) {
	t.Parallel()

	// A full tool round-trip transcript: user turn, assistant tool-call echo, and
	// a user tool-result reply — all valid.
	request := Request{
		MaxTokens: 16,
		Messages: []Message{
			User("weather?"),
			{Role: RoleAssistant, Parts: []Part{ToolCallPart{ID: "c1", Name: "get_weather", Input: json.RawMessage(`{"city":"SF"}`)}}},
			ToolResult("c1", "72F", false),
		},
		Tools: []Tool{{Name: "get_weather", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	}
	if err := request.validate(); err != nil {
		t.Fatalf("validate(): %v", err)
	}
}

func TestRequestValidateRejectsBadMessages(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		message Message
	}{
		{name: "both text and parts", message: Message{Role: RoleUser, Text: "hi", Parts: []Part{TextPart{Text: "hi"}}}},
		{name: "blank text part", message: Message{Role: RoleUser, Parts: []Part{TextPart{Text: "  "}}}},
		{name: "tool call missing name", message: Message{Role: RoleAssistant, Parts: []Part{ToolCallPart{ID: "c1", Input: json.RawMessage(`{}`)}}}},
		{name: "tool call bad json", message: Message{Role: RoleAssistant, Parts: []Part{ToolCallPart{ID: "c1", Name: "t", Input: json.RawMessage(`{bad`)}}}},
		{name: "tool result missing id", message: Message{Role: RoleUser, Parts: []Part{ToolResultPart{Content: "x"}}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			request := Request{MaxTokens: 16, Messages: []Message{test.message}}
			if err := request.validate(); !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("validate() error = %v, want ErrInvalidRequest", err)
			}
		})
	}
}

func TestRequestValidateRejectsBadTools(t *testing.T) {
	t.Parallel()

	object := json.RawMessage(`{"type":"object"}`)
	tests := []struct {
		name    string
		request Request
	}{
		{name: "empty name", request: Request{Tools: []Tool{{InputSchema: object}}}},
		{name: "duplicate names", request: Request{Tools: []Tool{{Name: "t", InputSchema: object}, {Name: "t", InputSchema: object}}}},
		{name: "non-object schema", request: Request{Tools: []Tool{{Name: "t", InputSchema: json.RawMessage(`[1]`)}}}},
		{name: "missing schema", request: Request{Tools: []Tool{{Name: "t"}}}},
		{name: "named choice absent", request: Request{ToolChoice: ToolChoice{Mode: ToolChoiceTool, Name: "missing"}, Tools: []Tool{{Name: "t", InputSchema: object}}}},
		{name: "unknown choice mode", request: Request{ToolChoice: ToolChoice{Mode: "bogus"}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			request := test.request
			request.MaxTokens = 16
			request.Messages = []Message{User("hi")}
			if err := request.validate(); !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("validate() error = %v, want ErrInvalidRequest", err)
			}
		})
	}
}

func TestFeaturesOfDetectsTools(t *testing.T) {
	t.Parallel()

	object := json.RawMessage(`{"type":"object"}`)
	tests := []struct {
		name    string
		request Request
	}{
		{name: "tools", request: Request{Tools: []Tool{{Name: "t", InputSchema: object}}}},
		{name: "tool choice", request: Request{ToolChoice: ToolChoice{Mode: ToolChoiceRequired}}},
		{name: "tool call part", request: Request{Messages: []Message{{Role: RoleAssistant, Parts: []Part{ToolCallPart{ID: "c", Name: "t", Input: object}}}}}},
		{name: "tool result part", request: Request{Messages: []Message{ToolResult("c", "x", false)}}},
		// Any Parts usage requires tool support, even a plain text part: a target
		// that cannot render Parts must reject rather than silently drop content.
		{name: "text part", request: Request{Messages: []Message{{Role: RoleUser, Parts: []Part{TextPart{Text: "hi"}}}}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			if got := featuresOf(test.request); !slices.Equal(got, []Feature{FeatureTools}) {
				t.Fatalf("featuresOf() = %v, want [tool calling]", got)
			}
		})
	}
}

func TestFeaturesOfDetectsSamplingParams(t *testing.T) {
	t.Parallel()

	temp := 0.7
	topK := 40
	tests := []struct {
		name    string
		request Request
		want    []Feature
	}{
		{name: "temperature", request: Request{Temperature: &temp}, want: []Feature{FeatureSamplingParams}},
		{name: "top-p", request: Request{TopP: &temp}, want: []Feature{FeatureSamplingParams}},
		{name: "top-k", request: Request{TopK: &topK}, want: []Feature{FeatureTopK}},
		{name: "stop sequences", request: Request{StopSequences: []string{"STOP"}}, want: []Feature{FeatureStopSequences}},
		{name: "user id", request: Request{UserID: "u1"}, want: []Feature{FeatureUserID}},
		{
			name:    "all together in inventory order",
			request: Request{Temperature: &temp, TopK: &topK, StopSequences: []string{"x"}, UserID: "u1"},
			want:    []Feature{FeatureSamplingParams, FeatureTopK, FeatureStopSequences, FeatureUserID},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			if got := featuresOf(test.request); !slices.Equal(got, test.want) {
				t.Fatalf("featuresOf() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestRequestValidateRejectsInvalidSamplingParams(t *testing.T) {
	t.Parallel()

	neg := -0.1
	nan := math.NaN()
	inf := math.Inf(1)
	negK := -1
	tests := []struct {
		name    string
		request Request
	}{
		{name: "negative temperature", request: Request{Temperature: &neg}},
		{name: "nan temperature", request: Request{Temperature: &nan}},
		{name: "inf top-p", request: Request{TopP: &inf}},
		{name: "negative top-k", request: Request{TopK: &negK}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			request := test.request
			request.MaxTokens = 16
			request.Messages = []Message{User("hi")}
			if err := request.validate(); !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("validate() error = %v, want ErrInvalidRequest", err)
			}
		})
	}
}

func TestRequestValidateAcceptsSamplingParams(t *testing.T) {
	t.Parallel()

	// Provider-specific upper bounds are not enforced: a large temperature passes
	// core validation and is left for the provider to accept or reject.
	temp := 2.0
	topP := 1.0
	topK := 500
	request := Request{
		MaxTokens:     16,
		Messages:      []Message{User("hi")},
		Temperature:   &temp,
		TopP:          &topP,
		TopK:          &topK,
		StopSequences: []string{"STOP"},
		UserID:        "user-123",
	}
	if err := request.validate(); err != nil {
		t.Fatalf("validate(): %v", err)
	}
}
