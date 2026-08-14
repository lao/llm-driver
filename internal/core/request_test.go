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
