package core

import (
	"encoding/json"
	"errors"
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
