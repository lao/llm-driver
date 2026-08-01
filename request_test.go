package llmwrapper

import (
	"errors"
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
