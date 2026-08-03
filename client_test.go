package llmdriver

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type fakeGenerationBackend struct {
	request  Request
	response *Response
	err      error
	calls    int
}

func (f *fakeGenerationBackend) Generate(_ context.Context, request Request) (*Response, error) {
	f.calls++
	f.request = request
	return f.response, f.err
}

func TestClientNewAcceptsFourTargets(t *testing.T) {
	t.Parallel()

	for _, provider := range []Provider{ProviderClaude, ProviderOpenAI} {
		for _, flavor := range []Flavor{FlavorAPI, FlavorCLI} {
			client, err := New(Config{Provider: provider, Flavor: flavor, Model: "model"})
			if err != nil {
				t.Fatalf("New(%s/%s): %v", provider, flavor, err)
			}
			if client == nil {
				t.Fatalf("New(%s/%s) returned nil client", provider, flavor)
			}
		}
	}
}

func TestClientNewRejectsInvalidConfig(t *testing.T) {
	t.Parallel()

	_, err := New(Config{Provider: ProviderClaude, Flavor: FlavorAPI})
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("New() error = %v, want ErrInvalidConfig", err)
	}
	var target *Error
	if !errors.As(err, &target) || target.Operation != "client.new" {
		t.Fatalf("New() error = %#v, want contextual *Error", err)
	}
}

func TestClientGenerateDelegatesAndStampsTarget(t *testing.T) {
	t.Parallel()

	request := Request{MaxTokens: 32, Messages: []Message{User("hello")}}
	fake := &fakeGenerationBackend{response: &Response{ID: "response-1", Model: "model", Text: "hi"}}
	client := &Client{
		provider: ProviderOpenAI,
		flavor:   FlavorCLI,
		model:    "model",
		backend:  fake,
	}

	response, err := client.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if fake.calls != 1 || !reflect.DeepEqual(fake.request, request) {
		t.Fatalf("backend calls = %d, request = %+v", fake.calls, fake.request)
	}
	if response.Provider != ProviderOpenAI || response.Flavor != FlavorCLI {
		t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
	}
}

func TestClientGenerateValidatesBeforeDelegating(t *testing.T) {
	t.Parallel()

	fake := &fakeGenerationBackend{}
	client := &Client{
		provider: ProviderClaude,
		flavor:   FlavorAPI,
		model:    "model",
		backend:  fake,
	}

	_, err := client.Generate(context.Background(), Request{})
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("Generate() error = %v, want ErrInvalidRequest", err)
	}
	if fake.calls != 0 {
		t.Fatalf("backend called %d times for invalid request", fake.calls)
	}
}
