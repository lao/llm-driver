package claudewrap

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// sdkTestServer returns an httptest server that replies with the given status
// and body to any request, capturing the last decoded request body.
func sdkTestServer(t *testing.T, status int, body string) (*httptest.Server, *map[string]any) {
	t.Helper()
	captured := new(map[string]any)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(raw, &m)
		*captured = m
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv, captured
}

const okBody = `{
  "id": "msg_1", "type": "message", "role": "assistant",
  "model": "claude-opus-4-8",
  "content": [{"type": "text", "text": "Hello, world"}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 10, "output_tokens": 5}
}`

func TestSDKMessagesNew(t *testing.T) {
	srv, captured := sdkTestServer(t, 200, okBody)

	c, err := New(WithSDK("test-key", WithSDKBaseURL(srv.URL)))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := c.Messages.New(context.Background(), MessageParams{
		MaxTokens: 1024,
		Messages:  []Message{UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	if resp.Text() != "Hello, world" {
		t.Errorf("Text() = %q", resp.Text())
	}
	if resp.Backend != "sdk" {
		t.Errorf("Backend = %q, want sdk", resp.Backend)
	}
	if resp.Role != RoleAssistant {
		t.Errorf("Role = %q", resp.Role)
	}
	if resp.StopReason != StopEndTurn {
		t.Errorf("StopReason = %q", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v", resp.Usage)
	}
	// Empty model must fall back to DefaultModel on the wire.
	if got := (*captured)["model"]; got != DefaultModel {
		t.Errorf("request model = %v, want %s", got, DefaultModel)
	}
}

func TestSDKMessagesNewError(t *testing.T) {
	srv, _ := sdkTestServer(t, 429, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)

	c, err := New(WithSDK("test-key", WithSDKBaseURL(srv.URL)))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = c.Messages.New(context.Background(), MessageParams{
		MaxTokens: 16,
		Messages:  []Message{UserMessage("hi")},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var e *Error
	if !errors.As(err, &e) {
		t.Fatalf("error is %T, want *claudewrap.Error", err)
	}
	if e.Backend != "sdk" {
		t.Errorf("Backend = %q, want sdk", e.Backend)
	}
	if e.Status != 429 {
		t.Errorf("Status = %d, want 429", e.Status)
	}
}

func TestNewNoBackend(t *testing.T) {
	_, err := New()
	if !errors.Is(err, ErrNoBackend) {
		t.Fatalf("err = %v, want ErrNoBackend", err)
	}
}

func TestNewDefaultModelOverride(t *testing.T) {
	srv, captured := sdkTestServer(t, 200, okBody)
	c, err := New(WithSDK("k", WithSDKBaseURL(srv.URL)), WithDefaultModel("claude-haiku-4-5"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = c.Messages.New(context.Background(), MessageParams{
		MaxTokens: 16,
		Messages:  []Message{UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	if got := (*captured)["model"]; got != "claude-haiku-4-5" {
		t.Errorf("request model = %v, want claude-haiku-4-5", got)
	}
}
