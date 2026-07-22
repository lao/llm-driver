package claudewrap

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// I1 — validation happens before the backend is touched.
func TestValidateMaxTokens(t *testing.T) {
	f := &fakeRun{stdout: []byte(okCLIBody)}
	c, _ := New(WithCLI(withCLIRunner(f.run)))
	_, err := c.Messages.New(context.Background(), MessageParams{
		Messages: []Message{UserMessage("hi")}, // MaxTokens omitted
	})
	var e *Error
	if !errors.As(err, &e) {
		t.Fatalf("err = %T, want *Error", err)
	}
	if !strings.Contains(e.Error(), "MaxTokens") {
		t.Errorf("error = %q, want mention of MaxTokens", e.Error())
	}
	if f.gotName != "" {
		t.Error("backend was invoked despite invalid params")
	}
}

func TestValidateNoMessages(t *testing.T) {
	f := &fakeRun{stdout: []byte(okCLIBody)}
	c, _ := New(WithCLI(withCLIRunner(f.run)))
	_, err := c.Messages.New(context.Background(), MessageParams{MaxTokens: 16})
	var e *Error
	if !errors.As(err, &e) {
		t.Fatalf("err = %T, want *Error", err)
	}
	if f.gotName != "" {
		t.Error("backend was invoked despite empty messages")
	}
}

// S1 — a non-API cause must not be printed twice.
func TestSDKErrorNoDuplicateMessage(t *testing.T) {
	err := sdkError("messages.new", errors.New("boom")).(*Error)
	if got := strings.Count(err.Error(), "boom"); got != 1 {
		t.Errorf("error = %q, want %q once, got %d", err.Error(), "boom", got)
	}
}

// S2 — a non-success subtype maps to an error even if is_error is unset.
func TestCLISubtypeError(t *testing.T) {
	body := `{"type":"result","subtype":"error_max_turns","is_error":false,"result":"","stop_reason":null,"session_id":"s","usage":{}}`
	f := &fakeRun{stdout: []byte(body)}
	c, _ := New(WithCLI(withCLIRunner(f.run)))
	_, err := c.Messages.New(context.Background(), MessageParams{MaxTokens: 16, Messages: []Message{UserMessage("hi")}})
	var e *Error
	if !errors.As(err, &e) {
		t.Fatalf("err = %T, want *Error", err)
	}
	if e.Type != "error_max_turns" {
		t.Errorf("Type = %q, want error_max_turns", e.Type)
	}
}

// S3 — flattenPrompt labels roles in multi-turn transcripts.
func TestFlattenPromptMultiTurn(t *testing.T) {
	got := flattenPrompt([]Message{
		UserMessage("first"),
		AssistantMessage("reply"),
		UserMessage("second"),
	})
	want := "User: first\n\nAssistant: reply\n\nUser: second"
	if got != want {
		t.Errorf("flattenPrompt = %q, want %q", got, want)
	}
}

func TestFlattenPromptSingleUser(t *testing.T) {
	if got := flattenPrompt([]Message{UserMessage("hi")}); got != "hi" {
		t.Errorf("flattenPrompt = %q, want hi", got)
	}
}

// S3 — SDK maps temperature and stop sequences onto the wire request.
func TestSDKParamMapping(t *testing.T) {
	srv, captured := sdkTestServer(t, 200, okBody)
	c, _ := New(WithSDK("k", WithSDKBaseURL(srv.URL)))
	temp := 0.5
	_, err := c.Messages.New(context.Background(), MessageParams{
		MaxTokens:   16,
		Temperature: &temp,
		StopSeqs:    []string{"STOP"},
		Messages:    []Message{UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	if got := (*captured)["temperature"]; got != 0.5 {
		t.Errorf("temperature = %v, want 0.5", got)
	}
	stops, _ := (*captured)["stop_sequences"].([]any)
	if len(stops) != 1 || stops[0] != "STOP" {
		t.Errorf("stop_sequences = %v, want [STOP]", (*captured)["stop_sequences"])
	}
}

// S5 — Backend is a typed constant.
func TestBackendConst(t *testing.T) {
	if BackendSDK != "sdk" || BackendCLI != "cli" {
		t.Fatal("backend constants drifted")
	}
	f := &fakeRun{stdout: []byte(okCLIBody)}
	c, _ := New(WithCLI(withCLIRunner(f.run)))
	if c.Backend() != BackendCLI {
		t.Errorf("Backend() = %q, want %q", c.Backend(), BackendCLI)
	}
}
