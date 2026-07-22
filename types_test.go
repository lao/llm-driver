package claudewrap

import (
	"errors"
	"strings"
	"testing"
)

func TestUserMessage(t *testing.T) {
	m := UserMessage("hi")
	if m.Role != RoleUser {
		t.Fatalf("role = %q, want %q", m.Role, RoleUser)
	}
	if len(m.Content) != 1 || m.Content[0].Type != BlockText || m.Content[0].Text != "hi" {
		t.Fatalf("content = %+v, want one text block %q", m.Content, "hi")
	}
}

func TestAssistantMessage(t *testing.T) {
	m := AssistantMessage("yo")
	if m.Role != RoleAssistant {
		t.Fatalf("role = %q, want %q", m.Role, RoleAssistant)
	}
	if len(m.Content) != 1 || m.Content[0].Text != "yo" {
		t.Fatalf("content = %+v", m.Content)
	}
}

func TestTextBlock(t *testing.T) {
	b := TextBlock("x")
	if b.Type != BlockText || b.Text != "x" {
		t.Fatalf("block = %+v", b)
	}
}

func TestResponseText(t *testing.T) {
	r := &Response{Content: []ContentBlock{
		TextBlock("Hello, "),
		{Type: BlockToolUse, ToolName: "get_weather"}, // must be skipped
		TextBlock("world"),
	}}
	if got := r.Text(); got != "Hello, world" {
		t.Fatalf("Text() = %q, want %q", got, "Hello, world")
	}
}

func TestResponseTextEmpty(t *testing.T) {
	r := &Response{}
	if got := r.Text(); got != "" {
		t.Fatalf("Text() = %q, want empty", got)
	}
}

func TestError(t *testing.T) {
	cause := errors.New("boom")
	e := &Error{Backend: "sdk", Op: "messages.new", Status: 429, Message: "rate limited", Err: cause}

	msg := e.Error()
	for _, want := range []string{"sdk", "messages.new", "rate limited"} {
		if !strings.Contains(msg, want) {
			t.Errorf("Error() = %q, missing %q", msg, want)
		}
	}
	if !errors.Is(e, cause) {
		t.Errorf("errors.Is(e, cause) = false, want true (Unwrap broken)")
	}
}

func TestErrorNilCause(t *testing.T) {
	e := &Error{Backend: "cli", Op: "messages.new", Message: "bad json"}
	if e.Error() == "" {
		t.Fatal("Error() empty")
	}
	if e.Unwrap() != nil {
		t.Fatal("Unwrap() should be nil when no cause")
	}
}

func TestSentinels(t *testing.T) {
	if ErrNoBackend == nil || ErrCLINotFound == nil {
		t.Fatal("sentinel errors must be non-nil")
	}
}
