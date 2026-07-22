package claudewrap

import (
	"context"
	"errors"
	"os/exec"
	"slices"
	"testing"
)

// fakeRun records the invocation and returns canned output.
type fakeRun struct {
	gotName  string
	gotArgs  []string
	gotStdin []byte

	stdout []byte
	stderr []byte
	exit   int
	err    error
}

func (f *fakeRun) run(_ context.Context, name string, args []string, stdin []byte) ([]byte, []byte, int, error) {
	f.gotName, f.gotArgs, f.gotStdin = name, args, stdin
	return f.stdout, f.stderr, f.exit, f.err
}

const okCLIBody = `{"type":"result","subtype":"success","is_error":false,` +
	`"result":"ok","stop_reason":"end_turn","session_id":"sess_1",` +
	`"usage":{"input_tokens":10,"output_tokens":188,"cache_read_input_tokens":0,"cache_creation_input_tokens":33332}}`

func TestCLIMessagesNew(t *testing.T) {
	f := &fakeRun{stdout: []byte(okCLIBody)}
	c, err := New(WithCLI(WithCLIPath("claude"), withCLIRunner(f.run)))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := c.Messages.New(context.Background(), MessageParams{
		MaxTokens: 64,
		Messages:  []Message{UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	if resp.Text() != "ok" {
		t.Errorf("Text() = %q", resp.Text())
	}
	if resp.Backend != "cli" {
		t.Errorf("Backend = %q, want cli", resp.Backend)
	}
	if resp.StopReason != StopEndTurn {
		t.Errorf("StopReason = %q", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 188 || resp.Usage.CacheCreationTokens != 33332 {
		t.Errorf("Usage = %+v", resp.Usage)
	}
	// Prompt goes on stdin (injection-safe), not as a shell string.
	if string(f.gotStdin) != "hi" {
		t.Errorf("stdin = %q, want hi", f.gotStdin)
	}
	for _, want := range [][2]string{{"-p", ""}, {"--output-format", "json"}, {"--model", DefaultModel}} {
		if !argPresent(f.gotArgs, want[0], want[1]) {
			t.Errorf("args %v missing %q %q", f.gotArgs, want[0], want[1])
		}
	}
}

func TestCLISystemPrompt(t *testing.T) {
	f := &fakeRun{stdout: []byte(okCLIBody)}
	c, _ := New(WithCLI(withCLIRunner(f.run)))
	_, err := c.Messages.New(context.Background(), MessageParams{
		MaxTokens: 16, System: "be terse",
		Messages: []Message{UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	if !argPresent(f.gotArgs, "--append-system-prompt", "be terse") {
		t.Errorf("args %v missing system prompt", f.gotArgs)
	}
}

func TestCLIExitError(t *testing.T) {
	f := &fakeRun{exit: 1, stderr: []byte("boom")}
	c, _ := New(WithCLI(withCLIRunner(f.run)))
	_, err := c.Messages.New(context.Background(), MessageParams{MaxTokens: 16, Messages: []Message{UserMessage("hi")}})
	var e *Error
	if !errors.As(err, &e) {
		t.Fatalf("err = %T, want *Error", err)
	}
	if e.Backend != "cli" || e.Status != 1 {
		t.Errorf("err = %+v, want cli/1", e)
	}
}

func TestCLIBadJSON(t *testing.T) {
	f := &fakeRun{stdout: []byte("not json")}
	c, _ := New(WithCLI(withCLIRunner(f.run)))
	_, err := c.Messages.New(context.Background(), MessageParams{MaxTokens: 16, Messages: []Message{UserMessage("hi")}})
	var e *Error
	if !errors.As(err, &e) {
		t.Fatalf("err = %T, want *Error", err)
	}
}

func TestCLINotFound(t *testing.T) {
	f := &fakeRun{err: exec.ErrNotFound}
	c, _ := New(WithCLI(withCLIRunner(f.run)))
	_, err := c.Messages.New(context.Background(), MessageParams{MaxTokens: 16, Messages: []Message{UserMessage("hi")}})
	if !errors.Is(err, ErrCLINotFound) {
		t.Fatalf("err = %v, want ErrCLINotFound", err)
	}
}

// argPresent reports whether flag appears in args, optionally followed by value
// (value "" means the flag is a bare switch).
func argPresent(args []string, flag, value string) bool {
	i := slices.Index(args, flag)
	if i < 0 {
		return false
	}
	if value == "" {
		return true
	}
	return i+1 < len(args) && args[i+1] == value
}
