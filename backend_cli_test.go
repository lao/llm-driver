package llmwrapper

import (
	"context"
	"errors"
	"os"
	"slices"
	"testing"
	"time"
)

type fakeCommandRunner struct {
	name     string
	args     []string
	stdin    []byte
	stdout   []byte
	stderr   []byte
	exitCode int
	err      error
	calls    int
}

func (r *fakeCommandRunner) run(_ context.Context, name string, args []string, stdin []byte) ([]byte, []byte, int, error) {
	r.calls++
	r.name = name
	r.args = append([]string(nil), args...)
	r.stdin = append([]byte(nil), stdin...)
	return r.stdout, r.stderr, r.exitCode, r.err
}

func (r *fakeCommandRunner) assertCall(t *testing.T, name string, args []string, stdin string) {
	t.Helper()
	if r.calls != 1 {
		t.Fatalf("runner calls = %d, want 1", r.calls)
	}
	if r.name != name {
		t.Fatalf("executable = %q, want %q", r.name, name)
	}
	if !slices.Equal(r.args, args) {
		t.Fatalf("args = %#v, want %#v", r.args, args)
	}
	if string(r.stdin) != stdin {
		t.Fatalf("stdin = %q, want %q", r.stdin, stdin)
	}
}

func TestPromptRenderSingleUser(t *testing.T) {
	t.Parallel()

	got := renderTranscript([]Message{User("Hello from stdin")})
	if got != "Hello from stdin" {
		t.Fatalf("renderTranscript() = %q", got)
	}
}

func TestPromptRenderMultiTurn(t *testing.T) {
	t.Parallel()

	got := renderTranscript([]Message{
		User("First question"),
		Assistant("First answer"),
		User("Follow-up"),
	})
	want := "User: First question\n\nAssistant: First answer\n\nUser: Follow-up"
	if got != want {
		t.Fatalf("renderTranscript() = %q, want %q", got, want)
	}
}

func TestCommandMissingExecutable(t *testing.T) {
	t.Parallel()

	_, _, _, err := defaultCommandRunner(
		context.Background(),
		"llmwrapper-command-that-does-not-exist",
		nil,
		nil,
	)
	if !errors.Is(err, ErrExecutableNotFound) {
		t.Fatalf("defaultCommandRunner() error = %v, want ErrExecutableNotFound", err)
	}
}

func TestCommandPreservesDeadline(t *testing.T) {
	t.Setenv("LLMWRAPPER_HELPER_PROCESS", "1")
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, _, _, err := defaultCommandRunner(
		ctx,
		os.Args[0],
		[]string{"-test.run=^TestCommandHelperProcess$"},
		nil,
	)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("defaultCommandRunner() error = %v, want context deadline", err)
	}
}

func TestCommandHelperProcess(t *testing.T) {
	if os.Getenv("LLMWRAPPER_HELPER_PROCESS") != "1" {
		return
	}
	for {
		time.Sleep(time.Hour)
	}
}

func TestCommandOutputIsBounded(t *testing.T) {
	t.Parallel()

	buffer := newBoundedBuffer(4)
	written, err := buffer.Write([]byte("123456"))
	if written != 4 || !errors.Is(err, errCLIOutputTooLarge) {
		t.Fatalf("Write() = %d, %v", written, err)
	}
	if got := string(buffer.Bytes()); got != "1234" {
		t.Fatalf("buffer = %q", got)
	}
}
