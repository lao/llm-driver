package llmdriver

import (
	"context"
	"slices"
	"testing"
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
