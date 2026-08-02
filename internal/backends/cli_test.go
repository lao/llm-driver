package backends

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/lao/claudewrap/internal/core"
)

func TestPromptRenderSingleUser(t *testing.T) {
	t.Parallel()

	got := renderTranscript([]core.Message{core.User("Hello from stdin")})
	if got != "Hello from stdin" {
		t.Fatalf("renderTranscript() = %q", got)
	}
}

func TestPromptRenderMultiTurn(t *testing.T) {
	t.Parallel()

	got := renderTranscript([]core.Message{
		core.User("First question"),
		core.Assistant("First answer"),
		core.User("Follow-up"),
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
	if !errors.Is(err, core.ErrExecutableNotFound) {
		t.Fatalf("defaultCommandRunner() error = %v, want ErrExecutableNotFound", err)
	}
}

func TestCommandPreservesDeadline(t *testing.T) {
	t.Setenv("LLMWRAPPER_HELPER_PROCESS", "sleep")
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

func TestCommandCancellationDoesNotWaitForDescendant(t *testing.T) {
	readyPath := filepath.Join(t.TempDir(), "ready")
	t.Setenv("LLMWRAPPER_HELPER_PROCESS", "parent")
	t.Setenv("LLMWRAPPER_HELPER_READY", readyPath)

	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() {
		_, _, _, err := defaultCommandRunner(
			ctx,
			os.Args[0],
			[]string{"-test.run=^TestCommandHelperProcess$"},
			nil,
		)
		result <- err
	}()

	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(readyPath); err == nil {
			break
		} else if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("os.Stat(%q): %v", readyPath, err)
		}
		if time.Now().After(deadline) {
			t.Fatal("descendant process did not start")
		}
		time.Sleep(10 * time.Millisecond)
	}

	cancel()
	var err error
	select {
	case err = <-result:
	case <-time.After(time.Second):
		select {
		case err = <-result:
			t.Error("defaultCommandRunner() waited for descendant stdio after cancellation")
		case <-time.After(4 * time.Second):
			t.Fatal("defaultCommandRunner() did not return after descendant exited")
		}
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("defaultCommandRunner() error = %v, want context.Canceled", err)
	}

	pidBytes, readErr := os.ReadFile(readyPath)
	if readErr != nil {
		t.Fatalf("read descendant pid: %v", readErr)
	}
	descendant, convErr := strconv.Atoi(strings.TrimSpace(string(pidBytes)))
	if convErr != nil {
		t.Fatalf("parse descendant pid %q: %v", pidBytes, convErr)
	}
	exitDeadline := time.Now().Add(5 * time.Second)
	for descendantAlive(descendant) {
		if time.Now().After(exitDeadline) {
			t.Fatalf("descendant process %d survived cancellation", descendant)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestCommandHelperProcess(t *testing.T) {
	if os.Getenv("LLMWRAPPER_HELPER_DESCENDANT") == "1" {
		time.Sleep(30 * time.Second)
		return
	}

	switch os.Getenv("LLMWRAPPER_HELPER_PROCESS") {
	case "":
		return
	case "parent":
		child := exec.Command(os.Args[0], "-test.run=^TestCommandHelperProcess$")
		child.Env = append(os.Environ(), "LLMWRAPPER_HELPER_DESCENDANT=1")
		child.Stdout = os.Stdout
		child.Stderr = os.Stderr
		if err := child.Start(); err != nil {
			t.Fatalf("start descendant: %v", err)
		}
		if err := os.WriteFile(os.Getenv("LLMWRAPPER_HELPER_READY"), []byte(strconv.Itoa(child.Process.Pid)), 0o600); err != nil {
			t.Fatalf("write ready file: %v", err)
		}
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
