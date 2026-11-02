package backends

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/lao/llm-driver/internal/core"
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
		"llmdriver-command-that-does-not-exist",
		nil,
		nil,
	)
	if !errors.Is(err, core.ErrExecutableNotFound) {
		t.Fatalf("defaultCommandRunner() error = %v, want ErrExecutableNotFound", err)
	}
}

func TestCommandPreservesDeadline(t *testing.T) {
	t.Setenv("LLM_DRIVER_HELPER_PROCESS", "sleep")
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
	t.Setenv("LLM_DRIVER_HELPER_PROCESS", "parent")
	t.Setenv("LLM_DRIVER_HELPER_READY", readyPath)

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
	if os.Getenv("LLM_DRIVER_HELPER_PROCESS") == "probe" {
		cwd, err := os.Getwd()
		if err != nil {
			t.Fatalf("getwd: %v", err)
		}
		fmt.Fprintf(os.Stdout, "probe=%s cwd=%s", os.Getenv("LLM_DRIVER_PROBE"), cwd)
		return
	}
	if os.Getenv("LLM_DRIVER_HELPER_DESCENDANT") == "1" {
		time.Sleep(30 * time.Second)
		return
	}

	switch os.Getenv("LLM_DRIVER_HELPER_PROCESS") {
	case "":
		return
	case "parent":
		child := exec.Command(os.Args[0], "-test.run=^TestCommandHelperProcess$")
		child.Env = append(os.Environ(), "LLM_DRIVER_HELPER_DESCENDANT=1")
		child.Stdout = os.Stdout
		child.Stderr = os.Stderr
		if err := child.Start(); err != nil {
			t.Fatalf("start descendant: %v", err)
		}
		if err := os.WriteFile(os.Getenv("LLM_DRIVER_HELPER_READY"), []byte(strconv.Itoa(child.Process.Pid)), 0o600); err != nil {
			t.Fatalf("write ready file: %v", err)
		}
	}

	for {
		time.Sleep(time.Hour)
	}
}

func TestClipMessageStripsTerminalEscapes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "SGR color", in: "\x1b[93m\x1b[1m! \x1b[0mpermission requested\n", want: "! permission requested"},
		{name: "private CSI", in: "before\x1b[?25lafter", want: "beforeafter"},
		{name: "OSC 52 BEL", in: "before\x1b]52;c;SGVscG8=\x07after", want: "beforeafter"},
		{name: "OSC 0 ST", in: "before\x1b]0;spoofed\x1b\\after", want: "beforeafter"},
		{name: "DCS", in: "before\x1bP1;2|payload\x1b\\after", want: "beforeafter"},
		{name: "single ESC", in: "before\x1b(Bafter", want: "beforeafter"},
		{name: "C0 and C1 controls", in: "a\x01b\x07c\x9bd\te\nf", want: "abcd\te\nf"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := clipMessage(test.in); got != test.want {
				t.Fatalf("clipMessage(%q) = %q, want %q", test.in, got, test.want)
			}
		})
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

func TestCommandRunnerAppliesEnvAndDir(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	resolvedDir, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Fatalf("EvalSymlinks(%q): %v", dir, err)
	}

	run := commandRunnerFor([]string{"LLM_DRIVER_HELPER_PROCESS=probe", "LLM_DRIVER_PROBE=hello"}, dir)
	stdout, _, exitCode, err := run(
		context.Background(),
		os.Args[0],
		[]string{"-test.run=^TestCommandHelperProcess$"},
		nil,
	)
	if err != nil || exitCode != 0 {
		t.Fatalf("run() = exit %d, err %v", exitCode, err)
	}
	want := fmt.Sprintf("probe=hello cwd=%s", resolvedDir)
	if !strings.Contains(string(stdout), want) {
		t.Fatalf("stdout = %q, want it to contain %q", stdout, want)
	}
}
