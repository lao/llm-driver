package llmwrapper

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeStreamingRunner replays scripted stdout lines through a pipe so streaming
// adapters observe them incrementally, without starting a real process.
type fakeStreamingRunner struct {
	lines    []string
	holdFrom int
	gate     chan struct{}
	stderr   []byte
	exitCode int
	startErr error
	waitErr  error

	calls  int
	name   string
	args   []string
	stdin  []byte
	closes int
}

type fakeCLIProcess struct {
	runner  *fakeStreamingRunner
	reader  *io.PipeReader
	writing chan struct{}
	closing chan struct{}
	once    sync.Once
}

func (r *fakeStreamingRunner) run(_ context.Context, name string, args []string, stdin []byte) (cliProcess, error) {
	r.calls++
	r.name = name
	r.args = append([]string(nil), args...)
	r.stdin = append([]byte(nil), stdin...)
	if r.startErr != nil {
		return nil, r.startErr
	}

	reader, writer := io.Pipe()
	process := &fakeCLIProcess{
		runner:  r,
		reader:  reader,
		writing: make(chan struct{}),
		closing: make(chan struct{}),
	}
	go func() {
		defer close(process.writing)
		for i, line := range r.lines {
			if r.gate != nil && i == r.holdFrom {
				select {
				case <-r.gate:
				case <-process.closing:
					_ = writer.CloseWithError(io.ErrClosedPipe)
					return
				}
			}
			if _, err := io.WriteString(writer, line+"\n"); err != nil {
				return
			}
		}
		_ = writer.Close()
	}()
	return process, nil
}

func (p *fakeCLIProcess) stdout() io.Reader { return p.reader }

func (p *fakeCLIProcess) wait() (int, []byte, error) {
	<-p.writing
	return p.runner.exitCode, p.runner.stderr, p.runner.waitErr
}

func (p *fakeCLIProcess) close() error {
	p.once.Do(func() {
		p.runner.closes++
		close(p.closing)
		_ = p.reader.CloseWithError(io.ErrClosedPipe)
	})
	<-p.writing
	return nil
}

func (r *fakeStreamingRunner) assertCall(t *testing.T, name string, args []string, stdin string) {
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

// newCLIStreamClient wires a CLI backend to a fake streaming runner. It skips
// New so tests can inject the runner without a public seam.
func newCLIStreamClient(config Config, runner streamingCommandRunner) *Client {
	cliConfig := cliBackendConfig{
		model:        config.Model,
		path:         config.CLIPath,
		args:         config.CLIArgs,
		streamRunner: runner,
	}
	var target backend
	if config.Provider == ProviderClaude {
		target = newClaudeCLIBackend(cliConfig)
	} else {
		target = newCodexCLIBackend(cliConfig)
	}
	return &Client{
		provider: config.Provider,
		flavor:   FlavorCLI,
		model:    config.Model,
		backend:  target,
	}
}

func TestStreamingCommandMissingExecutable(t *testing.T) {
	t.Parallel()

	_, err := defaultStreamingCommandRunner(
		context.Background(),
		"llmwrapper-command-that-does-not-exist",
		nil,
		nil,
	)
	if !errors.Is(err, ErrExecutableNotFound) {
		t.Fatalf("defaultStreamingCommandRunner() error = %v, want ErrExecutableNotFound", err)
	}
}

func TestStreamingCommandReadsOutputIncrementally(t *testing.T) {
	t.Setenv("LLMWRAPPER_STREAM_HELPER", "lines")

	process, err := defaultStreamingCommandRunner(
		context.Background(),
		os.Args[0],
		[]string{"-test.run=^TestStreamingCommandHelperProcess$"},
		[]byte("prompt"),
	)
	if err != nil {
		t.Fatalf("defaultStreamingCommandRunner(): %v", err)
	}
	defer process.close()

	// The helper blocks after printing, so reaching both lines proves stdout is
	// delivered before the process exits.
	buffer := make([]byte, len("line-1\nline-2\n"))
	if _, err := io.ReadFull(process.stdout(), buffer); err != nil {
		t.Fatalf("read stdout: %v", err)
	}
	if string(buffer) != "line-1\nline-2\n" {
		t.Fatalf("stdout = %q", buffer)
	}
}

func TestStreamingCommandCloseTerminatesProcessTree(t *testing.T) {
	readyPath := filepath.Join(t.TempDir(), "ready")
	t.Setenv("LLMWRAPPER_STREAM_HELPER", "parent")
	t.Setenv("LLMWRAPPER_HELPER_READY", readyPath)

	process, err := defaultStreamingCommandRunner(
		context.Background(),
		os.Args[0],
		[]string{"-test.run=^TestStreamingCommandHelperProcess$"},
		nil,
	)
	if err != nil {
		t.Fatalf("defaultStreamingCommandRunner(): %v", err)
	}

	buffer := make([]byte, len("line-1\n"))
	if _, err := io.ReadFull(process.stdout(), buffer); err != nil {
		t.Fatalf("read stdout: %v", err)
	}

	closed := make(chan error, 1)
	go func() { closed <- process.close() }()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("close(): %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("close() did not return")
	}

	descendant := readHelperPID(t, readyPath)
	exitDeadline := time.Now().Add(5 * time.Second)
	for descendantAlive(descendant) {
		if time.Now().After(exitDeadline) {
			t.Fatalf("descendant process %d survived close()", descendant)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestStreamingCommandCancellationIsReported(t *testing.T) {
	t.Setenv("LLMWRAPPER_STREAM_HELPER", "lines")

	ctx, cancel := context.WithCancel(context.Background())
	process, err := defaultStreamingCommandRunner(
		ctx,
		os.Args[0],
		[]string{"-test.run=^TestStreamingCommandHelperProcess$"},
		nil,
	)
	if err != nil {
		t.Fatalf("defaultStreamingCommandRunner(): %v", err)
	}
	defer process.close()

	buffer := make([]byte, len("line-1\n"))
	if _, err := io.ReadFull(process.stdout(), buffer); err != nil {
		t.Fatalf("read stdout: %v", err)
	}
	cancel()
	_, _ = io.ReadAll(process.stdout())

	_, _, waitErr := process.wait()
	if !errors.Is(waitErr, context.Canceled) {
		t.Fatalf("wait() error = %v, want context.Canceled", waitErr)
	}
}

func TestStreamingCommandReportsExitCodeAndStderr(t *testing.T) {
	t.Setenv("LLMWRAPPER_STREAM_HELPER", "fail")

	process, err := defaultStreamingCommandRunner(
		context.Background(),
		os.Args[0],
		[]string{"-test.run=^TestStreamingCommandHelperProcess$"},
		nil,
	)
	if err != nil {
		t.Fatalf("defaultStreamingCommandRunner(): %v", err)
	}
	defer process.close()

	if _, err := io.ReadAll(process.stdout()); err != nil {
		t.Fatalf("read stdout: %v", err)
	}
	exitCode, stderr, waitErr := process.wait()
	if waitErr != nil {
		t.Fatalf("wait() error = %v", waitErr)
	}
	if exitCode != 9 {
		t.Fatalf("exit code = %d, want 9", exitCode)
	}
	if !strings.Contains(string(stderr), "helper failed") {
		t.Fatalf("stderr = %q", stderr)
	}
}

func TestCLIChunkStreamRejectsOversizedLine(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{lines: []string{strings.Repeat("x", maxCLIOutputBytes+1)}}
	process, err := runner.run(context.Background(), "fake", nil, nil)
	if err != nil {
		t.Fatalf("run(): %v", err)
	}
	stream := newCLIChunkStream(ProviderClaude, process, &countingEventParser{})
	defer stream.close()

	_, err = stream.recv()
	if !errors.Is(err, errCLIOutputTooLarge) {
		t.Fatalf("recv() error = %v, want errCLIOutputTooLarge", err)
	}
	var target *Error
	if !errors.As(err, &target) || target.Operation != "generate_stream" || target.Provider != ProviderClaude {
		t.Fatalf("recv() error = %#v, want normalized *Error", err)
	}
}

func TestCLIChunkStreamReportsNonzeroExit(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{
		lines:    []string{"one"},
		exitCode: 17,
		stderr:   []byte("authentication failed\n"),
	}
	process, err := runner.run(context.Background(), "fake", nil, nil)
	if err != nil {
		t.Fatalf("run(): %v", err)
	}
	stream := newCLIChunkStream(ProviderOpenAI, process, &countingEventParser{})
	defer stream.close()

	if _, err := stream.recv(); err != nil {
		t.Fatalf("first recv(): %v", err)
	}
	_, err = stream.recv()
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("recv() error = %T %v, want *Error", err, err)
	}
	if target.Status != 17 || target.Message != "authentication failed" {
		t.Fatalf("error = %+v", target)
	}
}

// countingEventParser treats each line as one chunk of text.
type countingEventParser struct{ lines int }

func (p *countingEventParser) event(line []byte) (string, error) {
	p.lines++
	return string(line), nil
}

func (p *countingEventParser) response() (*Response, error) {
	return &Response{}, nil
}

func readHelperPID(t *testing.T, path string) int {
	t.Helper()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ready file: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil {
		t.Fatalf("parse pid %q: %v", raw, err)
	}
	return pid
}

func TestStreamingCommandHelperProcess(t *testing.T) {
	if os.Getenv("LLMWRAPPER_HELPER_DESCENDANT") == "1" {
		time.Sleep(30 * time.Second)
		return
	}

	switch os.Getenv("LLMWRAPPER_STREAM_HELPER") {
	case "":
		return
	case "fail":
		fmt.Fprintln(os.Stderr, "helper failed")
		os.Exit(9)
	case "parent":
		child := exec.Command(os.Args[0], "-test.run=^TestStreamingCommandHelperProcess$")
		child.Env = append(os.Environ(), "LLMWRAPPER_HELPER_DESCENDANT=1")
		// The descendant inherits stdout so close() must rely on WaitDelay
		// instead of waiting for every holder of the pipe.
		child.Stdout = os.Stdout
		child.Stderr = os.Stderr
		if err := child.Start(); err != nil {
			t.Fatalf("start descendant: %v", err)
		}
		if err := os.WriteFile(os.Getenv("LLMWRAPPER_HELPER_READY"), []byte(strconv.Itoa(child.Process.Pid)), 0o600); err != nil {
			t.Fatalf("write ready file: %v", err)
		}
		fmt.Fprintln(os.Stdout, "line-1")
	case "lines":
		fmt.Fprintln(os.Stdout, "line-1")
		fmt.Fprintln(os.Stdout, "line-2")
	}

	for {
		time.Sleep(time.Hour)
	}
}
