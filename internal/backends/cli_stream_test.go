package backends

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lao/llmwrapper/internal/core"
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

func TestStreamingCommandMissingExecutable(t *testing.T) {
	t.Parallel()

	_, err := defaultStreamingCommandRunner(
		context.Background(),
		"llmwrapper-command-that-does-not-exist",
		nil,
		nil,
	)
	if !errors.Is(err, core.ErrExecutableNotFound) {
		t.Fatalf("defaultStreamingCommandRunner() error = %v, want core.ErrExecutableNotFound", err)
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
	stream := newCLIChunkStream(core.ProviderClaude, process, &countingEventParser{})
	defer stream.Close()

	_, err = stream.Recv()
	if !errors.Is(err, errCLIOutputTooLarge) {
		t.Fatalf("recv() error = %v, want errCLIOutputTooLarge", err)
	}
	var target *core.Error
	if !errors.As(err, &target) || target.Operation != "generate_stream" || target.Provider != core.ProviderClaude {
		t.Fatalf("recv() error = %#v, want normalized *core.Error", err)
	}
}

// endlessCLIProcess writes one line and then never stops, and its wait blocks
// until stdout is closed: the shape of a CLI that keeps producing output after
// the stream has already failed and has not exited on its own.
type endlessCLIProcess struct {
	reader    *io.PipeReader
	waited    chan struct{}
	closeOnce sync.Once
}

func newEndlessCLIProcess(first string) *endlessCLIProcess {
	reader, writer := io.Pipe()
	process := &endlessCLIProcess{reader: reader, waited: make(chan struct{})}
	go func() {
		if _, err := io.WriteString(writer, first+"\n"); err != nil {
			return
		}
		filler := bytes.Repeat([]byte("x"), 64*1024)
		for {
			if _, err := writer.Write(filler); err != nil {
				return
			}
		}
	}()
	return process
}

func (p *endlessCLIProcess) stdout() io.Reader { return p.reader }

func (p *endlessCLIProcess) wait() (int, []byte, error) {
	<-p.waited
	return 0, nil, nil
}

func (p *endlessCLIProcess) close() error {
	p.closeOnce.Do(func() {
		_ = p.reader.CloseWithError(io.ErrClosedPipe)
		close(p.waited)
	})
	return nil
}

func TestCLIChunkStreamBoundsDrainAfterFailure(t *testing.T) {
	t.Parallel()

	// A CLI that keeps writing after the line that failed the stream must not be
	// able to hold recv open: the post-failure drain is bounded like every other
	// read on this path.
	process := newEndlessCLIProcess("not-json")
	stream := newCLIChunkStream(core.ProviderClaude, process, &claudeCLIStreamParser{model: "claude-test"})
	defer stream.Close()

	failed := make(chan error, 1)
	go func() {
		_, err := stream.Recv()
		failed <- err
	}()

	select {
	case err := <-failed:
		var target *core.Error
		if !errors.As(err, &target) || target.Operation != "generate_stream" {
			t.Fatalf("recv() error = %#v, want normalized *core.Error", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("recv() did not return: the drain after a failed line is unbounded")
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
	stream := newCLIChunkStream(core.ProviderOpenAI, process, &countingEventParser{})
	defer stream.Close()

	if _, err := stream.Recv(); err != nil {
		t.Fatalf("first recv(): %v", err)
	}
	_, err = stream.Recv()
	var target *core.Error
	if !errors.As(err, &target) {
		t.Fatalf("recv() error = %T %v, want *core.Error", err, err)
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

func (p *countingEventParser) response() (*core.Response, error) {
	return &core.Response{}, nil
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
	case "exit":
		fmt.Fprintln(os.Stdout, "line-1")
		fmt.Fprintln(os.Stdout, "line-2")
		return
	}

	for {
		time.Sleep(time.Hour)
	}
}

func TestCLIChunkStreamSkipsBlankLines(t *testing.T) {
	t.Parallel()

	// CLIs pad their JSONL output with blank and whitespace-only lines; those
	// must never reach the parser or surface as chunks.
	runner := &fakeStreamingRunner{lines: []string{"", "one", "   ", "", "two", ""}}
	process, err := runner.run(context.Background(), "fake", nil, nil)
	if err != nil {
		t.Fatalf("run(): %v", err)
	}
	parser := &countingEventParser{}
	stream := newCLIChunkStream(core.ProviderClaude, process, parser)
	defer stream.Close()

	var chunks []string
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, core.ErrStreamDone) {
			break
		}
		if err != nil {
			t.Fatalf("recv(): %v", err)
		}
		chunks = append(chunks, chunk.Text)
	}
	if strings.Join(chunks, "|") != "one|two" {
		t.Fatalf("chunks = %#v, want the two non-blank lines", chunks)
	}
	if parser.lines != 2 {
		t.Fatalf("parser saw %d lines, want 2", parser.lines)
	}
}

func TestCLIStreamCancellationIsNormalized(t *testing.T) {
	t.Parallel()

	// A cancelled run ends with the process reporting the context error rather
	// than an exit code, and that has to reach the caller as a normalized error.
	runner := &fakeStreamingRunner{
		lines:   claudeStreamJSONLines()[:5],
		stderr:  []byte("interrupted\n"),
		waitErr: context.Canceled,
	}
	client := newCLIStreamClient(core.Config{
		Provider: core.ProviderClaude,
		Flavor:   core.FlavorCLI,
		Model:    "claude-test",
	}, runner.run)

	stream, err := client.GenerateStream(context.Background(), streamRequest())
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	chunks, err, _ := collectStream(t, stream)
	if strings.Join(chunks, "|") != "Hello, |world" {
		t.Fatalf("chunks = %#v, want the chunks delivered before cancellation", chunks)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("stream error = %v, want context.Canceled", err)
	}
	var target *core.Error
	if !errors.As(err, &target) {
		t.Fatalf("stream error = %T %v, want *core.Error", err, err)
	}
	if target.Provider != core.ProviderClaude || target.Flavor != core.FlavorCLI || target.Operation != "generate_stream" {
		t.Fatalf("error context = %+v", target)
	}
	if target.Message != "interrupted" {
		t.Fatalf("message = %q, want the captured stderr", target.Message)
	}
	if target.Status != 0 {
		t.Fatalf("status = %d, want 0 for a cancelled run", target.Status)
	}
}

func TestCLIChunkStreamNonzeroExitWithoutStderr(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{lines: []string{"one"}, exitCode: 2}
	process, err := runner.run(context.Background(), "fake", nil, nil)
	if err != nil {
		t.Fatalf("run(): %v", err)
	}
	stream := newCLIChunkStream(core.ProviderOpenAI, process, &countingEventParser{})
	defer stream.Close()

	if _, err := stream.Recv(); err != nil {
		t.Fatalf("first recv(): %v", err)
	}
	_, err = stream.Recv()
	var target *core.Error
	if !errors.As(err, &target) {
		t.Fatalf("recv() error = %T %v, want *core.Error", err, err)
	}
	if target.Status != 2 {
		t.Fatalf("status = %d, want 2", target.Status)
	}
	// A silent failure still has to say something.
	if target.Message == "" {
		t.Fatal("message is empty for a nonzero exit without stderr")
	}
}

func TestCLIChunkStreamRepeatsFailureAfterError(t *testing.T) {
	t.Parallel()

	runner := &fakeStreamingRunner{lines: []string{"not-json", `{"type":"result","subtype":"success"}`}}
	process, err := runner.run(context.Background(), "fake", nil, nil)
	if err != nil {
		t.Fatalf("run(): %v", err)
	}
	stream := newCLIChunkStream(core.ProviderClaude, process, &claudeCLIStreamParser{model: "claude-test"})
	defer stream.Close()

	_, first := stream.Recv()
	if first == nil {
		t.Fatal("recv() error = nil for a malformed line")
	}
	// A stream that has already failed keeps reporting the same failure instead
	// of resuming into later output or waiting on the process a second time.
	_, second := stream.Recv()
	if second != first {
		t.Fatalf("second recv() error = %v, want the recorded failure %v", second, first)
	}
}

func TestStreamingCommandStartFailureIsReported(t *testing.T) {
	t.Parallel()

	// Executable permissions but not an executable format: LookPath succeeds and
	// Start is what fails, so the runner must report it instead of handing back a
	// process that never ran.
	path := filepath.Join(t.TempDir(), "not-a-binary")
	if err := os.WriteFile(path, []byte("\x00\x01not an executable"), 0o755); err != nil {
		t.Fatalf("write fake executable: %v", err)
	}

	process, err := defaultStreamingCommandRunner(context.Background(), path, nil, nil)
	if err == nil {
		_ = process.close()
		t.Fatal("defaultStreamingCommandRunner() error = nil for an unstartable binary")
	}
	if process != nil {
		t.Fatalf("process = %v, want nil after a failed start", process)
	}
	if errors.Is(err, core.ErrExecutableNotFound) {
		t.Fatalf("error = %v, want a start failure rather than a lookup failure", err)
	}
}

func TestStreamingCommandReportsCleanExit(t *testing.T) {
	t.Setenv("LLMWRAPPER_STREAM_HELPER", "exit")

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

	stdout, err := io.ReadAll(process.stdout())
	if err != nil {
		t.Fatalf("read stdout: %v", err)
	}
	if !strings.Contains(string(stdout), "line-1\nline-2\n") {
		t.Fatalf("stdout = %q", stdout)
	}

	exitCode, _, waitErr := process.wait()
	if waitErr != nil {
		t.Fatalf("wait() error = %v", waitErr)
	}
	if exitCode != 0 {
		t.Fatalf("exit code = %d, want 0", exitCode)
	}
	// wait is documented as safe to call more than once.
	if _, _, waitErr := process.wait(); waitErr != nil {
		t.Fatalf("second wait() error = %v", waitErr)
	}
}

func TestStreamCloseIsSafeConcurrentlyWithNext(t *testing.T) {
	t.Parallel()

	// The runner holds output after the first chunk, so Next is genuinely
	// blocked when Close runs.
	runner := &fakeStreamingRunner{
		lines:    claudeStreamJSONLines(),
		holdFrom: 4,
		gate:     make(chan struct{}),
	}
	client := newCLIStreamClient(core.Config{
		Provider: core.ProviderClaude,
		Flavor:   core.FlavorCLI,
		Model:    "claude-test",
	}, runner.run)

	stream, err := client.GenerateStream(context.Background(), streamRequest())
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	if !stream.Next() {
		t.Fatalf("Next() = false: %v", stream.Err())
	}

	advanced := make(chan bool, 1)
	go func() { advanced <- stream.Next() }()

	if err := stream.Close(); err != nil {
		t.Fatalf("Close(): %v", err)
	}
	select {
	case got := <-advanced:
		if got {
			t.Fatal("Next() = true after a concurrent Close()")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Next() did not return after Close()")
	}
}

func TestStreamCloseLeavesNoGoroutines(t *testing.T) {
	// Not parallel: the goroutine count is process-wide.
	baseline := runtime.NumGoroutine()

	for range 20 {
		runner := &fakeStreamingRunner{
			lines:    claudeStreamJSONLines(),
			holdFrom: 4,
			gate:     make(chan struct{}),
		}
		client := newCLIStreamClient(core.Config{
			Provider: core.ProviderClaude,
			Flavor:   core.FlavorCLI,
			Model:    "claude-test",
		}, runner.run)

		stream, err := client.GenerateStream(context.Background(), streamRequest())
		if err != nil {
			t.Fatalf("GenerateStream(): %v", err)
		}
		if !stream.Next() {
			t.Fatalf("Next() = false: %v", stream.Err())
		}
		// Abandon the stream mid-flight; Close must release everything it started.
		if err := stream.Close(); err != nil {
			t.Fatalf("Close(): %v", err)
		}
	}

	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > baseline+2 {
		if time.Now().After(deadline) {
			t.Fatalf("goroutines = %d, baseline %d", runtime.NumGoroutine(), baseline)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
