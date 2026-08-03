package llmwrapper

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
	"time"
)

const (
	maxCLIOutputBytes   = 16 << 20
	cliCommandWaitDelay = 250 * time.Millisecond
)

var errCLIOutputTooLarge = errors.New("llmwrapper: CLI output exceeds limit")

type commandRunner func(context.Context, string, []string, []byte) ([]byte, []byte, int, error)

type boundedBuffer struct {
	buffer   bytes.Buffer
	limit    int
	overflow bool
}

func newBoundedBuffer(limit int) *boundedBuffer {
	return &boundedBuffer{limit: limit}
}

func (b *boundedBuffer) Write(data []byte) (int, error) {
	remaining := b.limit - b.buffer.Len()
	if len(data) <= remaining {
		return b.buffer.Write(data)
	}
	if remaining > 0 {
		_, _ = b.buffer.Write(data[:remaining])
	}
	b.overflow = true
	return max(remaining, 0), errCLIOutputTooLarge
}

func (b *boundedBuffer) Bytes() []byte {
	return b.buffer.Bytes()
}

func defaultCommandRunner(ctx context.Context, name string, args []string, stdin []byte) ([]byte, []byte, int, error) {
	path, err := exec.LookPath(name)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("%w: %s", ErrExecutableNotFound, name)
	}

	command := exec.CommandContext(ctx, path, args...)
	configureCommand(command)
	command.WaitDelay = cliCommandWaitDelay
	command.Stdin = bytes.NewReader(stdin)
	stdout := newBoundedBuffer(maxCLIOutputBytes)
	stderr := newBoundedBuffer(maxCLIOutputBytes)
	command.Stdout = stdout
	command.Stderr = stderr

	err = command.Run()
	if err == nil {
		return stdout.Bytes(), stderr.Bytes(), 0, nil
	}
	if ctx.Err() != nil {
		return stdout.Bytes(), stderr.Bytes(), 0, ctx.Err()
	}
	if stdout.overflow || stderr.overflow {
		return stdout.Bytes(), stderr.Bytes(), 0, errCLIOutputTooLarge
	}

	var exitError *exec.ExitError
	if errors.As(err, &exitError) {
		return stdout.Bytes(), stderr.Bytes(), exitError.ExitCode(), nil
	}
	return stdout.Bytes(), stderr.Bytes(), 0, err
}

func executeCLI(
	ctx context.Context,
	provider Provider,
	name string,
	args []string,
	stdin []byte,
	runner commandRunner,
) ([]byte, error) {
	stdout, stderr, exitCode, err := runner(ctx, name, args, stdin)
	if err != nil {
		return nil, &Error{
			Provider:  provider,
			Flavor:    FlavorCLI,
			Operation: "generate",
			Message:   strings.TrimSpace(string(stderr)),
			Err:       err,
		}
	}
	if exitCode != 0 {
		message := strings.TrimSpace(string(stderr))
		if message == "" {
			message = "CLI command failed"
		}
		return stdout, &Error{
			Provider:  provider,
			Flavor:    FlavorCLI,
			Operation: "generate",
			Status:    exitCode,
			Message:   message,
		}
	}
	return stdout, nil
}

// cliProcess is a running provider CLI whose stdout is consumed incrementally.
type cliProcess interface {
	// stdout returns the process's standard output.
	stdout() io.Reader
	// wait blocks for exit and reports the exit code and captured stderr. It is
	// safe to call more than once and concurrently with close.
	wait() (int, []byte, error)
	// close terminates the process tree and releases the pipe. It is idempotent.
	close() error
}

// streamingCommandRunner starts a CLI whose output is read while it runs. It
// complements commandRunner, which buffers a whole batch invocation.
type streamingCommandRunner func(context.Context, string, []string, []byte) (cliProcess, error)

func defaultStreamingCommandRunner(
	ctx context.Context,
	name string,
	args []string,
	stdin []byte,
) (cliProcess, error) {
	path, err := exec.LookPath(name)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrExecutableNotFound, name)
	}

	// The derived context lets close terminate the process tree through the same
	// platform-specific cancel hook that context cancellation uses.
	runCtx, cancel := context.WithCancel(ctx)
	command := exec.CommandContext(runCtx, path, args...)
	configureCommand(command)
	command.WaitDelay = cliCommandWaitDelay
	command.Stdin = bytes.NewReader(stdin)
	stderr := newBoundedBuffer(maxCLIOutputBytes)
	command.Stderr = stderr

	pipe, err := command.StdoutPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	if err := command.Start(); err != nil {
		cancel()
		return nil, err
	}
	return &execCLIProcess{
		parent:  ctx,
		cancel:  cancel,
		command: command,
		pipe:    pipe,
		stderr:  stderr,
	}, nil
}

type execCLIProcess struct {
	parent  context.Context
	cancel  context.CancelFunc
	command *exec.Cmd
	pipe    io.ReadCloser
	stderr  *boundedBuffer

	waitOnce sync.Once
	waitErr  error
}

func (p *execCLIProcess) stdout() io.Reader {
	return p.pipe
}

func (p *execCLIProcess) wait() (int, []byte, error) {
	p.waitOnce.Do(func() { p.waitErr = p.command.Wait() })

	stderr := p.stderr.Bytes()
	if p.waitErr == nil {
		return 0, stderr, nil
	}
	if err := p.parent.Err(); err != nil {
		return 0, stderr, err
	}
	if p.stderr.overflow {
		return 0, stderr, errCLIOutputTooLarge
	}
	var exitError *exec.ExitError
	if errors.As(p.waitErr, &exitError) {
		return exitError.ExitCode(), stderr, nil
	}
	return 0, stderr, p.waitErr
}

func (p *execCLIProcess) close() error {
	p.cancel()
	p.waitOnce.Do(func() { p.waitErr = p.command.Wait() })
	return nil
}

// cliEventParser maps a provider CLI's output lines to response text and builds
// the final response once the process exits.
type cliEventParser interface {
	// event consumes one non-empty output line and returns the text it adds,
	// which is empty for lines that carry no response text.
	event(line []byte) (string, error)
	// response builds the final response after a clean exit.
	response() (*Response, error)
}

// cliChunkStream drives a streaming CLI subprocess: it scans stdout lines into
// chunks and turns exit status into the same normalized errors executeCLI
// produces for the batch path.
type cliChunkStream struct {
	provider Provider
	process  cliProcess
	scanner  *bufio.Scanner
	parser   cliEventParser
	failure  error
}

func newCLIChunkStream(provider Provider, process cliProcess, parser cliEventParser) *cliChunkStream {
	scanner := bufio.NewScanner(process.stdout())
	scanner.Buffer(make([]byte, 64*1024), maxCLIOutputBytes)
	return &cliChunkStream{
		provider: provider,
		process:  process,
		scanner:  scanner,
		parser:   parser,
	}
}

func (s *cliChunkStream) recv() (Chunk, error) {
	if s.failure != nil {
		return Chunk{}, s.failure
	}
	for s.scanner.Scan() {
		line := bytes.TrimSpace(s.scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		text, err := s.parser.event(line)
		if err != nil {
			return Chunk{}, s.fail(err)
		}
		if text != "" {
			return Chunk{Text: text}, nil
		}
	}
	if err := s.scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			err = errCLIOutputTooLarge
		}
		return Chunk{}, s.fail(s.streamError(err, ""))
	}
	if err := s.exit(); err != nil {
		s.failure = err
		return Chunk{}, err
	}
	return Chunk{}, errStreamDone
}

func (s *cliChunkStream) finish() (*Response, error) {
	return s.parser.response()
}

func (s *cliChunkStream) close() error {
	return s.process.close()
}

// fail records a terminal parse failure and, for parity with the batch path,
// stamps the process exit code onto it when the CLI also exited nonzero.
// Remaining output is drained first: waiting on a process whose stdout is not
// being read would deadlock.
func (s *cliChunkStream) fail(err error) error {
	s.failure = err

	_, _ = io.Copy(io.Discard, s.process.stdout())
	exitCode, _, waitErr := s.process.wait()
	if waitErr != nil || exitCode == 0 {
		return err
	}
	var contextual *Error
	if errors.As(err, &contextual) && contextual.Status == 0 {
		contextual.Status = exitCode
	}
	return err
}

func (s *cliChunkStream) exit() error {
	exitCode, stderr, err := s.process.wait()
	if err != nil {
		return s.streamError(err, strings.TrimSpace(string(stderr)))
	}
	if exitCode != 0 {
		message := strings.TrimSpace(string(stderr))
		if message == "" {
			message = "CLI command failed"
		}
		failure := s.streamError(nil, message)
		failure.Status = exitCode
		return failure
	}
	return nil
}

func (s *cliChunkStream) streamError(err error, message string) *Error {
	return &Error{
		Provider:  s.provider,
		Flavor:    FlavorCLI,
		Operation: "generate_stream",
		Message:   message,
		Err:       err,
	}
}

// startCLIStream launches a streaming CLI and normalizes launch failures.
func startCLIStream(
	ctx context.Context,
	provider Provider,
	name string,
	args []string,
	stdin []byte,
	runner streamingCommandRunner,
) (cliProcess, error) {
	process, err := runner(ctx, name, args, stdin)
	if err != nil {
		return nil, &Error{
			Provider:  provider,
			Flavor:    FlavorCLI,
			Operation: "generate_stream",
			Err:       err,
		}
	}
	return process, nil
}

func renderTranscript(messages []Message) string {
	if len(messages) == 1 && messages[0].Role == RoleUser {
		return messages[0].Text
	}

	var transcript strings.Builder
	for i, message := range messages {
		if i > 0 {
			transcript.WriteString("\n\n")
		}
		if message.Role == RoleAssistant {
			transcript.WriteString("Assistant: ")
		} else {
			transcript.WriteString("User: ")
		}
		transcript.WriteString(message.Text)
	}
	return transcript.String()
}
