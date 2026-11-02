package backends

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
	"unicode"
	"unicode/utf8"

	"github.com/lao/llm-driver/internal/core"
)

const (
	maxCLIOutputBytes   = 64 << 20
	cliCommandWaitDelay = 250 * time.Millisecond
	// cliFailDrainTimeout bounds the best-effort drain-and-stamp on the failure
	// path so a CLI that emits its final line and then hangs (no more bytes, no
	// exit, no stdout EOF) cannot hold recv open forever.
	cliFailDrainTimeout = 2 * time.Second
)

var errCLIOutputTooLarge = errors.New("llmdriver: CLI output exceeds limit")

// maxErrorMessageBytes bounds a CLI-derived message carried on an *Error, so a
// model-sized result or a large stderr dump does not bloat a single log line.
const maxErrorMessageBytes = 500

// clipMessage strips terminal escape sequences and control characters, trims
// surrounding whitespace, and caps a CLI-derived message on a UTF-8 rune
// boundary.
func clipMessage(s string) string {
	return clipMessageTo(s, maxErrorMessageBytes)
}

// sanitizeWindowFactor is how many budgets of raw input clipMessageTo and
// clipTailTo sanitize before clipping.
const sanitizeWindowFactor = 4

// clipMessageTo is clipMessage with an explicit byte budget, used when a
// diagnostic is appended to existing text.
func clipMessageTo(s string, limit int) string {
	if limit <= 0 {
		return ""
	}
	// Bound the sanitizer's work: escape stripping only shrinks text, so a
	// generous multiple of the budget still fills it for ordinary output.
	if w := limit * sanitizeWindowFactor; len(s) > w {
		end := w
		for end > 0 && !utf8.RuneStart(s[end]) {
			end--
		}
		s = s[:end]
	}
	s = sanitizeTerminal(s)
	s = strings.TrimSpace(s)
	if len(s) <= limit {
		return s
	}
	end := limit
	for end > 0 && !utf8.RuneStart(s[end]) {
		end--
	}
	return s[:end]
}

// clipTailTo is clipMessageTo keeping the end of the text: stderr from agent
// CLIs puts warnings first and the fatal line last, so the tail is the evidence.
func clipTailTo(s string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if w := limit * sanitizeWindowFactor; len(s) > w {
		start := len(s) - w
		for start < len(s) && !utf8.RuneStart(s[start]) {
			start++
		}
		s = s[start:]
	}
	s = strings.TrimSpace(sanitizeTerminal(s))
	if len(s) <= limit {
		return s
	}
	start := len(s) - limit
	for start < len(s) && !utf8.RuneStart(s[start]) {
		start++
	}
	return strings.TrimSpace(s[start:])
}

// sanitizeTerminal removes ECMA-48 escape sequences and control characters from
// CLI/provider diagnostics before they reach a log line or terminal. It covers
// CSI (including private parameter bytes), OSC/DCS/SOS/PM/APC through their
// BEL/ST terminator, other ESC sequences, and C0/C1 controls. Tabs, newlines and
// other line separators become a space so untrusted text cannot forge a log
// record; Unicode bidi/format controls are dropped.
func sanitizeTerminal(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		if s[i] != 0x1b {
			r, size := utf8.DecodeRuneInString(s[i:])
			if r == utf8.RuneError && size == 1 {
				i++
				continue
			}
			switch {
			case r == '\t' || r == '\n' || r == '\r' || r == 0x2028 || r == 0x2029:
				b.WriteByte(' ')
			case (unicode.Is(unicode.Cf, r) && r != 0x200c && r != 0x200d) || unicode.Is(unicode.Bidi_Control, r) || (r >= 0xE0000 && r <= 0xE007F):
				// Bidi/format controls can visually reorder a log line; the tag
				// block can hide an invisible payload. ZWNJ/ZWJ are kept: they
				// carry meaning in several scripts and emoji sequences.
			case r >= 0x20 && r != 0x7f && !(r >= 0x80 && r <= 0x9f):
				b.WriteString(s[i : i+size])
			}
			i += size
			continue
		}
		i++ // ESC
		if i >= len(s) {
			break
		}
		switch s[i] {
		case '[': // CSI: parameter bytes, intermediate bytes, final byte.
			i++
			for i < len(s) && s[i] >= 0x30 && s[i] <= 0x3f {
				i++
			}
			for i < len(s) && s[i] >= 0x20 && s[i] <= 0x2f {
				i++
			}
			if i < len(s) && s[i] >= 0x40 && s[i] <= 0x7e {
				i++
			}
		case ']', 'P', 'X', '^', '_': // OSC/DCS/SOS/PM/APC: through BEL or ST.
			i++
			for i < len(s) {
				if s[i] == 0x07 {
					i++
					break
				}
				if s[i] == '\n' {
					// Unterminated: stop so a malformed sequence cannot erase
					// the following lines (the newline is handled normally).
					break
				}
				if s[i] == 0x1b && i+1 < len(s) && s[i+1] == '\\' {
					i += 2
					break
				}
				i++
			}
		default: // ESC + optional intermediates + final byte.
			for i < len(s) && s[i] >= 0x20 && s[i] <= 0x2f {
				i++
			}
			if i < len(s) && s[i] >= 0x30 && s[i] <= 0x7e {
				i++
			}
		}
	}
	return b.String()
}

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
	return runCLI(ctx, name, args, stdin, nil, "")
}

// commandRunnerFor returns a CommandRunner whose child inherits env and runs in
// dir. A nil env inherits the parent environment; a non-nil empty env fails
// closed with no variables.
func commandRunnerFor(env []string, dir string) core.CommandRunner {
	return func(ctx context.Context, name string, args []string, stdin []byte) ([]byte, []byte, int, error) {
		return runCLI(ctx, name, args, stdin, env, dir)
	}
}

func runCLI(
	ctx context.Context,
	name string,
	args []string,
	stdin []byte,
	env []string,
	dir string,
) ([]byte, []byte, int, error) {
	path, err := exec.LookPath(name)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("%w: %s", core.ErrExecutableNotFound, name)
	}

	command := exec.CommandContext(ctx, path, args...)
	configureCommand(command)
	command.WaitDelay = cliCommandWaitDelay
	command.Dir = dir
	command.Env = env
	command.Stdin = bytes.NewReader(stdin)
	stdout := newBoundedBuffer(maxCLIOutputBytes)
	stderr := newBoundedBuffer(maxCLIOutputBytes)
	command.Stdout = stdout
	command.Stderr = stderr

	err = command.Run()
	// Reap any background child the CLI left behind, on success as well as
	// failure: a provider CLI that daemonizes a helper must not outlive the call.
	// When the context triggered Cancel, the group was already killed before Wait
	// returned (while the PID was still valid), so repeating after Wait would only
	// widen the PID-reuse race window for no benefit.
	if ctx.Err() == nil {
		reapCommandGroup(command)
	}
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

// executeCLI runs a batch CLI invocation and normalizes launch and exit
// failures. It returns stdout and stderr even on a nonzero exit, so a caller
// can surface a diagnostic the CLI printed instead of the generic process
// error. stderr is returned on success too: an agent CLI can exit 0 after
// auto-rejecting a tool permission and leave the only explanation on stderr.
func executeCLI(
	ctx context.Context,
	provider core.Provider,
	name string,
	args []string,
	stdin []byte,
	runner core.CommandRunner,
) ([]byte, []byte, error) {
	stdout, stderr, exitCode, err := runner(ctx, name, args, stdin)
	if err != nil {
		return nil, stderr, &core.Error{
			Provider:  provider,
			Flavor:    core.FlavorCLI,
			Operation: "generate",
			Message:   clipMessage(string(stderr)),
			Err:       err,
		}
	}
	if exitCode != 0 {
		message := clipMessage(string(stderr))
		if message == "" {
			message = "CLI command failed"
		}
		return stdout, stderr, &core.Error{
			Provider:  provider,
			Flavor:    core.FlavorCLI,
			Operation: "generate",
			Status:    exitCode,
			Message:   message,
		}
	}
	return stdout, stderr, nil
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
// complements core.CommandRunner, which buffers a whole batch invocation.
type streamingCommandRunner func(context.Context, string, []string, []byte) (cliProcess, error)

func defaultStreamingCommandRunner(
	ctx context.Context,
	name string,
	args []string,
	stdin []byte,
) (cliProcess, error) {
	return runCLIStream(ctx, name, args, stdin, nil, "")
}

// streamingRunnerFor mirrors commandRunnerFor for the streaming path.
func streamingRunnerFor(env []string, dir string) streamingCommandRunner {
	return func(ctx context.Context, name string, args []string, stdin []byte) (cliProcess, error) {
		return runCLIStream(ctx, name, args, stdin, env, dir)
	}
}

func runCLIStream(
	ctx context.Context,
	name string,
	args []string,
	stdin []byte,
	env []string,
	dir string,
) (cliProcess, error) {
	path, err := exec.LookPath(name)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", core.ErrExecutableNotFound, name)
	}

	// The derived context lets close terminate the process tree through the same
	// platform-specific cancel hook that context cancellation uses.
	runCtx, cancel := context.WithCancel(ctx)
	command := exec.CommandContext(runCtx, path, args...)
	configureCommand(command)
	command.WaitDelay = cliCommandWaitDelay
	command.Dir = dir
	command.Env = env
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
	p.waitOnce.Do(func() {
		p.waitErr = p.command.Wait()
		// Only reap when Cancel has not already killed the group—see runCLI.
		if p.parent.Err() == nil {
			reapCommandGroup(p.command)
		}
	})

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
	response() (*core.Response, error)
}

// cliChunkStream drives a streaming CLI subprocess: it scans stdout lines into
// chunks and turns exit status into the same normalized errors executeCLI
// produces for the batch path.
type cliChunkStream struct {
	provider core.Provider
	process  cliProcess
	scanner  *bufio.Scanner
	parser   cliEventParser
	failure  error
	// stderr is cached by exit() so Finish can attach a clean-exit diagnostic
	// without waiting on the process a second time.
	stderr []byte
	// exited records that exit() observed the process end. Finish only accepts
	// a text-only turn when this is true; that is the half of "completed process
	// plus text" the parser cannot see.
	exited bool
}

func newCLIChunkStream(provider core.Provider, process cliProcess, parser cliEventParser) *cliChunkStream {
	scanner := bufio.NewScanner(process.stdout())
	scanner.Buffer(make([]byte, 64*1024), maxCLIOutputBytes)
	return &cliChunkStream{
		provider: provider,
		process:  process,
		scanner:  scanner,
		parser:   parser,
	}
}

func (s *cliChunkStream) Recv() (core.Chunk, error) {
	if s.failure != nil {
		return core.Chunk{}, s.failure
	}
	for s.scanner.Scan() {
		line := bytes.TrimSpace(s.scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		text, err := s.parser.event(line)
		if err != nil {
			return core.Chunk{}, s.fail(err)
		}
		if text != "" {
			return core.Chunk{Text: text}, nil
		}
	}
	if err := s.scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			err = errCLIOutputTooLarge
		}
		return core.Chunk{}, s.fail(s.streamError(err, ""))
	}
	if err := s.exit(); err != nil {
		s.failure = err
		return core.Chunk{}, err
	}
	return core.Chunk{}, core.ErrStreamDone
}

func (s *cliChunkStream) Finish() (*core.Response, error) {
	if s.failure != nil {
		return nil, s.failure
	}
	if !s.exited {
		failure := s.streamError(nil, "stream finished before the CLI exited")
		failure.Code = "incomplete_stream"
		return nil, failure
	}
	response, err := s.parser.response()
	return response, attachCLIStderr(err, s.stderr)
}

// attachCLIStderr adds a non-empty stderr diagnostic to a CLI result error. A
// headless CLI can exit 0 after a result error (an auto-rejected tool, say),
// and the explanation is only on stderr.
func attachCLIStderr(err error, stderr []byte) error {
	if err == nil {
		return nil
	}
	var contextual *core.Error
	if !errors.As(err, &contextual) {
		return err
	}
	const separator = ": "
	detail := clipTailTo(string(stderr), maxErrorMessageBytes-len(separator))
	if detail == "" {
		return err
	}
	if contextual.Message == "" {
		contextual.Message = detail
		return err
	}
	// Reserve room for the stderr tail: it is the evidence the caller needs, so
	// truncating the appended part would defeat the purpose.
	room := maxErrorMessageBytes - len(detail) - len(separator)
	if room <= 0 {
		// The diagnostic fills the budget; do not emit a bare separator.
		contextual.Message = detail
		return err
	}
	contextual.Message = clipMessageTo(contextual.Message, room) + separator + detail
	return err
}

func (s *cliChunkStream) Close() error {
	return s.process.close()
}

// fail records a terminal parse failure and, for parity with the batch path,
// stamps the process exit code onto it when the CLI also exited nonzero.
// Remaining output is drained first: waiting on a process whose stdout is not
// being read would deadlock. Both the byte drain and the timer bound the wait:
// a CLI that keeps writing hits the byte limit, and a CLI that emits its final
// line then idles without exiting (no stdout EOF) hits cliFailDrainTimeout. On
// either bound the exit code is not worth waiting for, and close terminates the
// process (its cancel also unblocks the drain goroutine).
func (s *cliChunkStream) fail(err error) error {
	s.failure = err

	done := make(chan int, 1)
	go func() {
		drained, _ := io.Copy(io.Discard, io.LimitReader(s.process.stdout(), maxCLIOutputBytes))
		if drained >= maxCLIOutputBytes {
			done <- 0
			return
		}
		exitCode, _, waitErr := s.process.wait()
		if waitErr != nil {
			exitCode = 0
		}
		done <- exitCode
	}()

	timer := time.NewTimer(cliFailDrainTimeout)
	defer timer.Stop()
	select {
	case <-timer.C:
		return err
	case exitCode := <-done:
		if exitCode == 0 {
			return err
		}
		var contextual *core.Error
		if errors.As(err, &contextual) && contextual.Status == 0 {
			contextual.Status = exitCode
		}
		return err
	}
}

func (s *cliChunkStream) exit() error {
	exitCode, stderr, err := s.process.wait()
	s.stderr = stderr
	s.exited = true
	if err != nil {
		return s.streamError(err, clipMessage(string(stderr)))
	}
	if exitCode != 0 {
		message := clipMessage(string(stderr))
		if message == "" {
			message = "CLI command failed"
		}
		failure := s.streamError(nil, message)
		failure.Status = exitCode
		return failure
	}
	return nil
}

func (s *cliChunkStream) streamError(err error, message string) *core.Error {
	return &core.Error{
		Provider:  s.provider,
		Flavor:    core.FlavorCLI,
		Operation: "generate_stream",
		Message:   message,
		Err:       err,
	}
}

// startCLIStream launches a streaming CLI and normalizes launch failures.
func startCLIStream(
	ctx context.Context,
	provider core.Provider,
	name string,
	args []string,
	stdin []byte,
	runner streamingCommandRunner,
) (cliProcess, error) {
	process, err := runner(ctx, name, args, stdin)
	if err != nil {
		return nil, &core.Error{
			Provider:  provider,
			Flavor:    core.FlavorCLI,
			Operation: "generate_stream",
			Err:       err,
		}
	}
	return process, nil
}

func renderTranscript(messages []core.Message) string {
	if len(messages) == 1 && messages[0].Role == core.RoleUser {
		return messages[0].Text
	}

	var transcript strings.Builder
	for i, message := range messages {
		if i > 0 {
			transcript.WriteString("\n\n")
		}
		if message.Role == core.RoleAssistant {
			transcript.WriteString("Assistant: ")
		} else {
			transcript.WriteString("User: ")
		}
		transcript.WriteString(message.Text)
	}
	return transcript.String()
}
