package backends

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/lao/llmwrapper/internal/core"
)

const (
	maxCLIOutputBytes   = 16 << 20
	cliCommandWaitDelay = 250 * time.Millisecond
)

var errCLIOutputTooLarge = errors.New("llmwrapper: CLI output exceeds limit")

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
		return nil, nil, 0, fmt.Errorf("%w: %s", core.ErrExecutableNotFound, name)
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
	provider core.Provider,
	name string,
	args []string,
	stdin []byte,
	runner core.CommandRunner,
) ([]byte, error) {
	stdout, stderr, exitCode, err := runner(ctx, name, args, stdin)
	if err != nil {
		return nil, &core.Error{
			Provider:  provider,
			Flavor:    core.FlavorCLI,
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
		return stdout, &core.Error{
			Provider:  provider,
			Flavor:    core.FlavorCLI,
			Operation: "generate",
			Status:    exitCode,
			Message:   message,
		}
	}
	return stdout, nil
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
