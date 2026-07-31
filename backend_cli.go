package llmwrapper

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
)

type commandRunner func(context.Context, string, []string, []byte) ([]byte, []byte, int, error)

func defaultCommandRunner(ctx context.Context, name string, args []string, stdin []byte) ([]byte, []byte, int, error) {
	path, err := exec.LookPath(name)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("%w: %s", ErrExecutableNotFound, name)
	}

	command := exec.CommandContext(ctx, path, args...)
	command.Stdin = bytes.NewReader(stdin)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr

	err = command.Run()
	if err == nil {
		return stdout.Bytes(), stderr.Bytes(), 0, nil
	}

	var exitError *exec.ExitError
	if errors.As(err, &exitError) {
		return stdout.Bytes(), stderr.Bytes(), exitError.ExitCode(), nil
	}
	return stdout.Bytes(), stderr.Bytes(), 0, err
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
