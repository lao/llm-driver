package claudewrap

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os/exec"
	"strings"
)

// commandRunner executes a command with the given stdin and returns its stdout,
// stderr, exit code, and any launch error. Injected for testing.
type commandRunner func(ctx context.Context, name string, args []string, stdin []byte) (stdout, stderr []byte, exitCode int, err error)

// cliBackend drives a local Claude Code CLI via `claude -p`.
type cliBackend struct {
	path         string
	extraArgs    []string
	defaultModel string
	runner       commandRunner
}

func newCLIBackend(c *config) (*cliBackend, error) {
	path := c.cliPath
	if path == "" {
		path = "claude"
	}
	runner := c.cliRunner
	if runner == nil {
		runner = defaultCLIRunner
	}
	return &cliBackend{
		path:         path,
		extraArgs:    c.cliExtraArgs,
		defaultModel: c.defaultModel,
		runner:       runner,
	}, nil
}

func (b *cliBackend) name() string { return "cli" }

// cliResult is the subset of `claude -p --output-format json` output we map.
type cliResult struct {
	Subtype    string   `json:"subtype"`
	IsError    bool     `json:"is_error"`
	Result     string   `json:"result"`
	StopReason string   `json:"stop_reason"`
	SessionID  string   `json:"session_id"`
	Usage      cliUsage `json:"usage"`
}

type cliUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

func (b *cliBackend) messages(ctx context.Context, p MessageParams) (*Response, error) {
	model := p.Model
	if model == "" {
		model = b.defaultModel
	}
	if model == "" {
		model = DefaultModel
	}

	args := []string{"-p", "--output-format", "json", "--model", model}
	if p.System != "" {
		args = append(args, "--append-system-prompt", p.System)
	}
	args = append(args, b.extraArgs...)

	stdout, stderr, exit, err := b.runner(ctx, b.path, args, []byte(flattenPrompt(p.Messages)))
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) || errors.Is(err, ErrCLINotFound) {
			return nil, ErrCLINotFound
		}
		return nil, &Error{Backend: "cli", Op: "messages.new", Status: exit, Message: strings.TrimSpace(string(stderr)), Err: err}
	}
	if exit != 0 {
		return nil, &Error{Backend: "cli", Op: "messages.new", Status: exit, Message: strings.TrimSpace(string(stderr))}
	}

	var res cliResult
	if err := json.Unmarshal(stdout, &res); err != nil {
		return nil, &Error{Backend: "cli", Op: "messages.new", Message: "invalid CLI JSON output", Err: err}
	}
	if res.IsError {
		return nil, &Error{Backend: "cli", Op: "messages.new", Type: res.Subtype, Message: res.Result}
	}

	return &Response{
		ID:         res.SessionID,
		Model:      model,
		Role:       RoleAssistant,
		Content:    []ContentBlock{TextBlock(res.Result)},
		StopReason: StopReason(res.StopReason),
		Backend:    "cli",
		Usage: Usage{
			InputTokens:         res.Usage.InputTokens,
			OutputTokens:        res.Usage.OutputTokens,
			CacheReadTokens:     res.Usage.CacheReadInputTokens,
			CacheCreationTokens: res.Usage.CacheCreationInputTokens,
		},
	}, nil
}

// flattenPrompt renders a transcript into a single prompt string. A lone user
// turn is passed verbatim; otherwise turns are labelled by role. See SPEC §7 for
// the fidelity limitations this implies.
func flattenPrompt(msgs []Message) string {
	if len(msgs) == 1 && msgs[0].Role == RoleUser {
		return messageText(msgs[0])
	}
	var b strings.Builder
	for i, m := range msgs {
		if i > 0 {
			b.WriteString("\n\n")
		}
		label := "User"
		if m.Role == RoleAssistant {
			label = "Assistant"
		}
		b.WriteString(label)
		b.WriteString(": ")
		b.WriteString(messageText(m))
	}
	return b.String()
}

func messageText(m Message) string {
	var b strings.Builder
	for _, blk := range m.Content {
		if blk.Type == BlockText {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// defaultCLIRunner executes the real claude binary. The prompt is delivered on
// stdin — never interpolated into a shell string.
func defaultCLIRunner(ctx context.Context, name string, args []string, stdin []byte) ([]byte, []byte, int, error) {
	if _, err := exec.LookPath(name); err != nil {
		return nil, nil, 0, ErrCLINotFound
	}
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdin = bytes.NewReader(stdin)
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb

	err := cmd.Run()
	exit := 0
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			exit = ee.ExitCode()
			err = nil // non-zero exit is reported via exit code, not err
		}
	}
	return out.Bytes(), errb.Bytes(), exit, err
}
