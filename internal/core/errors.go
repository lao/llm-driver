package core

import (
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrInvalidConfig identifies client configuration rejected by New.
	ErrInvalidConfig = errors.New("llmdriver: invalid config")
	// ErrInvalidRequest identifies a request rejected before provider invocation.
	ErrInvalidRequest = errors.New("llmdriver: invalid request")
	// ErrExecutableNotFound identifies a missing provider CLI executable.
	ErrExecutableNotFound = errors.New("llmdriver: executable not found")
	// ErrUnsupportedFeature identifies a request feature the selected target
	// cannot honor. The wrapping message names the feature and the target.
	ErrUnsupportedFeature = errors.New("llmdriver: unsupported feature")
)

// CodeMissingFinalMessage identifies a completed Codex CLI turn that emitted
// no completed agent message.
const CodeMissingFinalMessage = "missing_final_message"

// CodeToolCalls identifies a CLI turn that ended on a tool call, with or
// without partial text, as when an agent CLI auto-rejects a tool permission.
const CodeToolCalls = "tool_calls"

// Error adds target and operation context to configuration and runtime failures.
// For CLI targets, process-exit errors use a nonzero Status with empty Reason
// and Code; API/result errors populate at least one of those fields.
type Error struct {
	// Provider identifies the failing provider, when known.
	Provider Provider
	// Flavor identifies the failing transport flavor, when known.
	Flavor Flavor
	// Operation identifies the library operation that failed.
	Operation string
	// Status is an HTTP status or process exit code, when available.
	Status int
	// Reason is a provider or CLI terminal reason, when available (for the
	// Claude CLI, the result envelope's terminal_reason).
	Reason string
	// Code is the provider or CLI failure code, when available.
	Code string
	// SessionID identifies a CLI session when available. It does not prove the
	// session completed; callers must verify its persisted state independently.
	SessionID string
	// PartialText is untrusted output received before a failed CLI turn. It is
	// omitted from Error() and must not be treated as a completed response.
	PartialText string
	// Message is a concise upstream or normalized failure message. CLI errors
	// may include sanitized, bounded stderr even after exit 0. Sensitive content
	// (including prompts or credentials) is not redacted; filter before logging.
	Message string
	// Err preserves the underlying cause for errors.Is and errors.As.
	Err error
}

// Error returns a concise target-aware description.
func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}

	var b strings.Builder
	b.WriteString("llmdriver")
	if target := formatTarget(e.Provider, e.Flavor); target != "" {
		b.WriteString(": ")
		b.WriteString(target)
	}
	if e.Operation != "" {
		b.WriteString(": ")
		b.WriteString(e.Operation)
	}
	if e.Status != 0 {
		fmt.Fprintf(&b, " (status %d)", e.Status)
	}
	if e.Code != "" {
		b.WriteString(" [")
		b.WriteString(e.Code)
		b.WriteByte(']')
	}

	detail := e.Message
	if detail == "" && e.Err != nil {
		detail = e.Err.Error()
	}
	if detail != "" {
		b.WriteString(": ")
		b.WriteString(detail)
	}
	return b.String()
}

// Unwrap exposes the underlying cause for errors.Is and errors.As.
func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func formatTarget(provider Provider, flavor Flavor) string {
	switch {
	case provider != "" && flavor != "":
		return string(provider) + "/" + string(flavor)
	case provider != "":
		return string(provider)
	case flavor != "":
		return string(flavor)
	default:
		return ""
	}
}
