package llmwrapper

import (
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrInvalidConfig identifies client configuration rejected by New.
	ErrInvalidConfig = errors.New("llmwrapper: invalid config")
	// ErrInvalidRequest identifies a request rejected before provider invocation.
	ErrInvalidRequest = errors.New("llmwrapper: invalid request")
	// ErrExecutableNotFound identifies a missing provider CLI executable.
	ErrExecutableNotFound = errors.New("llmwrapper: executable not found")
)

// Error adds target and operation context to configuration and runtime failures.
type Error struct {
	// Provider identifies the failing provider, when known.
	Provider Provider
	// Flavor identifies the failing transport flavor, when known.
	Flavor Flavor
	// Operation identifies the library operation that failed.
	Operation string
	// Status is an HTTP status or process exit code, when available.
	Status int
	// Code is the provider or CLI failure code, when available.
	Code string
	// Message is a concise upstream or normalized failure message.
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
	b.WriteString("llmwrapper")
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
