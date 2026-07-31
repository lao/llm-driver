package claudewrap

import (
	"errors"
	"fmt"
)

// Sentinel errors returned for configuration and environment problems.
var (
	// ErrInvalidConfig identifies invalid llmwrapper client configuration.
	ErrInvalidConfig = errors.New("llmwrapper: invalid config")
	// ErrNoBackend is returned by New when no backend option (WithSDK/WithCLI) is given.
	ErrNoBackend = errors.New("claudewrap: no backend configured")
	// ErrCLINotFound is returned by the cli backend when the claude binary is not on PATH.
	ErrCLINotFound = errors.New("claudewrap: claude CLI not found on PATH")
)

// Error is the unified error type both backends wrap failures in.
type Error struct {
	Backend string // "sdk" | "cli"
	Op      string // e.g. "messages.new"
	Status  int    // HTTP status (sdk) or process exit code (cli); 0 if not applicable
	Type    string // API error type (sdk) when available
	Message string // human-readable summary
	Err     error  // wrapped cause, if any
}

func (e *Error) Error() string {
	s := fmt.Sprintf("claudewrap: %s %s", e.Backend, e.Op)
	if e.Status != 0 {
		s += fmt.Sprintf(" (status %d)", e.Status)
	}
	if e.Type != "" {
		s += fmt.Sprintf(" [%s]", e.Type)
	}
	if e.Message != "" {
		s += ": " + e.Message
	}
	if e.Err != nil {
		s += ": " + e.Err.Error()
	}
	return s
}

// Unwrap exposes the wrapped cause for errors.Is / errors.As.
func (e *Error) Unwrap() error { return e.Err }
