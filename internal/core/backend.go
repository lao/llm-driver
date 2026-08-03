package core

import "context"

// Backend executes one provider-neutral generation request against a
// concrete provider and flavor.
type Backend interface {
	Generate(context.Context, Request) (*Response, error)
}

// CommandRunner executes a CLI command and returns stdout, stderr, the exit
// code, and any execution error.
type CommandRunner func(context.Context, string, []string, []byte) ([]byte, []byte, int, error)

// ValidateConfig reports whether the config selects a supported target.
func ValidateConfig(c Config) error {
	return c.validate()
}

// ValidateRequest reports whether the request is sendable to any target.
func ValidateRequest(r Request) error {
	return r.validate()
}
