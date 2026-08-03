package core

import (
	"context"
	"errors"
)

// Backend executes one provider-neutral generation request against a
// concrete provider and flavor.
type Backend interface {
	Generate(context.Context, Request) (*Response, error)
	GenerateStream(context.Context, Request) (ChunkStream, error)
}

// ErrStreamDone is the sentinel a ChunkStream returns when no chunk remains.
// It never reaches callers of the public API.
var ErrStreamDone = errors.New("llmdriver: stream done")

// ChunkStream is the per-adapter half of streaming: it maps provider events to
// chunks and to a final Response. Its methods are exported because backend
// implementations live in a separate package from the Stream that drives them.
// Text accumulation, empty-chunk suppression, and target stamping happen once
// in the root Stream.
type ChunkStream interface {
	// Recv returns the next chunk, or ErrStreamDone once the provider stream
	// ended normally.
	Recv() (Chunk, error)
	// Finish returns the final response, valid only after Recv reported
	// ErrStreamDone. Text is filled in by the Stream.
	Finish() (*Response, error)
	// Close releases the underlying transport or subprocess.
	Close() error
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
