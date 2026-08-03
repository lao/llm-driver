package llmwrapper

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/lao/llmwrapper/internal/core"
)

// Stream iterates response text chunks and terminates in a normalized
// Response. It is not safe for concurrent use except that Close may be called
// while another goroutine is inside Next.
type Stream struct {
	provider Provider
	flavor   Flavor
	model    string
	source   core.ChunkStream

	closed    atomic.Bool
	closeOnce sync.Once
	closeErr  error

	current  Chunk
	text     strings.Builder
	err      error
	done     bool
	response *Response
}

// GenerateStream validates the request and starts one streaming generation
// through the configured target. The caller must Close the returned stream.
func (c *Client) GenerateStream(ctx context.Context, request Request) (*Stream, error) {
	if err := core.ValidateRequest(request); err != nil {
		return nil, c.streamError(err, "")
	}

	source, err := c.backend.GenerateStream(ctx, request)
	if err != nil {
		return nil, err
	}
	if source == nil {
		return nil, c.streamError(nil, "backend returned a nil stream")
	}
	return &Stream{
		provider: c.provider,
		flavor:   c.flavor,
		model:    c.model,
		source:   source,
	}, nil
}

func (c *Client) streamError(err error, message string) *Error {
	return &Error{
		Provider:  c.provider,
		Flavor:    c.flavor,
		Operation: "generate_stream",
		Message:   message,
		Err:       err,
	}
}

// Next advances to the next chunk. It reports false at the end of the stream,
// after Close, and on error; use Err to tell those cases apart.
func (s *Stream) Next() bool {
	if s.err != nil || s.done || s.closed.Load() {
		return false
	}
	for {
		chunk, err := s.source.Recv()
		if s.closed.Load() {
			return false
		}
		switch {
		case errors.Is(err, core.ErrStreamDone):
			s.done = true
			return false
		case err != nil:
			s.err = s.wrap(err)
			return false
		case chunk.Text == "":
			continue
		}
		s.current = chunk
		s.text.WriteString(chunk.Text)
		return true
	}
}

// Current returns the chunk reached by the most recent Next that reported true.
func (s *Stream) Current() Chunk {
	return s.current
}

// Err returns the terminal stream error, or nil when the stream ended normally.
func (s *Stream) Err() error {
	return s.err
}

// Response returns the final normalized response once the stream is exhausted.
// Its Text is the concatenation of every yielded chunk. Calling it before
// exhaustion, or after a stream error, returns a normalized *Error.
func (s *Stream) Response() (*Response, error) {
	if s.err != nil {
		return nil, s.err
	}
	if s.response != nil {
		return s.response, nil
	}
	if !s.done {
		return nil, s.streamError(nil, "stream is not exhausted")
	}

	response, err := s.source.Finish()
	if err != nil {
		s.err = s.wrap(err)
		return nil, s.err
	}
	if response == nil {
		s.err = s.streamError(nil, "backend returned a nil response")
		return nil, s.err
	}

	response.Text = s.text.String()
	response.Provider = s.provider
	response.Flavor = s.flavor
	if response.Model == "" {
		response.Model = s.model
	}
	s.response = response
	return response, nil
}

// Close releases the stream's transport or subprocess. It is idempotent and
// may be called while another goroutine is blocked in Next.
func (s *Stream) Close() error {
	s.closed.Store(true)
	s.closeOnce.Do(func() { s.closeErr = s.source.Close() })
	return s.closeErr
}

func (s *Stream) wrap(err error) error {
	var contextual *Error
	if errors.As(err, &contextual) {
		return err
	}
	return s.streamError(err, "")
}

func (s *Stream) streamError(err error, message string) *Error {
	return &Error{
		Provider:  s.provider,
		Flavor:    s.flavor,
		Operation: "generate_stream",
		Message:   message,
		Err:       err,
	}
}
