package backends

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/lao/llm-driver/internal/core"
)

// testSchema and testJSON are the shared structured-output fixtures: the JSON
// Schema a request sends and the JSON document the model returns for it.
var testSchema = json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)

const testJSON = `{"answer":"42"}`

// streamClient injects a fake streaming runner into a CLI backend without a
// public seam, mirroring what the root Client does for the batch path. It lets
// the CLI streaming tests drive the backend's ChunkStream through the same
// accumulate-and-stamp surface the root Stream exposes.
type streamClient struct {
	provider core.Provider
	flavor   core.Flavor
	model    string
	backend  core.Backend
}

func newCLIStreamClient(config core.Config, runner streamingCommandRunner) *streamClient {
	cliConfig := cliBackendConfig{
		acceptUnverified: config.AcceptUnverifiedOpenCode,
		model:            config.Model,
		path:             config.CLIPath,
		args:             config.CLIArgs,
		streamRunner:     runner,
	}
	var target core.Backend
	switch config.Provider {
	case core.ProviderClaude:
		target = newClaudeCLIBackend(cliConfig)
	case core.ProviderOpenAI:
		target = newCodexCLIBackend(cliConfig)
	default:
		target = newOpenCodeCLIBackend(cliConfig)
	}
	return &streamClient{
		provider: config.Provider,
		flavor:   core.FlavorCLI,
		model:    config.Model,
		backend:  target,
	}
}

func (c *streamClient) GenerateStream(ctx context.Context, request core.Request) (*testStream, error) {
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
	return &testStream{
		provider: c.provider,
		flavor:   c.flavor,
		model:    c.model,
		source:   source,
	}, nil
}

func (c *streamClient) streamError(err error, message string) *core.Error {
	return &core.Error{
		Provider:  c.provider,
		Flavor:    c.flavor,
		Operation: "generate_stream",
		Message:   message,
		Err:       err,
	}
}

// testStream reproduces the root Stream's driving surface (Next/Current/Err/
// Response/Close, accumulation and target stamping) so backend streaming tests
// can assert on the same behavior without importing the root package.
type testStream struct {
	provider core.Provider
	flavor   core.Flavor
	model    string
	source   core.ChunkStream

	closed    atomic.Bool
	closeOnce sync.Once
	closeErr  error

	current  core.Chunk
	text     strings.Builder
	err      error
	done     bool
	response *core.Response
}

func (s *testStream) Next() bool {
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
		// Mirror the root Stream: only text deltas build Response.Text; tool-input
		// deltas are surfaced via Current but must not pollute the response text.
		if chunk.Kind == core.KindText {
			s.text.WriteString(chunk.Text)
		}
		return true
	}
}

func (s *testStream) Current() core.Chunk {
	return s.current
}

func (s *testStream) Err() error {
	return s.err
}

func (s *testStream) Response() (*core.Response, error) {
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

func (s *testStream) Close() error {
	s.closed.Store(true)
	s.closeOnce.Do(func() { s.closeErr = s.source.Close() })
	return s.closeErr
}

func (s *testStream) wrap(err error) error {
	var contextual *core.Error
	if errors.As(err, &contextual) {
		return err
	}
	return s.streamError(err, "")
}

func (s *testStream) streamError(err error, message string) *core.Error {
	return &core.Error{
		Provider:  s.provider,
		Flavor:    s.flavor,
		Operation: "generate_stream",
		Message:   message,
		Err:       err,
	}
}

func streamRequest() core.Request {
	return core.Request{MaxTokens: 32, Messages: []core.Message{core.User("Hello")}}
}

// collectStream drains a stream and returns its chunks, terminal error, and
// final response.
func collectStream(t *testing.T, stream *testStream) ([]string, error, *core.Response) {
	t.Helper()

	var chunks []string
	for stream.Next() {
		chunks = append(chunks, stream.Current().Text)
	}
	if err := stream.Err(); err != nil {
		return chunks, err, nil
	}
	response, err := stream.Response()
	return chunks, err, response
}
