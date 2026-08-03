package llmwrapper

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// newSSETestServer serves one text/event-stream response built from raw event
// blocks, flushing each so adapters observe them incrementally. Blocks are sent
// verbatim, which lets tests supply malformed frames.
func newSSETestServer(t *testing.T, status int, events []string) (*httptest.Server, *capturedHTTPRequest) {
	t.Helper()

	captured := new(capturedHTTPRequest)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		captured.method = request.Method
		captured.path = request.URL.Path
		captured.header = request.Header.Clone()
		if err := json.Unmarshal(body, &captured.body); err != nil {
			t.Errorf("decode request: %v", err)
		}

		if status != http.StatusOK {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			for _, event := range events {
				_, _ = io.WriteString(w, event)
			}
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, event := range events {
			_, _ = io.WriteString(w, event)
			w.(http.Flusher).Flush()
		}
	}))
	t.Cleanup(server.Close)
	return server, captured
}

// collectStream drains a stream and returns its chunks, terminal error, and
// final response.
func collectStream(t *testing.T, stream *Stream) ([]string, error, *Response) {
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

// scriptedStream replays a fixed chunk/error script so the Stream contract can
// be tested without a provider.
type scriptedStream struct {
	chunks    []string
	err       error
	response  *Response
	finishErr error

	index    int
	closes   int
	closeErr error
	blocked  chan struct{}
}

func (s *scriptedStream) recv() (Chunk, error) {
	if s.index < len(s.chunks) {
		text := s.chunks[s.index]
		s.index++
		return Chunk{Text: text}, nil
	}
	if s.blocked != nil {
		<-s.blocked
	}
	if s.err != nil {
		return Chunk{}, s.err
	}
	return Chunk{}, errStreamDone
}

func (s *scriptedStream) finish() (*Response, error) {
	if s.finishErr != nil {
		return nil, s.finishErr
	}
	return s.response, nil
}

func (s *scriptedStream) close() error {
	s.closes++
	if s.blocked != nil {
		close(s.blocked)
		s.blocked = nil
	}
	return s.closeErr
}

func newScriptedClient(t *testing.T, source chunkStream) *Client {
	t.Helper()

	client, err := newClient(Config{
		Provider: ProviderClaude,
		Flavor:   FlavorAPI,
		Model:    "stream-test",
		APIKey:   "test-key",
	}, nil)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	client.backend = &scriptedBackend{source: source}
	return client
}

type scriptedBackend struct {
	source chunkStream
	err    error
}

func (b *scriptedBackend) generate(context.Context, Request) (*Response, error) {
	return nil, errors.New("not used")
}

func (b *scriptedBackend) generateStream(context.Context, Request) (chunkStream, error) {
	if b.err != nil {
		return nil, b.err
	}
	return b.source, nil
}

// generateStream keeps the batch-path fake in client_test.go satisfying the
// backend seam; streaming tests use scriptedBackend.
func (f *fakeGenerationBackend) generateStream(context.Context, Request) (chunkStream, error) {
	return nil, errors.New("fakeGenerationBackend does not stream")
}

func streamRequest() Request {
	return Request{MaxTokens: 32, Messages: []Message{User("Hello")}}
}

func TestStreamYieldsChunksThenResponse(t *testing.T) {
	t.Parallel()

	source := &scriptedStream{
		chunks:   []string{"Hello, ", "", "world"},
		response: &Response{ID: "resp-1", Usage: Usage{InputTokens: 3, OutputTokens: 5}},
	}
	client := newScriptedClient(t, source)

	stream, err := client.GenerateStream(context.Background(), streamRequest())
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	var got []string
	for stream.Next() {
		got = append(got, stream.Current().Text)
		// Current() stays stable until the next advance.
		if stream.Current().Text != got[len(got)-1] {
			t.Fatalf("Current() is not stable")
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if len(got) != 2 || got[0] != "Hello, " || got[1] != "world" {
		t.Fatalf("chunks = %#v, want the two non-empty chunks", got)
	}

	response, err := stream.Response()
	if err != nil {
		t.Fatalf("Response(): %v", err)
	}
	if response.Text != "Hello, world" {
		t.Fatalf("Response().Text = %q, want the concatenated chunks", response.Text)
	}
	if response.ID != "resp-1" || response.Model != "stream-test" {
		t.Fatalf("response = %+v", response)
	}
	if response.Provider != ProviderClaude || response.Flavor != FlavorAPI {
		t.Fatalf("response target = %s/%s", response.Provider, response.Flavor)
	}
	if response.Usage != (Usage{InputTokens: 3, OutputTokens: 5}) {
		t.Fatalf("usage = %+v", response.Usage)
	}
}

func TestStreamResponseBeforeExhaustion(t *testing.T) {
	t.Parallel()

	client := newScriptedClient(t, &scriptedStream{chunks: []string{"a", "b"}, response: &Response{}})
	stream, err := client.GenerateStream(context.Background(), streamRequest())
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	if !stream.Next() {
		t.Fatal("Next() = false, want the first chunk")
	}
	_, err = stream.Response()
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("Response() error = %T %v, want *Error", err, err)
	}
	if target.Provider != ProviderClaude || target.Flavor != FlavorAPI || target.Operation != "generate_stream" {
		t.Fatalf("error context = %+v", target)
	}
}

func TestStreamMidStreamError(t *testing.T) {
	t.Parallel()

	failure := errors.New("provider exploded")
	client := newScriptedClient(t, &scriptedStream{chunks: []string{"partial"}, err: failure})
	stream, err := client.GenerateStream(context.Background(), streamRequest())
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	var chunks int
	for stream.Next() {
		chunks++
	}
	if chunks != 1 {
		t.Fatalf("chunks = %d, want 1", chunks)
	}
	if !errors.Is(stream.Err(), failure) {
		t.Fatalf("Err() = %v, want %v", stream.Err(), failure)
	}
	var target *Error
	if !errors.As(stream.Err(), &target) || target.Operation != "generate_stream" {
		t.Fatalf("Err() = %#v, want normalized *Error", stream.Err())
	}
	if _, err := stream.Response(); !errors.Is(err, failure) {
		t.Fatalf("Response() error = %v, want %v", err, failure)
	}
}

func TestStreamCloseIsIdempotent(t *testing.T) {
	t.Parallel()

	source := &scriptedStream{chunks: []string{"a"}, response: &Response{}}
	client := newScriptedClient(t, source)
	stream, err := client.GenerateStream(context.Background(), streamRequest())
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}

	if err := stream.Close(); err != nil {
		t.Fatalf("Close(): %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("second Close(): %v", err)
	}
	if source.closes != 1 {
		t.Fatalf("underlying closes = %d, want 1", source.closes)
	}
	if stream.Next() {
		t.Fatal("Next() = true after Close()")
	}
	if _, err := stream.Response(); err == nil {
		t.Fatal("Response() error = nil after Close() without exhaustion")
	}
}

func TestStreamCloseMidStreamStopsIteration(t *testing.T) {
	t.Parallel()

	source := &scriptedStream{chunks: []string{"a", "b"}, blocked: make(chan struct{}), response: &Response{}}
	client := newScriptedClient(t, source)
	stream, err := client.GenerateStream(context.Background(), streamRequest())
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}

	if !stream.Next() {
		t.Fatal("Next() = false, want the first chunk")
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close(): %v", err)
	}
	if stream.Next() {
		t.Fatal("Next() = true after Close()")
	}
	if stream.Err() != nil {
		t.Fatalf("Err() = %v after Close(), want nil", stream.Err())
	}
}

func TestGenerateStreamValidatesRequest(t *testing.T) {
	t.Parallel()

	backend := &scriptedBackend{source: &scriptedStream{}}
	client := newScriptedClient(t, nil)
	client.backend = backend

	_, err := client.GenerateStream(context.Background(), Request{})
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("GenerateStream() error = %v, want ErrInvalidRequest", err)
	}
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("GenerateStream() error = %T %v, want *Error", err, err)
	}
	if target.Provider != ProviderClaude || target.Flavor != FlavorAPI || target.Operation != "generate_stream" {
		t.Fatalf("error context = %+v", target)
	}
}

func TestGenerateStreamPropagatesStartupError(t *testing.T) {
	t.Parallel()

	startup := errors.New("startup failed")
	client := newScriptedClient(t, nil)
	client.backend = &scriptedBackend{err: startup}

	if _, err := client.GenerateStream(context.Background(), streamRequest()); !errors.Is(err, startup) {
		t.Fatalf("GenerateStream() error = %v, want %v", err, startup)
	}
}

func TestStreamFinishError(t *testing.T) {
	t.Parallel()

	failure := errors.New("no final result")
	client := newScriptedClient(t, &scriptedStream{chunks: []string{"a"}, finishErr: failure})
	stream, err := client.GenerateStream(context.Background(), streamRequest())
	if err != nil {
		t.Fatalf("GenerateStream(): %v", err)
	}
	defer stream.Close()

	for stream.Next() {
	}
	if stream.Err() != nil {
		t.Fatalf("Err() = %v, want nil before Response()", stream.Err())
	}
	if _, err := stream.Response(); !errors.Is(err, failure) {
		t.Fatalf("Response() error = %v, want %v", err, failure)
	}
	if !errors.Is(stream.Err(), failure) {
		t.Fatalf("Err() = %v after failed Response()", stream.Err())
	}
}
