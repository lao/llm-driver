package llmwrapper

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
)

type anthropicBackend struct {
	client anthropic.Client
	model  string
}

func newAnthropicBackend(config apiBackendConfig) *anthropicBackend {
	options := []option.RequestOption{}
	if config.apiKey != "" {
		options = append(options, option.WithAPIKey(config.apiKey))
	}
	if config.baseURL != "" {
		options = append(options, option.WithBaseURL(config.baseURL))
	}
	if config.httpClient != nil {
		options = append(options, option.WithHTTPClient(config.httpClient))
	}
	return &anthropicBackend{
		client: anthropic.NewClient(options...),
		model:  config.model,
	}
}

func (b *anthropicBackend) generate(ctx context.Context, request Request) (*Response, error) {
	message, err := b.client.Messages.New(ctx, b.params(request))
	if err != nil {
		return nil, normalizeAnthropicError(err)
	}
	return anthropicResponse(message), nil
}

func (b *anthropicBackend) generateStream(ctx context.Context, request Request) (chunkStream, error) {
	stream := b.client.Messages.NewStreaming(ctx, b.params(request))
	// NewStreaming records transport and non-2xx failures before any event is
	// read, so a failed start is reported by GenerateStream rather than Next.
	if err := stream.Err(); err != nil {
		_ = stream.Close()
		return nil, normalizeAnthropicStreamError(err)
	}
	return &anthropicChunkStream{stream: stream}, nil
}

func (b *anthropicBackend) params(request Request) anthropic.MessageNewParams {
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(b.model),
		MaxTokens: int64(request.MaxTokens),
		Messages:  make([]anthropic.MessageParam, 0, len(request.Messages)),
	}
	if request.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: request.System}}
	}
	for _, message := range request.Messages {
		block := anthropic.NewTextBlock(message.Text)
		if message.Role == RoleAssistant {
			params.Messages = append(params.Messages, anthropic.NewAssistantMessage(block))
		} else {
			params.Messages = append(params.Messages, anthropic.NewUserMessage(block))
		}
	}
	return params
}

// anthropicChunkStream turns Messages SSE events into chunks while the SDK's
// Message.Accumulate helper rebuilds the final message for finish.
type anthropicChunkStream struct {
	stream  *ssestream.Stream[anthropic.MessageStreamEventUnion]
	message anthropic.Message
}

func (s *anthropicChunkStream) recv() (Chunk, error) {
	for s.stream.Next() {
		event := s.stream.Current()
		if err := s.message.Accumulate(event); err != nil {
			return Chunk{}, normalizeAnthropicStreamError(err)
		}
		if event.Type == "content_block_delta" && event.Delta.Type == "text_delta" {
			return Chunk{Text: event.Delta.Text}, nil
		}
	}
	if err := s.stream.Err(); err != nil {
		return Chunk{}, normalizeAnthropicStreamError(err)
	}
	return Chunk{}, errStreamDone
}

func (s *anthropicChunkStream) finish() (*Response, error) {
	return anthropicResponse(&s.message), nil
}

func (s *anthropicChunkStream) close() error {
	return s.stream.Close()
}

func anthropicResponse(message *anthropic.Message) *Response {
	var text strings.Builder
	for _, block := range message.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}
	return &Response{
		ID:               message.ID,
		Model:            string(message.Model),
		Text:             text.String(),
		CompletionReason: normalizeAnthropicCompletionReason(message.StopReason),
		Usage: Usage{
			InputTokens:              int(message.Usage.InputTokens),
			OutputTokens:             int(message.Usage.OutputTokens),
			CachedInputTokens:        int(message.Usage.CacheReadInputTokens),
			CacheCreationInputTokens: int(message.Usage.CacheCreationInputTokens),
		},
	}
}

func normalizeAnthropicError(err error) error {
	return normalizeAnthropicOperationError(err, "generate")
}

func normalizeAnthropicStreamError(err error) error {
	return normalizeAnthropicOperationError(err, "generate_stream")
}

func normalizeAnthropicOperationError(err error, operation string) error {
	normalized := &Error{
		Provider:  ProviderClaude,
		Flavor:    FlavorAPI,
		Operation: operation,
		Err:       err,
	}
	var apiError *anthropic.Error
	if errors.As(err, &apiError) {
		normalized.Status = apiError.StatusCode
		normalized.Code = string(apiError.Type())
		var payload struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal([]byte(apiError.RawJSON()), &payload) == nil {
			normalized.Message = payload.Error.Message
		}
	}
	return normalized
}

func normalizeAnthropicCompletionReason(reason anthropic.StopReason) CompletionReason {
	switch reason {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence:
		return CompletionReasonStop
	case anthropic.StopReasonMaxTokens:
		return CompletionReasonMaxTokens
	case anthropic.StopReasonRefusal:
		return CompletionReasonRefusal
	default:
		return ""
	}
}
