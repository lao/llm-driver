package backends

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"

	"github.com/lao/llm-driver/internal/core"
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

func (b *anthropicBackend) Generate(ctx context.Context, request core.Request) (*core.Response, error) {
	message, err := b.client.Messages.New(ctx, b.params(request))
	if err != nil {
		return nil, normalizeAnthropicError(err)
	}
	return anthropicResponse(message), nil
}

func (b *anthropicBackend) GenerateStream(ctx context.Context, request core.Request) (core.ChunkStream, error) {
	stream := b.client.Messages.NewStreaming(ctx, b.params(request))
	// NewStreaming records transport and non-2xx failures before any event is
	// read, so a failed start is reported by GenerateStream rather than Next.
	if err := stream.Err(); err != nil {
		_ = stream.Close()
		return nil, normalizeAnthropicStreamError(err)
	}
	return &anthropicChunkStream{stream: stream}, nil
}

func (b *anthropicBackend) params(request core.Request) anthropic.MessageNewParams {
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
		if message.Role == core.RoleAssistant {
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
	stopped bool
}

func (s *anthropicChunkStream) Recv() (core.Chunk, error) {
	for s.stream.Next() {
		event := s.stream.Current()
		if err := s.message.Accumulate(event); err != nil {
			return core.Chunk{}, normalizeAnthropicStreamError(err)
		}
		if event.Type == "message_stop" {
			s.stopped = true
		}
		if event.Type == "content_block_delta" && event.Delta.Type == "text_delta" {
			return core.Chunk{Text: event.Delta.Text}, nil
		}
	}
	if err := s.stream.Err(); err != nil {
		return core.Chunk{}, normalizeAnthropicStreamError(err)
	}
	return core.Chunk{}, core.ErrStreamDone
}

// Finish requires the terminal message_stop event: a body that ends early is a
// clean EOF to the SDK, so without this check a truncated stream would be
// reported as a complete response with partial text and stale usage.
func (s *anthropicChunkStream) Finish() (*core.Response, error) {
	if !s.stopped {
		return nil, &core.Error{
			Provider:  core.ProviderClaude,
			Flavor:    core.FlavorAPI,
			Operation: "generate_stream",
			Code:      "missing_message_completion",
			Message:   "Anthropic stream ended without a completed message",
		}
	}
	return anthropicResponse(&s.message), nil
}

func (s *anthropicChunkStream) Close() error {
	return s.stream.Close()
}

func anthropicResponse(message *anthropic.Message) *core.Response {
	var text strings.Builder
	for _, block := range message.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}
	return &core.Response{
		ID:               message.ID,
		Model:            string(message.Model),
		Text:             text.String(),
		CompletionReason: normalizeAnthropicCompletionReason(message.StopReason),
		Usage: core.Usage{
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
	normalized := &core.Error{
		Provider:  core.ProviderClaude,
		Flavor:    core.FlavorAPI,
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

func normalizeAnthropicCompletionReason(reason anthropic.StopReason) core.CompletionReason {
	switch reason {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence:
		return core.CompletionReasonStop
	case anthropic.StopReasonMaxTokens:
		return core.CompletionReasonMaxTokens
	case anthropic.StopReasonRefusal:
		return core.CompletionReasonRefusal
	default:
		return ""
	}
}
