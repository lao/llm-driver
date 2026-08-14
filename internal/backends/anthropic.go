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
	response := anthropicResponse(message)
	if err := setStructuredOutput(response, request.OutputSchema, core.ProviderClaude, core.FlavorAPI, "generate"); err != nil {
		return nil, err
	}
	return response, nil
}

func (b *anthropicBackend) GenerateStream(ctx context.Context, request core.Request) (core.ChunkStream, error) {
	stream := b.client.Messages.NewStreaming(ctx, b.params(request))
	// NewStreaming records transport and non-2xx failures before any event is
	// read, so a failed start is reported by GenerateStream rather than Next.
	if err := stream.Err(); err != nil {
		_ = stream.Close()
		return nil, normalizeAnthropicStreamError(err)
	}
	return &anthropicChunkStream{stream: stream, schema: request.OutputSchema}, nil
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
	if request.OutputSchema != nil {
		// Validated in core, so SchemaObject cannot fail here.
		schema, _ := request.OutputSchema.SchemaObject()
		params.OutputConfig.Format = anthropic.JSONOutputFormatParam{Schema: schema}
	}
	if request.Effort != "" {
		// Core validated the value against the target's allowed set.
		params.OutputConfig.Effort = anthropic.OutputConfigEffort(request.Effort)
	}
	// Sampling params are passed through; the provider enforces its own bounds
	// and surfaces rejections (e.g. newest models reject temperature) as the
	// normalized API error.
	if request.Temperature != nil {
		params.Temperature = anthropic.Float(*request.Temperature)
	}
	if request.TopP != nil {
		params.TopP = anthropic.Float(*request.TopP)
	}
	if request.TopK != nil {
		params.TopK = anthropic.Int(int64(*request.TopK))
	}
	if len(request.StopSequences) > 0 {
		params.StopSequences = request.StopSequences
	}
	if request.UserID != "" {
		params.Metadata = anthropic.MetadataParam{UserID: anthropic.String(request.UserID)}
	}
	return params
}

// anthropicChunkStream turns Messages SSE events into chunks while the SDK's
// Message.Accumulate helper rebuilds the final message for finish.
type anthropicChunkStream struct {
	stream  *ssestream.Stream[anthropic.MessageStreamEventUnion]
	message anthropic.Message
	schema  *core.OutputSchema
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
	response := anthropicResponse(&s.message)
	// The accumulated message text equals the concatenated deltas the root
	// Stream reports, so validating it here populates StructuredOutput from the
	// same bytes the caller received.
	if err := setStructuredOutput(response, s.schema, core.ProviderClaude, core.FlavorAPI, "generate_stream"); err != nil {
		return nil, err
	}
	return response, nil
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
	case anthropic.StopReasonToolUse:
		return core.CompletionReasonToolUse
	default:
		return ""
	}
}
