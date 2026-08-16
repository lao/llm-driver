package backends

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
		params.Messages = append(params.Messages, anthropicMessage(message))
	}
	if len(request.Tools) > 0 {
		params.Tools = anthropicTools(request.Tools)
	}
	if request.ToolChoice.Mode != core.ToolChoiceAuto {
		params.ToolChoice = anthropicToolChoice(request.ToolChoice)
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

// anthropicMessage maps a neutral message to an SDK message. A text-only
// message keeps the exact single-text-block form earlier phases produced; a
// parts message expands to the matching content blocks.
func anthropicMessage(message core.Message) anthropic.MessageParam {
	if len(message.Parts) == 0 {
		block := anthropic.NewTextBlock(message.Text)
		if message.Role == core.RoleAssistant {
			return anthropic.NewAssistantMessage(block)
		}
		return anthropic.NewUserMessage(block)
	}
	blocks := make([]anthropic.ContentBlockParamUnion, 0, len(message.Parts))
	for _, part := range message.Parts {
		switch p := part.(type) {
		case core.TextPart:
			blocks = append(blocks, anthropic.NewTextBlock(p.Text))
		case core.ToolCallPart:
			blocks = append(blocks, anthropic.NewToolUseBlock(p.ID, json.RawMessage(p.Input), p.Name))
		case core.ToolResultPart:
			blocks = append(blocks, anthropic.NewToolResultBlock(p.ToolCallID, p.Content, p.IsError))
		default:
			panic(fmt.Sprintf("anthropic: unhandled part type %T (core validation should reject it)", part))
		}
	}
	if message.Role == core.RoleAssistant {
		return anthropic.NewAssistantMessage(blocks...)
	}
	return anthropic.NewUserMessage(blocks...)
}

// anthropicTools maps neutral tools to the SDK's custom-tool params. The schema
// is validated as a JSON object in core, so the unmarshal cannot fail here.
func anthropicTools(tools []core.Tool) []anthropic.ToolUnionParam {
	mapped := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		var schema anthropic.ToolInputSchemaParam
		_ = json.Unmarshal(tool.InputSchema, &schema)
		toolParam := anthropic.ToolParam{Name: tool.Name, InputSchema: schema}
		if tool.Description != "" {
			toolParam.Description = anthropic.String(tool.Description)
		}
		mapped = append(mapped, anthropic.ToolUnionParam{OfTool: &toolParam})
	}
	return mapped
}

// anthropicToolChoice maps a non-auto neutral choice to the SDK union. Required
// maps to "any" (the model must use some tool).
func anthropicToolChoice(choice core.ToolChoice) anthropic.ToolChoiceUnionParam {
	switch choice.Mode {
	case core.ToolChoiceNone:
		return anthropic.ToolChoiceUnionParam{OfNone: &anthropic.ToolChoiceNoneParam{}}
	case core.ToolChoiceRequired:
		return anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}
	case core.ToolChoiceTool:
		return anthropic.ToolChoiceParamOfTool(choice.Name)
	default:
		return anthropic.ToolChoiceUnionParam{}
	}
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
		if event.Type == "content_block_delta" {
			switch event.Delta.Type {
			case "text_delta":
				return core.Chunk{Text: event.Delta.Text}, nil
			case "input_json_delta":
				// The SDK's Accumulate rebuilds the full tool_use input, so Finish
				// reports the parsed calls; this surfaces the raw JSON as progress.
				return core.Chunk{Kind: core.KindToolInput, Text: event.Delta.PartialJSON}, nil
			}
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
	var toolCalls []core.ToolCall
	for _, block := range message.Content {
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
		case "tool_use":
			toolCalls = append(toolCalls, core.ToolCall{ID: block.ID, Name: block.Name, Input: block.Input})
		}
	}
	return &core.Response{
		ID:               message.ID,
		Model:            string(message.Model),
		Text:             text.String(),
		ToolCalls:        toolCalls,
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
