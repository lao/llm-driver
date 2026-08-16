package backends

import (
	"context"
	"errors"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"

	"github.com/lao/llm-driver/internal/core"
)

type openAIBackend struct {
	client openai.Client
	model  string
}

func newOpenAIBackend(config apiBackendConfig) *openAIBackend {
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
	return &openAIBackend{
		client: openai.NewClient(options...),
		model:  config.model,
	}
}

func (b *openAIBackend) Generate(ctx context.Context, request core.Request) (*core.Response, error) {
	response, err := b.client.Responses.New(ctx, b.params(request))
	if err != nil {
		return nil, normalizeOpenAIError(err, "generate")
	}
	if err := validateOpenAIResponse(response, "generate"); err != nil {
		return nil, err
	}
	result := openAIResponse(response)
	if err := setStructuredOutput(result, request.OutputSchema, core.ProviderOpenAI, core.FlavorAPI, "generate"); err != nil {
		return nil, err
	}
	return result, nil
}

func (b *openAIBackend) GenerateStream(ctx context.Context, request core.Request) (core.ChunkStream, error) {
	stream := b.client.Responses.NewStreaming(ctx, b.params(request))
	// NewStreaming records transport and non-2xx failures before any event is
	// read, so a failed start is reported by GenerateStream rather than Next.
	if err := stream.Err(); err != nil {
		_ = stream.Close()
		return nil, normalizeOpenAIError(err, "generate_stream")
	}
	return &openAIChunkStream{stream: stream, schema: request.OutputSchema}, nil
}

func (b *openAIBackend) params(request core.Request) responses.ResponseNewParams {
	input := make(responses.ResponseInputParam, 0, len(request.Messages))
	for _, message := range request.Messages {
		content := responses.ResponseInputMessageContentListParam{
			responses.ResponseInputContentParamOfInputText(message.Text),
		}
		item := responses.ResponseInputItemParamOfMessage(
			content,
			responses.EasyInputMessageRole(message.Role),
		)
		if message.Role == core.RoleAssistant {
			// Preserve the completed-answer phase so current Codex models keep
			// full follow-up quality on prior assistant turns.
			item.OfMessage.Phase = responses.EasyInputMessagePhaseFinalAnswer
		}
		input = append(input, item)
	}

	params := responses.ResponseNewParams{
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: input,
		},
		Model:           openai.ResponsesModel(b.model),
		MaxOutputTokens: openai.Int(int64(request.MaxTokens)),
	}
	if request.System != "" {
		params.Instructions = openai.String(request.System)
	}
	if request.Effort != "" {
		// Core validated the value against the target's allowed set (low..high).
		params.Reasoning = openai.ReasoningParam{Effort: openai.ReasoningEffort(request.Effort)}
	}
	if request.OutputSchema != nil {
		// Validated in core, so SchemaObject cannot fail here.
		schema, _ := request.OutputSchema.SchemaObject()
		name := request.OutputSchema.Name
		if name == "" {
			name = "response"
		}
		format := responses.ResponseFormatTextConfigParamOfJSONSchema(name, schema)
		format.OfJSONSchema.Strict = openai.Bool(true)
		params.Text = responses.ResponseTextConfigParam{Format: format}
	}
	return params
}

// openAIChunkStream turns Responses SSE output-text deltas into chunks and
// keeps the completed response event for finish.
type openAIChunkStream struct {
	stream    *ssestream.Stream[responses.ResponseStreamEventUnion]
	schema    *core.OutputSchema
	completed *responses.Response
}

func (s *openAIChunkStream) Recv() (core.Chunk, error) {
	for s.stream.Next() {
		event := s.stream.Current()
		switch event.Type {
		case "response.output_text.delta", "response.refusal.delta":
			return core.Chunk{Text: event.Delta}, nil
		case "response.completed", "response.failed", "response.incomplete":
			response := event.Response
			s.completed = &response
		case "error":
			return core.Chunk{}, openAIStreamEventError(event)
		}
	}
	if err := s.stream.Err(); err != nil {
		return core.Chunk{}, normalizeOpenAIError(err, "generate_stream")
	}
	return core.Chunk{}, core.ErrStreamDone
}

func (s *openAIChunkStream) Finish() (*core.Response, error) {
	if s.completed == nil {
		return nil, &core.Error{
			Provider:  core.ProviderOpenAI,
			Flavor:    core.FlavorAPI,
			Operation: "generate_stream",
			Code:      "missing_response_completion",
			Message:   "OpenAI stream ended without a completed response",
		}
	}
	if err := validateOpenAIResponse(s.completed, "generate_stream"); err != nil {
		return nil, err
	}
	response := openAIResponse(s.completed)
	// The completed response's output text equals the concatenated deltas the
	// root Stream reports, so StructuredOutput comes from the same bytes.
	if err := setStructuredOutput(response, s.schema, core.ProviderOpenAI, core.FlavorAPI, "generate_stream"); err != nil {
		return nil, err
	}
	return response, nil
}

func (s *openAIChunkStream) Close() error {
	return s.stream.Close()
}

func openAIStreamEventError(event responses.ResponseStreamEventUnion) error {
	code := event.Code
	if code == "" {
		code = "stream_error"
	}
	message := event.Message
	if message == "" {
		message = "OpenAI stream reported an error"
	}
	return &core.Error{
		Provider:  core.ProviderOpenAI,
		Flavor:    core.FlavorAPI,
		Operation: "generate_stream",
		Code:      code,
		Message:   message,
	}
}

func openAIResponse(response *responses.Response) *core.Response {
	return &core.Response{
		ID:               response.ID,
		Model:            string(response.Model),
		Text:             openAIResponseText(response),
		CompletionReason: openAICompletionReason(response),
		Usage: core.Usage{
			InputTokens:              int(response.Usage.InputTokens),
			OutputTokens:             int(response.Usage.OutputTokens),
			CachedInputTokens:        int(response.Usage.InputTokensDetails.CachedTokens),
			CacheCreationInputTokens: int(response.Usage.InputTokensDetails.CacheWriteTokens),
			ReasoningTokens:          int(response.Usage.OutputTokensDetails.ReasoningTokens),
		},
	}
}

func validateOpenAIResponse(response *responses.Response, operation string) error {
	// An incomplete response that stopped for a reason the library maps to a
	// CompletionReason (output-token limit or content filter) is a normal result
	// carrying partial text, not an error.
	if response.Status == responses.ResponseStatusIncomplete && response.Error.Code == "" {
		switch response.IncompleteDetails.Reason {
		case "max_output_tokens", "content_filter":
			return nil
		}
	}
	if response.Status != responses.ResponseStatusCompleted {
		code := string(response.Status)
		message := "OpenAI response did not complete"
		if response.Error.Code != "" {
			code = string(response.Error.Code)
			message = response.Error.Message
		} else if response.IncompleteDetails.Reason != "" {
			code = response.IncompleteDetails.Reason
			message = "OpenAI response was incomplete: " + response.IncompleteDetails.Reason
		}
		if code == "" {
			code = "unknown_status"
		}
		return &core.Error{
			Provider:  core.ProviderOpenAI,
			Flavor:    core.FlavorAPI,
			Operation: operation,
			Code:      code,
			Message:   message,
		}
	}

	return nil
}

func openAIResponseText(response *responses.Response) string {
	var text strings.Builder
	for _, item := range response.Output {
		for _, content := range item.Content {
			switch content.Type {
			case "output_text":
				text.WriteString(content.Text)
			case "refusal":
				text.WriteString(content.Refusal)
			}
		}
	}
	return text.String()
}

func openAICompletionReason(response *responses.Response) core.CompletionReason {
	if response.Status == responses.ResponseStatusIncomplete {
		switch response.IncompleteDetails.Reason {
		case "max_output_tokens":
			return core.CompletionReasonMaxTokens
		case "content_filter":
			return core.CompletionReasonContentFilter
		}
	}
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Type == "refusal" {
				return core.CompletionReasonRefusal
			}
		}
	}
	return core.CompletionReasonStop
}

func normalizeOpenAIError(err error, operation string) error {
	normalized := &core.Error{
		Provider:  core.ProviderOpenAI,
		Flavor:    core.FlavorAPI,
		Operation: operation,
		Err:       err,
	}
	var apiError *openai.Error
	if errors.As(err, &apiError) {
		normalized.Status = apiError.StatusCode
		normalized.Code = apiError.Code
		normalized.Message = apiError.Message
		if normalized.Code == "" {
			normalized.Code = apiError.Type
		}
	}
	return normalized
}
