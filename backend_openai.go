package llmwrapper

import (
	"context"
	"errors"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
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

func (b *openAIBackend) generate(ctx context.Context, request Request) (*Response, error) {
	response, err := b.client.Responses.New(ctx, b.params(request))
	if err != nil {
		return nil, normalizeOpenAIError(err, "generate")
	}
	if err := validateOpenAIResponse(response, "generate"); err != nil {
		return nil, err
	}
	return openAIResponse(response), nil
}

func (b *openAIBackend) generateStream(ctx context.Context, request Request) (chunkStream, error) {
	stream := b.client.Responses.NewStreaming(ctx, b.params(request))
	// NewStreaming records transport and non-2xx failures before any event is
	// read, so a failed start is reported by GenerateStream rather than Next.
	if err := stream.Err(); err != nil {
		_ = stream.Close()
		return nil, normalizeOpenAIError(err, "generate_stream")
	}
	return &openAIChunkStream{stream: stream}, nil
}

func (b *openAIBackend) params(request Request) responses.ResponseNewParams {
	input := make(responses.ResponseInputParam, 0, len(request.Messages))
	for _, message := range request.Messages {
		content := responses.ResponseInputMessageContentListParam{
			responses.ResponseInputContentParamOfInputText(message.Text),
		}
		item := responses.ResponseInputItemParamOfMessage(
			content,
			responses.EasyInputMessageRole(message.Role),
		)
		if message.Role == RoleAssistant {
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
	return params
}

// openAIChunkStream turns Responses SSE output-text deltas into chunks and
// keeps the completed response event for finish.
type openAIChunkStream struct {
	stream    *ssestream.Stream[responses.ResponseStreamEventUnion]
	completed *responses.Response
}

func (s *openAIChunkStream) recv() (Chunk, error) {
	for s.stream.Next() {
		event := s.stream.Current()
		switch event.Type {
		case "response.output_text.delta", "response.refusal.delta":
			return Chunk{Text: event.Delta}, nil
		case "response.completed", "response.failed", "response.incomplete":
			response := event.Response
			s.completed = &response
		case "error":
			return Chunk{}, openAIStreamEventError(event)
		}
	}
	if err := s.stream.Err(); err != nil {
		return Chunk{}, normalizeOpenAIError(err, "generate_stream")
	}
	return Chunk{}, errStreamDone
}

func (s *openAIChunkStream) finish() (*Response, error) {
	if s.completed == nil {
		return nil, &Error{
			Provider:  ProviderOpenAI,
			Flavor:    FlavorAPI,
			Operation: "generate_stream",
			Code:      "missing_response_completion",
			Message:   "OpenAI stream ended without a completed response",
		}
	}
	if err := validateOpenAIResponse(s.completed, "generate_stream"); err != nil {
		return nil, err
	}
	return openAIResponse(s.completed), nil
}

func (s *openAIChunkStream) close() error {
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
	return &Error{
		Provider:  ProviderOpenAI,
		Flavor:    FlavorAPI,
		Operation: "generate_stream",
		Code:      code,
		Message:   message,
	}
}

func openAIResponse(response *responses.Response) *Response {
	return &Response{
		ID:               response.ID,
		Model:            string(response.Model),
		Text:             openAIResponseText(response),
		CompletionReason: openAICompletionReason(response),
		Usage: Usage{
			InputTokens:              int(response.Usage.InputTokens),
			OutputTokens:             int(response.Usage.OutputTokens),
			CachedInputTokens:        int(response.Usage.InputTokensDetails.CachedTokens),
			CacheCreationInputTokens: int(response.Usage.InputTokensDetails.CacheWriteTokens),
			ReasoningTokens:          int(response.Usage.OutputTokensDetails.ReasoningTokens),
		},
	}
}

func validateOpenAIResponse(response *responses.Response, operation string) error {
	if response.Status == responses.ResponseStatusIncomplete &&
		response.IncompleteDetails.Reason == "max_output_tokens" &&
		response.Error.Code == "" {
		return nil
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
		return &Error{
			Provider:  ProviderOpenAI,
			Flavor:    FlavorAPI,
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

func openAICompletionReason(response *responses.Response) CompletionReason {
	if response.Status == responses.ResponseStatusIncomplete &&
		response.IncompleteDetails.Reason == "max_output_tokens" {
		return CompletionReasonMaxTokens
	}
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Type == "refusal" {
				return CompletionReasonRefusal
			}
		}
	}
	return CompletionReasonStop
}

func normalizeOpenAIError(err error, operation string) error {
	normalized := &Error{
		Provider:  ProviderOpenAI,
		Flavor:    FlavorAPI,
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
