package llmwrapper

import (
	"context"
	"errors"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
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

	response, err := b.client.Responses.New(ctx, params)
	if err != nil {
		return nil, normalizeOpenAIError(err)
	}
	if err := validateOpenAIResponse(response); err != nil {
		return nil, err
	}
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
	}, nil
}

func validateOpenAIResponse(response *responses.Response) error {
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
			Operation: "generate",
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

func normalizeOpenAIError(err error) error {
	normalized := &Error{
		Provider:  ProviderOpenAI,
		Flavor:    FlavorAPI,
		Operation: "generate",
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

// TODO(S3): replaced by the Responses streaming adapter.
func (b *openAIBackend) generateStream(context.Context, Request) (chunkStream, error) {
	return nil, errors.New("not implemented")
}
