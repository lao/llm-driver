package llmwrapper

import (
	"context"
	"errors"

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
		input = append(input, responses.ResponseInputItemParamOfMessage(
			content,
			responses.EasyInputMessageRole(message.Role),
		))
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
		ID:    response.ID,
		Model: string(response.Model),
		Text:  response.OutputText(),
		Usage: Usage{
			InputTokens:       int(response.Usage.InputTokens),
			OutputTokens:      int(response.Usage.OutputTokens),
			CachedInputTokens: int(response.Usage.InputTokensDetails.CachedTokens),
			ReasoningTokens:   int(response.Usage.OutputTokensDetails.ReasoningTokens),
		},
	}, nil
}

func validateOpenAIResponse(response *responses.Response) error {
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

	for _, item := range response.Output {
		if item.Type != "message" {
			continue
		}
		for _, content := range item.Content {
			if content.Type == "refusal" {
				message := content.Refusal
				if message == "" {
					message = "OpenAI response was refused"
				}
				return &Error{
					Provider:  ProviderOpenAI,
					Flavor:    FlavorAPI,
					Operation: "generate",
					Code:      "refusal",
					Message:   message,
				}
			}
		}
	}
	return nil
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
		if normalized.Code == "" {
			normalized.Code = apiError.Type
		}
	}
	return normalized
}
