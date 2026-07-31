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

func newOpenAIBackend(config Config) *openAIBackend {
	options := []option.RequestOption{}
	if config.APIKey != "" {
		options = append(options, option.WithAPIKey(config.APIKey))
	}
	if config.BaseURL != "" {
		options = append(options, option.WithBaseURL(config.BaseURL))
	}
	if config.HTTPClient != nil {
		options = append(options, option.WithHTTPClient(config.HTTPClient))
	}
	return &openAIBackend{
		client: openai.NewClient(options...),
		model:  config.Model,
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
