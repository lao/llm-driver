package backends

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/lao/llmwrapper/internal/core"
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

	message, err := b.client.Messages.New(ctx, params)
	if err != nil {
		return nil, normalizeAnthropicError(err)
	}

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
	}, nil
}

func normalizeAnthropicError(err error) error {
	normalized := &core.Error{
		Provider:  core.ProviderClaude,
		Flavor:    core.FlavorAPI,
		Operation: "generate",
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
