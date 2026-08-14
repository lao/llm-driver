package backends

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
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
		if len(message.Parts) == 0 {
			input = append(input, openAITextMessage(message.Role, message.Text))
			continue
		}
		for _, part := range message.Parts {
			switch p := part.(type) {
			case core.TextPart:
				input = append(input, openAITextMessage(message.Role, p.Text))
			case core.ToolCallPart:
				input = append(input, responses.ResponseInputItemParamOfFunctionCall(string(p.Input), p.ID, p.Name))
			case core.ToolResultPart:
				input = append(input, responses.ResponseInputItemParamOfFunctionCallOutput(p.ToolCallID, p.Content))
			default:
				panic(fmt.Sprintf("openai: unhandled part type %T (core validation should reject it)", part))
			}
		}
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
	// Sampling params are passed through; TopK and StopSequences are rejected in
	// core (the Responses API lacks both), so only Temperature/TopP/UserID map.
	if request.Temperature != nil {
		params.Temperature = openai.Float(*request.Temperature)
	}
	if request.TopP != nil {
		params.TopP = openai.Float(*request.TopP)
	}
	if request.UserID != "" {
		params.SafetyIdentifier = openai.String(request.UserID)
	}
	if len(request.Tools) > 0 {
		params.Tools = openAITools(request.Tools)
	}
	if request.ToolChoice.Mode != core.ToolChoiceAuto {
		params.ToolChoice = openAIToolChoice(request.ToolChoice)
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

// openAITextMessage builds one input message item, preserving the final-answer
// phase on assistant turns that the text path has always set.
func openAITextMessage(role core.Role, text string) responses.ResponseInputItemUnionParam {
	content := responses.ResponseInputMessageContentListParam{
		responses.ResponseInputContentParamOfInputText(text),
	}
	item := responses.ResponseInputItemParamOfMessage(content, responses.EasyInputMessageRole(role))
	if role == core.RoleAssistant {
		// Preserve the completed-answer phase so current Codex models keep full
		// follow-up quality on prior assistant turns.
		item.OfMessage.Phase = responses.EasyInputMessagePhaseFinalAnswer
	}
	return item
}

// openAITools maps neutral tools to Responses function tools. The schema is
// validated as a JSON object in core, so the unmarshal cannot fail here.
func openAITools(tools []core.Tool) []responses.ToolUnionParam {
	mapped := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		var parameters map[string]any
		_ = json.Unmarshal(tool.InputSchema, &parameters)
		function := responses.ToolParamOfFunction(tool.Name, parameters, false)
		if tool.Description != "" {
			function.OfFunction.Description = openai.String(tool.Description)
		}
		mapped = append(mapped, function)
	}
	return mapped
}

// openAIToolChoice maps a non-auto neutral choice to the Responses union.
func openAIToolChoice(choice core.ToolChoice) responses.ResponseNewParamsToolChoiceUnion {
	switch choice.Mode {
	case core.ToolChoiceNone:
		return responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone)}
	case core.ToolChoiceRequired:
		return responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired)}
	case core.ToolChoiceTool:
		return responses.ResponseNewParamsToolChoiceUnion{OfFunctionTool: &responses.ToolChoiceFunctionParam{Name: choice.Name}}
	default:
		return responses.ResponseNewParamsToolChoiceUnion{}
	}
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
		case "response.function_call_arguments.delta":
			// The completed response carries the parsed calls, so Finish reports
			// them; this surfaces the raw JSON arguments as progress.
			return core.Chunk{Kind: core.KindToolInput, Text: event.Delta}, nil
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
		ToolCalls:        openAIToolCalls(response),
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

// openAIToolCalls extracts the function_call output items as neutral tool calls,
// preserving provider order (parallel calls included).
func openAIToolCalls(response *responses.Response) []core.ToolCall {
	var toolCalls []core.ToolCall
	for _, item := range response.Output {
		if item.Type == "function_call" {
			toolCalls = append(toolCalls, core.ToolCall{
				ID:    item.CallID,
				Name:  item.Name,
				Input: json.RawMessage(item.Arguments.OfString),
			})
		}
	}
	return toolCalls
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
	hasToolCall := false
	for _, item := range response.Output {
		if item.Type == "function_call" {
			hasToolCall = true
		}
		for _, content := range item.Content {
			if content.Type == "refusal" {
				return core.CompletionReasonRefusal
			}
		}
	}
	// A completed turn that requested tools maps to tool_use, matching Anthropic's
	// stop_reason so the round-trip normalizes identically on both API flavors.
	if hasToolCall {
		return core.CompletionReasonToolUse
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
