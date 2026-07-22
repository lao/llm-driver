package claudewrap

import (
	"context"
	"errors"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// sdkBackend talks HTTP to the Anthropic API via the official anthropic-sdk-go.
type sdkBackend struct {
	client       anthropic.Client
	defaultModel string
}

func newSDKBackend(c *config) *sdkBackend {
	opts := []option.RequestOption{}
	if c.sdkAPIKey != "" {
		opts = append(opts, option.WithAPIKey(c.sdkAPIKey))
	}
	if c.sdkBaseURL != "" {
		opts = append(opts, option.WithBaseURL(c.sdkBaseURL))
	}
	if c.sdkHTTP != nil {
		opts = append(opts, option.WithHTTPClient(c.sdkHTTP))
	}
	return &sdkBackend{
		client:       anthropic.NewClient(opts...),
		defaultModel: c.defaultModel,
	}
}

func (b *sdkBackend) name() string { return "sdk" }

func (b *sdkBackend) messages(ctx context.Context, p MessageParams) (*Response, error) {
	model := p.Model
	if model == "" {
		model = b.defaultModel
	}
	if model == "" {
		model = DefaultModel
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: int64(p.MaxTokens),
		Messages:  toSDKMessages(p.Messages),
	}
	if p.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: p.System}}
	}
	if p.Temperature != nil {
		params.Temperature = anthropic.Float(*p.Temperature)
	}
	if len(p.StopSeqs) > 0 {
		params.StopSequences = p.StopSeqs
	}

	msg, err := b.client.Messages.New(ctx, params)
	if err != nil {
		return nil, sdkError("messages.new", err)
	}
	return fromSDKMessage(msg), nil
}

// toSDKMessages maps neutral messages to SDK message params. Only text blocks
// are forwarded on the request side in v0.1.
func toSDKMessages(msgs []Message) []anthropic.MessageParam {
	out := make([]anthropic.MessageParam, 0, len(msgs))
	for _, m := range msgs {
		blocks := make([]anthropic.ContentBlockParamUnion, 0, len(m.Content))
		for _, blk := range m.Content {
			if blk.Type == BlockText {
				blocks = append(blocks, anthropic.NewTextBlock(blk.Text))
			}
		}
		if m.Role == RoleAssistant {
			out = append(out, anthropic.NewAssistantMessage(blocks...))
		} else {
			out = append(out, anthropic.NewUserMessage(blocks...))
		}
	}
	return out
}

// fromSDKMessage maps an SDK response to the neutral Response.
func fromSDKMessage(m *anthropic.Message) *Response {
	r := &Response{
		ID:         m.ID,
		Model:      string(m.Model),
		Role:       Role(m.Role),
		StopReason: StopReason(m.StopReason),
		Backend:    "sdk",
		Usage: Usage{
			InputTokens:         int(m.Usage.InputTokens),
			OutputTokens:        int(m.Usage.OutputTokens),
			CacheReadTokens:     int(m.Usage.CacheReadInputTokens),
			CacheCreationTokens: int(m.Usage.CacheCreationInputTokens),
		},
	}
	for _, blk := range m.Content {
		switch blk.Type {
		case "text":
			r.Content = append(r.Content, TextBlock(blk.Text))
		case "tool_use":
			r.Content = append(r.Content, ContentBlock{
				Type:      BlockToolUse,
				ToolName:  blk.Name,
				ToolInput: blk.Input,
				ToolUseID: blk.ID,
			})
		}
	}
	return r
}

// sdkError wraps an anthropic-sdk-go error in a *Error, extracting status/type
// when the cause is an *anthropic.Error.
func sdkError(op string, err error) error {
	e := &Error{Backend: "sdk", Op: op, Err: err, Message: err.Error()}
	var apierr *anthropic.Error
	if errors.As(err, &apierr) {
		e.Status = apierr.StatusCode
		e.Type = string(apierr.Type())
		e.Message = ""
	}
	return e
}
