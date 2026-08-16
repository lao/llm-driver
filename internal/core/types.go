package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

// Role identifies the author of a Message.
type Role string

const (
	// RoleUser identifies user input.
	RoleUser Role = "user"
	// RoleAssistant identifies model output supplied as conversation history.
	RoleAssistant Role = "assistant"
)

// Message is one turn in a generation request. Exactly one of Text or Parts is
// set: Text is the text-only fast path preserved from earlier phases, and Parts
// carries the structured content (tool calls, tool results) tool calling needs.
type Message struct {
	// Role identifies the user or assistant author.
	Role Role `json:"role"`
	// Text is the message's non-empty text content when Parts is empty.
	Text string `json:"text,omitempty"`
	// Parts is the message's structured content when Text is empty.
	Parts []Part `json:"parts,omitempty"`
}

// Part is one piece of a Message's structured content. The set is closed: only
// this package can implement it (unexported marker method), so a Message.Parts
// value can only hold TextPart, ToolCallPart, or ToolResultPart. Callers build
// them as values and adapters type-switch over them.
type Part interface {
	isPart()
}

// TextPart is plain text content within a Parts message.
type TextPart struct {
	// Text is the part's non-empty text content.
	Text string
}

// ToolCallPart is an assistant tool-call echo carried back into a request so the
// provider sees the call it previously made. Input is the raw JSON arguments.
type ToolCallPart struct {
	// ID is the provider-assigned tool-call identifier.
	ID string
	// Name is the called tool's name.
	Name string
	// Input is the tool-call arguments as a JSON document.
	Input json.RawMessage
}

// ToolResultPart is the caller-supplied result of a tool call, matched to the
// call by ToolCallID. IsError marks the content as an execution error.
type ToolResultPart struct {
	// ToolCallID matches the ToolCall this result answers.
	ToolCallID string
	// Content is the tool's textual result.
	Content string
	// IsError reports whether Content describes a tool execution failure. It is
	// honored only on claude/api; the OpenAI Responses API has no error flag on
	// function_call_output, so on openai/api the Content still reaches the model
	// but the error signal is dropped.
	IsError bool
}

func (TextPart) isPart()       {}
func (ToolCallPart) isPart()   {}
func (ToolResultPart) isPart() {}

// Tool describes a caller-executed tool the model may call. InputSchema is the
// JSON Schema (a JSON object) for the tool's arguments.
type Tool struct {
	// Name is the tool's unique, non-empty name.
	Name string `json:"name"`
	// Description optionally documents the tool for the model.
	Description string `json:"description,omitempty"`
	// InputSchema is the JSON Schema object describing the tool's input.
	InputSchema json.RawMessage `json:"input_schema"`
}

// ToolChoiceMode selects how the model may use tools. The zero value is auto.
type ToolChoiceMode string

const (
	// ToolChoiceAuto lets the model decide whether to call a tool. It is the zero
	// value, so an unset ToolChoice means auto.
	ToolChoiceAuto ToolChoiceMode = ""
	// ToolChoiceNone forbids tool calls.
	ToolChoiceNone ToolChoiceMode = "none"
	// ToolChoiceRequired forces the model to call some tool.
	ToolChoiceRequired ToolChoiceMode = "required"
	// ToolChoiceTool forces the model to call the named tool.
	ToolChoiceTool ToolChoiceMode = "tool"
)

// ToolChoice controls tool selection. The zero value (Mode ToolChoiceAuto) leaves
// tool choice unset. Name is required only when Mode is ToolChoiceTool.
type ToolChoice struct {
	// Mode selects the tool-choice strategy; the zero value is auto.
	Mode ToolChoiceMode `json:"mode,omitempty"`
	// Name is the tool to force when Mode is ToolChoiceTool.
	Name string `json:"name,omitempty"`
}

// ToolCall is a tool invocation the model requested. Input is the raw JSON
// arguments the caller passes to the tool.
type ToolCall struct {
	// ID is the provider-assigned tool-call identifier.
	ID string `json:"id"`
	// Name is the called tool's name.
	Name string `json:"name"`
	// Input is the tool-call arguments as a JSON document.
	Input json.RawMessage `json:"input"`
}

// Effort selects a reasoning-effort level. The zero value "" leaves effort
// unset, preserving today's behavior. The enum is the Anthropic-style superset;
// values are provider-validated, never remapped: each target rejects values it
// does not support (openai/api tops out at EffortHigh).
type Effort string

const (
	// EffortLow requests the lowest reasoning effort.
	EffortLow Effort = "low"
	// EffortMedium requests medium reasoning effort.
	EffortMedium Effort = "medium"
	// EffortHigh requests high reasoning effort.
	EffortHigh Effort = "high"
	// EffortXHigh requests extra-high reasoning effort; openai/api rejects it.
	EffortXHigh Effort = "xhigh"
	// EffortMax requests maximum reasoning effort; openai/api rejects it.
	EffortMax Effort = "max"
)

// OutputSchema constrains a response to a JSON document matching a JSON Schema.
type OutputSchema struct {
	// Name labels the schema. It is required by the openai mapping and defaulted
	// when empty; other targets ignore it.
	Name string `json:"name,omitempty"`
	// Schema is the JSON Schema, which must be a JSON object.
	Schema json.RawMessage `json:"schema"`
}

// SchemaObject parses Schema as the JSON object every target's structured-output
// parameter requires. Request validation rejects a non-object schema, so adapters
// may call it on a validated request and treat an error as impossible.
func (s OutputSchema) SchemaObject() (map[string]any, error) {
	var object map[string]any
	if err := json.Unmarshal(s.Schema, &object); err != nil || object == nil {
		return nil, fmt.Errorf("%w: OutputSchema.Schema must be a JSON object", ErrInvalidRequest)
	}
	return object, nil
}

// Request is a provider-neutral text generation request.
type Request struct {
	// System is an optional instruction applied to the request.
	System string `json:"system,omitempty"`
	// Messages is the non-empty user/assistant transcript.
	Messages []Message `json:"messages"`
	// MaxTokens is required and enforced by API flavors. CLI flavors validate
	// it for request portability but cannot reliably enforce it.
	MaxTokens int `json:"max_tokens"`
	// OutputSchema optionally constrains the response to a JSON document matching
	// a JSON Schema; the result is exposed on Response.StructuredOutput.
	OutputSchema *OutputSchema `json:"output_schema,omitempty"`
	// Effort optionally selects a reasoning-effort level. The zero value ""
	// leaves it unset. A value the target does not support is rejected with
	// ErrUnsupportedFeature before invocation.
	Effort Effort `json:"effort,omitempty"`
	// Temperature optionally sets the sampling temperature. Nil leaves it unset.
	// It is passed through to API flavors (validated only as non-negative and
	// finite); the provider enforces its own upper bound and may reject it.
	Temperature *float64 `json:"temperature,omitempty"`
	// TopP optionally sets nucleus-sampling probability mass. Nil leaves it
	// unset. Passed through like Temperature (non-negative, finite).
	TopP *float64 `json:"top_p,omitempty"`
	// TopK optionally sets top-k sampling. Nil leaves it unset. Only anthropic/api
	// supports it; other targets reject it with ErrUnsupportedFeature. Non-negative.
	TopK *int `json:"top_k,omitempty"`
	// StopSequences optionally sets sequences that stop generation. Empty leaves
	// it unset. Only anthropic/api supports it; other targets reject it.
	StopSequences []string `json:"stop_sequences,omitempty"`
	// UserID optionally sets an abuse-tracking end-user identifier. Empty leaves
	// it unset. API flavors map it (anthropic metadata.user_id, openai
	// safety_identifier); CLI flavors reject it with ErrUnsupportedFeature.
	UserID string `json:"user_id,omitempty"`
	// Tools optionally offers caller-executed tools to the model. Empty leaves it
	// unset. Only API flavors support tools; CLI flavors reject them.
	Tools []Tool `json:"tools,omitempty"`
	// ToolChoice optionally constrains tool selection. The zero value is auto.
	ToolChoice ToolChoice `json:"tool_choice,omitzero"`
}

// Usage reports token counts made available by a target.
type Usage struct {
	// InputTokens is the total input token count reported by the target.
	InputTokens int `json:"input_tokens"`
	// OutputTokens is the total output token count reported by the target.
	OutputTokens int `json:"output_tokens"`
	// CachedInputTokens is the reused prompt-cache token count, when reported.
	CachedInputTokens int `json:"cached_input_tokens,omitempty"`
	// CacheCreationInputTokens is the newly cached input count, when reported.
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	// ReasoningTokens is the reasoning output count, when reported.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// CompletionReason describes why a generation ended, when reported.
type CompletionReason string

const (
	// CompletionReasonStop indicates a normal model or stop-sequence ending.
	CompletionReasonStop CompletionReason = "stop"
	// CompletionReasonMaxTokens indicates that the output-token limit was reached.
	CompletionReasonMaxTokens CompletionReason = "max_tokens"
	// CompletionReasonRefusal indicates that the model returned a refusal.
	CompletionReasonRefusal CompletionReason = "refusal"
	// CompletionReasonToolUse indicates that generation ended to call a tool.
	CompletionReasonToolUse CompletionReason = "tool_use"
	// CompletionReasonContentFilter indicates that a content filter stopped generation.
	CompletionReasonContentFilter CompletionReason = "content_filter"
)

// Response is a provider-neutral text generation result.
type Response struct {
	// ID is the provider response or CLI session/thread identifier, when reported.
	ID string `json:"id,omitempty"`
	// Model is the configured or provider-reported model name.
	Model string `json:"model"`
	// Text is the normalized response text.
	Text string `json:"text"`
	// Usage contains the token counts reported by the selected target.
	Usage Usage `json:"usage"`
	// CompletionReason describes why generation ended, when reported.
	CompletionReason CompletionReason `json:"completion_reason,omitempty"`
	// StructuredOutput is the JSON document produced when OutputSchema was set,
	// validated as JSON. Text keeps the raw model output.
	StructuredOutput json.RawMessage `json:"structured_output,omitempty"`
	// ToolCalls are the tool invocations the model requested, in provider order.
	// The caller executes them and replies with ToolResult messages.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// Provider identifies the selected provider.
	Provider Provider `json:"provider"`
	// Flavor identifies the selected transport flavor.
	Flavor Flavor `json:"flavor"`
}

// ChunkKind classifies a streaming Chunk. The zero value KindText marks a
// response-text delta, so existing consumers that ignore Kind keep their
// behavior; new kinds are additive.
type ChunkKind int

const (
	// KindText marks a response-text delta. It is the zero value.
	KindText ChunkKind = iota
	// KindToolInput marks a delta of a tool call's JSON input arguments.
	KindToolInput
)

// Chunk is one incremental piece of a streaming response. Text is never empty.
// For KindText it is response text; for KindToolInput it is a fragment of the
// current tool call's JSON arguments.
type Chunk struct {
	// Kind classifies the delta; the zero value KindText is response text.
	Kind ChunkKind
	// Text is the incremental delta; it is never empty.
	Text string
}

// User builds a user-role text message.
func User(text string) Message {
	return Message{Role: RoleUser, Text: text}
}

// Assistant builds an assistant-role text message.
func Assistant(text string) Message {
	return Message{Role: RoleAssistant, Text: text}
}

// ToolResult builds a user-role message carrying a single tool result, matched
// to a prior ToolCall by id. isErr marks content as a tool execution error.
func ToolResult(id string, content string, isErr bool) Message {
	return Message{Role: RoleUser, Parts: []Part{ToolResultPart{ToolCallID: id, Content: content, IsError: isErr}}}
}

func (r Request) validate() error {
	if r.MaxTokens <= 0 {
		return fmt.Errorf("%w: MaxTokens must be greater than 0", ErrInvalidRequest)
	}
	if len(r.Messages) == 0 {
		return fmt.Errorf("%w: at least one message is required", ErrInvalidRequest)
	}
	for i, message := range r.Messages {
		if message.Role != RoleUser && message.Role != RoleAssistant {
			return fmt.Errorf("%w: message %d has invalid role %q", ErrInvalidRequest, i, message.Role)
		}
		if err := message.validate(i); err != nil {
			return err
		}
	}
	if r.OutputSchema != nil {
		if _, err := r.OutputSchema.SchemaObject(); err != nil {
			return err
		}
	}
	if err := validateTools(r.Tools); err != nil {
		return err
	}
	if err := validateToolChoice(r.ToolChoice, r.Tools); err != nil {
		return err
	}
	// Sampling params are passed through per SPEC note ¹: reject only clearly
	// invalid values and let the provider enforce its own upper bounds.
	if err := validateSamplingParam(r.Temperature, "Temperature"); err != nil {
		return err
	}
	if err := validateSamplingParam(r.TopP, "TopP"); err != nil {
		return err
	}
	if r.TopK != nil && *r.TopK < 0 {
		return fmt.Errorf("%w: TopK must be non-negative", ErrInvalidRequest)
	}
	return nil
}

// validate enforces exactly one of Text or Parts, then the per-form rules: a
// text message needs non-blank text; a parts message needs at least one part
// and every part well-formed.
func (m Message) validate(i int) error {
	hasParts := len(m.Parts) > 0
	if hasParts && m.Text != "" {
		return fmt.Errorf("%w: message %d sets both Text and Parts; exactly one is allowed", ErrInvalidRequest, i)
	}
	if !hasParts {
		if strings.TrimSpace(m.Text) == "" {
			return fmt.Errorf("%w: message %d text is required", ErrInvalidRequest, i)
		}
		return nil
	}
	for j, part := range m.Parts {
		if err := validatePart(part); err != nil {
			return fmt.Errorf("%w: message %d part %d: %s", ErrInvalidRequest, i, j, err)
		}
	}
	return nil
}

// validatePart checks a single content part. The returned error's text is
// wrapped by the caller, so it carries no sentinel of its own.
func validatePart(part Part) error {
	switch p := part.(type) {
	case TextPart:
		if strings.TrimSpace(p.Text) == "" {
			return errors.New("text is required")
		}
	case ToolCallPart:
		if p.ID == "" || p.Name == "" {
			return errors.New("tool call requires id and name")
		}
		if !json.Valid(p.Input) {
			return errors.New("tool call input must be valid JSON")
		}
	case ToolResultPart:
		if p.ToolCallID == "" {
			return errors.New("tool result requires a tool-call id")
		}
	default:
		return fmt.Errorf("unknown part type %T", part)
	}
	return nil
}

// validateTools rejects empty or duplicate tool names and non-object schemas.
func validateTools(tools []Tool) error {
	seen := make(map[string]bool, len(tools))
	for _, tool := range tools {
		if tool.Name == "" {
			return fmt.Errorf("%w: tool name is required", ErrInvalidRequest)
		}
		if seen[tool.Name] {
			return fmt.Errorf("%w: duplicate tool name %q", ErrInvalidRequest, tool.Name)
		}
		seen[tool.Name] = true
		var object map[string]any
		if err := json.Unmarshal(tool.InputSchema, &object); err != nil || object == nil {
			return fmt.Errorf("%w: tool %q InputSchema must be a JSON object", ErrInvalidRequest, tool.Name)
		}
	}
	return nil
}

// validateToolChoice rejects an unknown mode and a named tool absent from Tools.
func validateToolChoice(choice ToolChoice, tools []Tool) error {
	switch choice.Mode {
	case ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired:
		return nil
	case ToolChoiceTool:
		for _, tool := range tools {
			if tool.Name == choice.Name {
				return nil
			}
		}
		return fmt.Errorf("%w: ToolChoice names tool %q, which is not in Tools", ErrInvalidRequest, choice.Name)
	default:
		return fmt.Errorf("%w: unknown tool choice mode %q", ErrInvalidRequest, choice.Mode)
	}
}

// validateSamplingParam rejects a non-nil sampling value that is negative or
// non-finite (NaN/Inf); provider-specific upper bounds are not enforced.
func validateSamplingParam(value *float64, name string) error {
	if value == nil {
		return nil
	}
	if math.IsNaN(*value) || math.IsInf(*value, 0) || *value < 0 {
		return fmt.Errorf("%w: %s must be a non-negative finite number", ErrInvalidRequest, name)
	}
	return nil
}
