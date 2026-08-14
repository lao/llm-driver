package core

import (
	"encoding/json"
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

// Message is one text turn in a generation request.
type Message struct {
	// Role identifies the user or assistant author.
	Role Role `json:"role"`
	// Text is the message's non-empty text content.
	Text string `json:"text"`
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
	// Provider identifies the selected provider.
	Provider Provider `json:"provider"`
	// Flavor identifies the selected transport flavor.
	Flavor Flavor `json:"flavor"`
}

// Chunk is one incremental piece of response text.
type Chunk struct {
	// Text is the incremental response text; it is never empty.
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
		if strings.TrimSpace(message.Text) == "" {
			return fmt.Errorf("%w: message %d text is required", ErrInvalidRequest, i)
		}
	}
	if r.OutputSchema != nil {
		if _, err := r.OutputSchema.SchemaObject(); err != nil {
			return err
		}
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
