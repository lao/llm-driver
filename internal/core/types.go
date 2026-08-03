package core

import (
	"fmt"
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

// Request is a provider-neutral text generation request.
type Request struct {
	// System is an optional instruction applied to the request.
	System string `json:"system,omitempty"`
	// Messages is the non-empty user/assistant transcript.
	Messages []Message `json:"messages"`
	// MaxTokens is required and enforced by API flavors. CLI flavors validate
	// it for request portability but cannot reliably enforce it.
	MaxTokens int `json:"max_tokens"`
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
	// Provider identifies the selected provider.
	Provider Provider `json:"provider"`
	// Flavor identifies the selected transport flavor.
	Flavor Flavor `json:"flavor"`
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
	return nil
}
