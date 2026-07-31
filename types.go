package llmwrapper

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
	Role Role   `json:"role"`
	Text string `json:"text"`
}

// Request is a provider-neutral text generation request.
type Request struct {
	System    string    `json:"system,omitempty"`
	Messages  []Message `json:"messages"`
	MaxTokens int       `json:"max_tokens"`
}

// Usage reports token counts made available by a target.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CachedInputTokens        int `json:"cached_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
}

// Response is a provider-neutral text generation result.
type Response struct {
	ID       string   `json:"id,omitempty"`
	Model    string   `json:"model"`
	Text     string   `json:"text"`
	Usage    Usage    `json:"usage"`
	Provider Provider `json:"provider"`
	Flavor   Flavor   `json:"flavor"`
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
