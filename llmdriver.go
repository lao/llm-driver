package llmdriver

import "github.com/lao/llm-driver/internal/core"

// The package's data types live in internal/core so backend implementations
// can share them without an import cycle. These aliases keep the public API
// on the root package.
type (
	// Provider identifies the model provider selected by a Config.
	Provider = core.Provider
	// Flavor identifies how a provider is invoked.
	Flavor = core.Flavor
	// Config selects a provider, transport flavor, and model for a Client.
	Config = core.Config
	// Role identifies the author of a Message.
	Role = core.Role
	// Message is one text turn in a generation request.
	Message = core.Message
	// Request is a provider-neutral text generation request.
	Request = core.Request
	// Usage reports token counts made available by a target.
	Usage = core.Usage
	// CompletionReason describes why a generation ended, when reported.
	CompletionReason = core.CompletionReason
	// Response is a provider-neutral text generation result.
	Response = core.Response
	// Error adds target and operation context to configuration and runtime
	// failures.
	Error = core.Error
	// Chunk is one incremental piece of streaming response text.
	Chunk = core.Chunk
)

const (
	// ProviderClaude selects Anthropic Claude.
	ProviderClaude = core.ProviderClaude
	// ProviderOpenAI selects OpenAI.
	ProviderOpenAI = core.ProviderOpenAI
	// FlavorAPI invokes a provider's hosted API through its official Go SDK.
	FlavorAPI = core.FlavorAPI
	// FlavorCLI invokes the provider's locally authenticated command-line tool.
	FlavorCLI = core.FlavorCLI
	// RoleUser identifies user input.
	RoleUser = core.RoleUser
	// RoleAssistant identifies model output supplied as conversation history.
	RoleAssistant = core.RoleAssistant
	// CompletionReasonStop indicates a normal model or stop-sequence ending.
	CompletionReasonStop = core.CompletionReasonStop
	// CompletionReasonMaxTokens indicates that the output-token limit was reached.
	CompletionReasonMaxTokens = core.CompletionReasonMaxTokens
	// CompletionReasonRefusal indicates that the model returned a refusal.
	CompletionReasonRefusal = core.CompletionReasonRefusal
)

var (
	// ErrInvalidConfig identifies client configuration rejected by New.
	ErrInvalidConfig = core.ErrInvalidConfig
	// ErrInvalidRequest identifies a request rejected before provider invocation.
	ErrInvalidRequest = core.ErrInvalidRequest
	// ErrExecutableNotFound identifies a missing provider CLI executable.
	ErrExecutableNotFound = core.ErrExecutableNotFound
)

// User builds a user-role text message.
func User(text string) Message {
	return core.User(text)
}

// Assistant builds an assistant-role text message.
func Assistant(text string) Message {
	return core.Assistant(text)
}
