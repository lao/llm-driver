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
	// ClaudeOptions carries Claude CLI flags for a CLI-flavored client.
	ClaudeOptions = core.ClaudeOptions
	// CodexOptions carries Codex CLI flags for a CLI-flavored client.
	CodexOptions = core.CodexOptions
	// Role identifies the author of a Message.
	Role = core.Role
	// Message is one text turn in a generation request.
	Message = core.Message
	// Request is a provider-neutral text generation request.
	Request = core.Request
	// OutputSchema constrains a response to a JSON document matching a JSON Schema.
	OutputSchema = core.OutputSchema
	// Effort selects a reasoning-effort level a target may reject.
	Effort = core.Effort
	// Usage reports token counts made available by a target.
	Usage = core.Usage
	// CompletionReason describes why a generation ended, when reported.
	CompletionReason = core.CompletionReason
	// Response is a provider-neutral text generation result.
	Response = core.Response
	// Error adds target and operation context to configuration and runtime
	// failures. For CLI targets, process-exit errors use a nonzero Status with
	// empty Reason and Code; API/result errors populate at least one of those
	// fields.
	Error = core.Error
	// Chunk is one incremental piece of a streaming response.
	Chunk = core.Chunk
	// ChunkKind classifies a streaming Chunk; the zero value is text.
	ChunkKind = core.ChunkKind
	// Feature identifies an optional generation capability a target may reject.
	Feature = core.Feature
	// Part is one piece of a Message's structured content.
	Part = core.Part
	// TextPart is plain text content within a Parts message.
	TextPart = core.TextPart
	// ToolCallPart is an assistant tool-call echo carried back into a request.
	ToolCallPart = core.ToolCallPart
	// ToolResultPart is the caller-supplied result of a tool call.
	ToolResultPart = core.ToolResultPart
	// Tool describes a caller-executed tool the model may call.
	Tool = core.Tool
	// ToolChoice controls tool selection; the zero value is auto.
	ToolChoice = core.ToolChoice
	// ToolChoiceMode selects how the model may use tools.
	ToolChoiceMode = core.ToolChoiceMode
	// ToolCall is a tool invocation the model requested.
	ToolCall = core.ToolCall
)

const (
	// ProviderClaude selects Anthropic Claude.
	ProviderClaude = core.ProviderClaude
	// ProviderOpenAI selects OpenAI.
	ProviderOpenAI = core.ProviderOpenAI
	// ProviderOpenCode selects the local opencode agent CLI (2.x). Its model is
	// a "provider/model" id from `opencode models` (see ListModels); Effort is
	// appended as the "#variant" suffix used by opencode run --model.
	ProviderOpenCode = core.ProviderOpenCode
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
	// CompletionReasonToolUse indicates that generation ended to call a tool.
	CompletionReasonToolUse = core.CompletionReasonToolUse
	// CompletionReasonContentFilter indicates that a content filter stopped generation.
	CompletionReasonContentFilter = core.CompletionReasonContentFilter
	// CompletionReasonUnverified means no terminal finish reason was reported;
	// truncation cannot be ruled out (opencode 2.x text-only turns).
	CompletionReasonUnverified = core.CompletionReasonUnverified
	// CodeMissingFinalMessage identifies a completed Codex CLI turn that emitted
	// no completed agent message.
	CodeMissingFinalMessage = core.CodeMissingFinalMessage
	// CodeToolCalls identifies a CLI turn that ended on a tool call without
	// producing response text.
	CodeToolCalls = core.CodeToolCalls

	// EffortLow requests the lowest reasoning effort.
	EffortLow = core.EffortLow
	// EffortMedium requests medium reasoning effort.
	EffortMedium = core.EffortMedium
	// EffortHigh requests high reasoning effort.
	EffortHigh = core.EffortHigh
	// EffortXHigh requests extra-high reasoning effort; openai/api rejects it.
	EffortXHigh = core.EffortXHigh
	// EffortMax requests maximum reasoning effort; openai/api rejects it.
	EffortMax = core.EffortMax

	// FeatureStructuredOutput requests a JSON-schema-constrained response.
	FeatureStructuredOutput = core.FeatureStructuredOutput
	// FeatureReasoningEffort requests a reasoning-effort level.
	FeatureReasoningEffort = core.FeatureReasoningEffort
	// FeatureReasoningVisibility requests visible reasoning or thinking output.
	FeatureReasoningVisibility = core.FeatureReasoningVisibility
	// FeatureTools requests caller-executed tool calling.
	FeatureTools = core.FeatureTools
	// FeatureImageInput requests image content in the input.
	FeatureImageInput = core.FeatureImageInput
	// FeatureDocumentInput requests PDF or document content in the input.
	FeatureDocumentInput = core.FeatureDocumentInput
	// FeatureSamplingParams requests temperature or top-p sampling controls.
	FeatureSamplingParams = core.FeatureSamplingParams
	// FeatureTopK requests top-k sampling, which openai/api lacks.
	FeatureTopK = core.FeatureTopK
	// FeatureStopSequences requests stop sequences, which openai/api lacks.
	FeatureStopSequences = core.FeatureStopSequences
	// FeatureWebSearch requests provider-executed web search.
	FeatureWebSearch = core.FeatureWebSearch
	// FeatureUserID requests an abuse-tracking end-user identifier.
	FeatureUserID = core.FeatureUserID
	// FeatureContinuation requests conversation continuation by identifier.
	FeatureContinuation = core.FeatureContinuation
	// FeatureTokenCounting requests standalone token counting.
	FeatureTokenCounting = core.FeatureTokenCounting

	// KindText marks a response-text streaming delta; it is the zero value.
	KindText = core.KindText
	// KindToolInput marks a delta of a tool call's JSON input arguments.
	KindToolInput = core.KindToolInput

	// ToolChoiceAuto lets the model decide whether to call a tool (zero value).
	ToolChoiceAuto = core.ToolChoiceAuto
	// ToolChoiceNone forbids tool calls.
	ToolChoiceNone = core.ToolChoiceNone
	// ToolChoiceRequired forces the model to call some tool.
	ToolChoiceRequired = core.ToolChoiceRequired
	// ToolChoiceTool forces the model to call the named tool.
	ToolChoiceTool = core.ToolChoiceTool
)

var (
	// ErrInvalidConfig identifies client configuration rejected by New.
	ErrInvalidConfig = core.ErrInvalidConfig
	// ErrInvalidRequest identifies a request rejected before provider invocation.
	ErrInvalidRequest = core.ErrInvalidRequest
	// ErrExecutableNotFound identifies a missing provider CLI executable.
	ErrExecutableNotFound = core.ErrExecutableNotFound
	// ErrUnsupportedFeature identifies a request feature the selected target
	// cannot honor.
	ErrUnsupportedFeature = core.ErrUnsupportedFeature
)

// AllFeatures returns every Feature in parity-inventory order.
func AllFeatures() []Feature {
	return core.AllFeatures()
}

// User builds a user-role text message.
func User(text string) Message {
	return core.User(text)
}

// Assistant builds an assistant-role text message.
func Assistant(text string) Message {
	return core.Assistant(text)
}

// ToolResult builds a user-role message carrying a single tool result, matched
// to a prior ToolCall by id. isErr marks content as a tool execution error.
func ToolResult(id string, content string, isErr bool) Message {
	return core.ToolResult(id, content, isErr)
}
