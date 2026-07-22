package claudewrap

import (
	"encoding/json"
	"strings"
)

// DefaultModel is used when MessageParams.Model is empty.
const DefaultModel = "claude-opus-4-8"

// Role identifies the author of a Message.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// BlockType discriminates the kind of a ContentBlock.
type BlockType string

const (
	BlockText       BlockType = "text"
	BlockToolUse    BlockType = "tool_use"
	BlockToolResult BlockType = "tool_result"
)

// StopReason reports why the model stopped generating.
type StopReason string

const (
	StopEndTurn      StopReason = "end_turn"
	StopMaxTokens    StopReason = "max_tokens"
	StopStopSequence StopReason = "stop_sequence"
	StopToolUse      StopReason = "tool_use"
	StopRefusal      StopReason = "refusal"
)

// EventType discriminates a streaming StreamEvent.
type EventType string

const (
	EventMessageStart EventType = "message_start"
	EventTextDelta    EventType = "text_delta"
	EventMessageStop  EventType = "message_stop"
)

// ContentBlock is one piece of a Message or Response: text, a tool call, or a
// tool result. Fields not relevant to Type are left zero.
type ContentBlock struct {
	Type BlockType `json:"type"`

	// Text is populated for BlockText.
	Text string `json:"text,omitempty"`

	// Tool fields are populated for BlockToolUse / BlockToolResult.
	ToolName  string          `json:"tool_name,omitempty"`
	ToolInput json.RawMessage `json:"tool_input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
}

// Message is a single conversational turn.
type Message struct {
	Role    Role           `json:"role"`
	Content []ContentBlock `json:"content"`
}

// Tool describes a callable tool exposed to the model. Best-effort on the CLI
// backend (see package docs / SPEC §7).
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

// MessageParams is the neutral request shape mirroring the Anthropic Messages API.
type MessageParams struct {
	Model       string            // falls back to the client default, then DefaultModel
	MaxTokens   int               // required
	System      string            // optional system prompt
	Messages    []Message         // required, at least one
	Temperature *float64          // optional
	StopSeqs    []string          // optional
	Tools       []Tool            // optional (best-effort on cli backend)
	Metadata    map[string]string // optional
}

// Usage reports token accounting for a Response.
type Usage struct {
	InputTokens         int `json:"input_tokens"`
	OutputTokens        int `json:"output_tokens"`
	CacheReadTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationTokens int `json:"cache_creation_input_tokens,omitempty"`
}

// Response is the neutral result shape returned by both backends.
type Response struct {
	ID         string         `json:"id"`
	Model      string         `json:"model"`
	Role       Role           `json:"role"`
	Content    []ContentBlock `json:"content"`
	StopReason StopReason     `json:"stop_reason"`
	Usage      Usage          `json:"usage"`

	// Backend records which flavor served this response: "sdk" or "cli".
	Backend string `json:"backend"`
}

// Text returns the concatenation of all BlockText content, ignoring other block
// types. Convenience for the common "just give me the string" case.
func (r *Response) Text() string {
	if r == nil {
		return ""
	}
	var b strings.Builder
	for _, blk := range r.Content {
		if blk.Type == BlockText {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// UserMessage builds a user-role Message with a single text block.
func UserMessage(text string) Message {
	return Message{Role: RoleUser, Content: []ContentBlock{TextBlock(text)}}
}

// AssistantMessage builds an assistant-role Message with a single text block.
func AssistantMessage(text string) Message {
	return Message{Role: RoleAssistant, Content: []ContentBlock{TextBlock(text)}}
}

// TextBlock builds a text ContentBlock.
func TextBlock(text string) ContentBlock {
	return ContentBlock{Type: BlockText, Text: text}
}
