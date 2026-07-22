// Package claudewrap provides a backend-agnostic Go client for Claude that
// mirrors the Anthropic Messages API surface.
//
// A single interface is served by one of two interchangeable backends, chosen
// at construction time:
//
//   - "sdk" — talks HTTP to the Anthropic API via the official anthropic-sdk-go.
//   - "cli" — shells out to a locally installed Claude Code CLI (claude -p).
//
// Callers work exclusively with the library-owned neutral types declared in this
// package; the underlying anthropic-sdk-go types never cross the package boundary.
//
//	c, _ := claudewrap.New(claudewrap.WithSDK(apiKey)) // or WithCLI()
//	resp, err := c.Messages.New(ctx, claudewrap.MessageParams{
//	    Model:     "claude-opus-4-8",
//	    MaxTokens: 1024,
//	    Messages:  []claudewrap.Message{claudewrap.UserMessage("Hello")},
//	})
package claudewrap
