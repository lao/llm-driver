// Package llmdriver provides one text-generation API backed by switchable
// Claude and OpenAI API or CLI targets, or the local opencode agent CLI.
//
// Construct a client with an explicit provider, flavor, and model, then send a
// provider-neutral request:
//
//	client, err := llmdriver.New(llmdriver.Config{
//		Provider: llmdriver.ProviderOpenAI,
//		Flavor:   llmdriver.FlavorCLI,
//		Model:    "gpt-5.6-sol",
//	})
//	if err != nil {
//		return err
//	}
//
//	response, err := client.Generate(ctx, llmdriver.Request{
//		System:    "Answer concisely.",
//		Messages:  []llmdriver.Message{llmdriver.User("Hello")},
//		MaxTokens: 1024,
//	})
//	if err != nil {
//		return err
//	}
//	fmt.Println(response.Text)
//
// GenerateStream runs the same request incrementally and ends in the same
// normalized Response:
//
//	stream, err := client.GenerateStream(ctx, request)
//	if err != nil {
//		return err
//	}
//	defer stream.Close()
//
//	for stream.Next() {
//		fmt.Print(stream.Current().Text)
//	}
//	response, err := stream.Response()
//
// Changing Config is sufficient to select Claude API, Claude CLI, OpenAI API,
// Codex CLI, or opencode CLI. CLI targets use local authentication and may not
// support every hosted-API control, including an exact maximum output-token
// limit. Chunk granularity is target-defined: the CLI flavors may deliver
// fewer, larger chunks than the hosted APIs.
//
// opencode fronts many upstream providers, so its model is a "provider/model"
// id. ListModels runs `opencode models` and returns the ids the installed
// opencode can actually use.
//
// Optional generation features beyond the baseline (system, messages,
// MaxTokens, streaming) are gated per target. Client.Supports reports whether
// the configured target honors a Feature, and Generate and GenerateStream
// reject a request exercising an unsupported feature with ErrUnsupportedFeature,
// naming the feature and target rather than silently degrading. The README
// capability matrix lists each feature's support across the targets.
//
// Tool calling (API flavors only) is driven by Request.Tools and
// Request.ToolChoice. When the model calls a tool the Response carries
// CompletionReasonToolUse and one ToolCall per invocation; the caller runs each
// call and replies with ToolResult messages, echoing the calls as ToolCallPart
// content. A Message sets exactly one of Text or Parts: Text is the text-only
// fast path the User and Assistant helpers produce, and Parts carries tool
// calls and results. Streaming surfaces tool-argument deltas as Chunk values
// with Kind KindToolInput, with the parsed ToolCalls on the final Response.
package llmdriver
