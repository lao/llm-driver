// Package llmwrapper provides one text-generation API backed by switchable
// Claude and OpenAI API or CLI targets.
//
// Construct a client with an explicit provider, flavor, and model, then send a
// provider-neutral request:
//
//	client, err := llmwrapper.New(llmwrapper.Config{
//		Provider: llmwrapper.ProviderOpenAI,
//		Flavor:   llmwrapper.FlavorCLI,
//		Model:    "gpt-5.6-sol",
//	})
//	if err != nil {
//		return err
//	}
//
//	response, err := client.Generate(ctx, llmwrapper.Request{
//		System:    "Answer concisely.",
//		Messages:  []llmwrapper.Message{llmwrapper.User("Hello")},
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
// or Codex CLI. CLI targets use local authentication and may not support every
// hosted-API control, including an exact maximum output-token limit. Chunk
// granularity is target-defined: the CLI flavors may deliver fewer, larger
// chunks than the hosted APIs.
package llmwrapper
