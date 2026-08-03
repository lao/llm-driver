package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	llmwrapper "github.com/lao/llmwrapper"
)

const defaultMaxTokens = 1024

type parsedOptions struct {
	config  llmwrapper.Config
	request llmwrapper.Request
	stream  bool
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	os.Exit(exitCode(ctx, os.Args[1:], os.Stdout, os.Stderr))
}

func exitCode(ctx context.Context, args []string, stdout io.Writer, stderr io.Writer) int {
	if err := run(ctx, args, stdout, stderr); err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	return 0
}

func run(ctx context.Context, args []string, stdout io.Writer, stderr io.Writer) error {
	parsed, err := parseOptions(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return nil
	}
	if err != nil {
		return err
	}

	client, err := llmwrapper.New(parsed.config)
	if err != nil {
		return err
	}
	if parsed.stream {
		return generateStream(ctx, client, parsed.request, stdout)
	}

	response, err := client.Generate(ctx, parsed.request)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(stdout, response.Text)
	return err
}

// generateStream prints chunks as they arrive. The final output matches the
// non-streaming path: the chunks concatenate to the same text, and Response
// still reports any failure the stream ended with.
func generateStream(ctx context.Context, client *llmwrapper.Client, request llmwrapper.Request, stdout io.Writer) error {
	stream, err := client.GenerateStream(ctx, request)
	if err != nil {
		return err
	}
	defer stream.Close()

	for stream.Next() {
		if _, err := io.WriteString(stdout, stream.Current().Text); err != nil {
			return err
		}
	}
	if err := stream.Err(); err != nil {
		return err
	}
	if _, err := stream.Response(); err != nil {
		return err
	}
	_, err = fmt.Fprintln(stdout)
	return err
}

func parseOptions(args []string, output io.Writer) (parsedOptions, error) {
	flags := flag.NewFlagSet("switchable", flag.ContinueOnError)
	flags.SetOutput(output)
	providerValue := flags.String("provider", "", "provider: claude or openai (required)")
	flavorValue := flags.String("flavor", "", "flavor: api or cli (required)")
	model := flags.String("model", "", "provider model name (required)")
	prompt := flags.String("prompt", "", "text prompt to send (required)")
	maxTokens := flags.Int("max-tokens", defaultMaxTokens, "maximum output tokens (API flavors; default 1024)")
	stream := flags.Bool("stream", false, "print response text incrementally as it arrives")
	flags.Usage = func() {
		fmt.Fprintln(output, "Usage: switchable --provider <claude|openai> --flavor <api|cli> --model <name> --prompt <text> [--max-tokens <n>] [--stream]")
		fmt.Fprintln(output)
		flags.PrintDefaults()
		fmt.Fprintln(output, "\nAuthentication:")
		fmt.Fprintln(output, "  API: set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
		fmt.Fprintln(output, "  CLI: use your existing local claude or codex login.")
	}

	if err := flags.Parse(args); err != nil {
		return parsedOptions{}, err
	}
	if flags.NArg() != 0 {
		return parsedOptions{}, fmt.Errorf("unexpected positional argument %q", flags.Arg(0))
	}

	provider, err := parseProvider(*providerValue)
	if err != nil {
		return parsedOptions{}, err
	}
	flavor, err := parseFlavor(*flavorValue)
	if err != nil {
		return parsedOptions{}, err
	}
	if strings.TrimSpace(*model) == "" {
		return parsedOptions{}, errors.New("--model is required")
	}
	if strings.TrimSpace(*prompt) == "" {
		return parsedOptions{}, errors.New("--prompt is required")
	}
	if *maxTokens <= 0 {
		return parsedOptions{}, errors.New("--max-tokens must be greater than 0")
	}

	return parsedOptions{
		config: llmwrapper.Config{
			Provider: provider,
			Flavor:   flavor,
			Model:    *model,
		},
		request: llmwrapper.Request{
			Messages:  []llmwrapper.Message{llmwrapper.User(*prompt)},
			MaxTokens: *maxTokens,
		},
		stream: *stream,
	}, nil
}

func parseProvider(value string) (llmwrapper.Provider, error) {
	switch value {
	case string(llmwrapper.ProviderClaude):
		return llmwrapper.ProviderClaude, nil
	case string(llmwrapper.ProviderOpenAI):
		return llmwrapper.ProviderOpenAI, nil
	case "":
		return "", errors.New("--provider is required")
	default:
		return "", fmt.Errorf("invalid --provider %q: use claude or openai", value)
	}
}

func parseFlavor(value string) (llmwrapper.Flavor, error) {
	switch value {
	case string(llmwrapper.FlavorAPI):
		return llmwrapper.FlavorAPI, nil
	case string(llmwrapper.FlavorCLI):
		return llmwrapper.FlavorCLI, nil
	case "":
		return "", errors.New("--flavor is required")
	default:
		return "", fmt.Errorf("invalid --flavor %q: use api or cli", value)
	}
}
