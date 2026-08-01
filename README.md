# llmwrapper

`llmwrapper` is a small Go library with one text-generation API and four
switchable targets:

| Provider | Flavor | Target | Authentication |
| --- | --- | --- | --- |
| `claude` | `api` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `claude` | `cli` | Local `claude -p` | Existing Claude CLI login |
| `openai` | `api` | OpenAI Responses API | `OPENAI_API_KEY` |
| `openai` | `cli` | Local `codex exec` | Existing Codex CLI login |

Changing the provider, flavor, or model does not change application call sites.

## Install

```bash
go get github.com/lao/claudewrap
```

The module requires Go 1.24 or later.

## Use

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/lao/claudewrap"
)

func main() {
	client, err := llmwrapper.New(llmwrapper.Config{
		Provider: llmwrapper.ProviderOpenAI,
		Flavor:   llmwrapper.FlavorCLI,
		Model:    "gpt-5.6-sol",
	})
	if err != nil {
		log.Fatal(err)
	}

	response, err := client.Generate(context.Background(), llmwrapper.Request{
		System:    "Answer concisely.",
		Messages:  []llmwrapper.Message{llmwrapper.User("What is dependency inversion?")},
		MaxTokens: 1024,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(response.Text)
}
```

To switch targets, change only `Config`:

```go
// Claude API
llmwrapper.Config{Provider: llmwrapper.ProviderClaude, Flavor: llmwrapper.FlavorAPI, Model: "claude-sonnet-4-5"}

// Local Claude CLI
llmwrapper.Config{Provider: llmwrapper.ProviderClaude, Flavor: llmwrapper.FlavorCLI, Model: "claude-sonnet-4-5"}

// OpenAI API
llmwrapper.Config{Provider: llmwrapper.ProviderOpenAI, Flavor: llmwrapper.FlavorAPI, Model: "gpt-5.6-sol"}

// Local Codex CLI
llmwrapper.Config{Provider: llmwrapper.ProviderOpenAI, Flavor: llmwrapper.FlavorCLI, Model: "gpt-5.6-sol"}
```

The request and `Generate` call stay the same. Models are always explicit; the
library does not silently select or update one.

## Switchable CLI example

The included example takes the target and model from command-line flags:

```bash
go run ./examples/switchable \
  --provider openai \
  --flavor cli \
  --model gpt-5.6-sol \
  --prompt "Explain dependency inversion in one paragraph"
```

Run `go run ./examples/switchable --help` for all options. Use
`--provider claude --flavor api` (and a Claude model), for example, without
changing the example's generation code.

## Authentication and configuration

API targets use their official Go SDKs. Set `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`, or pass `Config.APIKey`. `Config.BaseURL` and
`Config.HTTPClient` are optional overrides for proxies, custom transports, and
tests. Treat custom endpoints and transports as trusted application
configuration: API credentials and prompts are sent through them.

CLI targets use the installed tool's existing login. `CLIPath` can select a
nonstandard executable and `CLIArgs` appends direct arguments. Prompts are sent
through stdin and no shell is invoked. Treat both CLI options as trusted
application configuration because they control local process execution.

Errors carry provider, flavor, operation, status, and provider code when those
values are available. Use `errors.Is` for sentinels such as
`ErrInvalidRequest` and `ErrExecutableNotFound`, and `errors.As` for `*Error`.

## Scope and CLI differences

The shared contract is non-streaming, text-only generation with system text,
multi-turn user/assistant messages, normalized text, and available usage.

The CLI flavors intentionally wrap agent CLIs, not byte-for-byte equivalents of
the hosted APIs:

- API flavors map and enforce `MaxTokens`. CLI flavors validate it as part of
  the portable request but have no reliable equivalent flag.
- Claude CLI runs without session persistence and in safe mode with agent
  tools disabled. Codex runs ephemerally with a read-only sandbox. Agent CLI
  behavior can still differ from a hosted model endpoint.
- System text is passed through Claude's system-prompt flag and Codex's
  per-invocation developer-instructions setting. Conversation text stays on
  stdin. System text may therefore be visible to local process inspection.
- CLI subprocesses inherit the application's working directory and environment
  so local authentication works. Codex's read-only sandbox prevents writes, not
  reads; isolate the host process when prompts are not trusted.
- Usage fields are populated only when a target reports them.

Streaming, tool/function calling, images, structured output, retries, automatic
fallback, and persisted conversations are outside this library's current scope.

## Develop

Default tests are offline: they use local HTTP test servers and injected fake
processes, never provider credentials or real agent CLIs.

```bash
go test ./...
go test -race ./...
go vet ./...
```

The CLI flavors' flags (`claude -p --output-format json ...` and
`codex exec --json ...`) cannot be validated by the offline suite because it
injects a fake process runner. An opt-in smoke test launches the real binaries
so a renamed or removed flag surfaces instead of failing only at runtime. It is
excluded by default and skips when a binary or model is missing:

```bash
CLAUDEWRAP_CLAUDE_CLI_MODEL=claude-sonnet-4-5 \
CLAUDEWRAP_CODEX_CLI_MODEL=gpt-5.6-sol \
  go test -tags integration -run 'CLIIntegration' .
```
