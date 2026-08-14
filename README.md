# llm-driver

`llm-driver` is a small Go library with one text-generation API and four
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
go get github.com/lao/llm-driver
```

The module requires Go 1.24 or later.

## Use

```go
package main

import (
	"context"
	"fmt"
	"log"

	llm-driver "github.com/lao/llm-driver"
)

func main() {
	client, err := llmdriver.New(llmdriver.Config{
		Provider: llmdriver.ProviderOpenAI,
		Flavor:   llmdriver.FlavorCLI,
		Model:    "gpt-5.6-sol",
	})
	if err != nil {
		log.Fatal(err)
	}

	response, err := client.Generate(context.Background(), llmdriver.Request{
		System:    "Answer concisely.",
		Messages:  []llmdriver.Message{llmdriver.User("What is dependency inversion?")},
		MaxTokens: 1024,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(response.Text)
}
```

## Stream

`GenerateStream` runs the same request incrementally and ends in the same
normalized `Response`:

```go
stream, err := client.GenerateStream(context.Background(), llmdriver.Request{
	System:    "Answer concisely.",
	Messages:  []llmdriver.Message{llmdriver.User("What is dependency inversion?")},
	MaxTokens: 1024,
})
if err != nil {
	log.Fatal(err)
}
defer stream.Close()

for stream.Next() {
	fmt.Print(stream.Current().Text)
}

response, err := stream.Response()
if err != nil {
	log.Fatal(err)
}
fmt.Println(response.Usage.OutputTokens)
```

`Next` reports false at the end of the stream and on failure; `Err`
distinguishes the two. `Response` returns the final result once the stream is
exhausted, with text equal to the concatenation of every chunk. `Close` aborts
an in-flight stream, is idempotent, and may be called while another goroutine
is inside `Next`. Cancelling the context terminates the stream.

The same loop works for all four targets. Chunk granularity is
target-defined and is not normalized: see the CLI differences below.

To switch targets, change only `Config`:

```go
// Claude API
llmdriver.Config{Provider: llmdriver.ProviderClaude, Flavor: llmdriver.FlavorAPI, Model: "claude-sonnet-4-5"}

// Local Claude CLI
llmdriver.Config{Provider: llmdriver.ProviderClaude, Flavor: llmdriver.FlavorCLI, Model: "claude-sonnet-4-5"}

// OpenAI API
llmdriver.Config{Provider: llmdriver.ProviderOpenAI, Flavor: llmdriver.FlavorAPI, Model: "gpt-5.6-sol"}

// Local Codex CLI
llmdriver.Config{Provider: llmdriver.ProviderOpenAI, Flavor: llmdriver.FlavorCLI, Model: "gpt-5.6-sol"}
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

Add `--stream` to print the response as it arrives:

```bash
go run ./examples/switchable \
  --provider claude --flavor cli --model claude-sonnet-4-5 \
  --stream --prompt "Explain dependency inversion in one paragraph"
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

## Capabilities

Beyond the baseline surface (system text, messages, `MaxTokens`, streaming),
each optional generation feature is either supported on a target or rejected
with `ErrUnsupportedFeature` — the library never silently degrades. `Generate`
and `GenerateStream` validate a request against the selected target before
invoking it, and `Client.Supports(feature)` answers the same question ahead of
time.

`✓` = supported; `✗` = rejected with `ErrUnsupportedFeature` naming the feature
and target.

| Feature | claude/api | openai/api | claude/cli | codex/cli |
| --- | --- | --- | --- | --- |
| System, messages, MaxTokens, streaming | ✓ | ✓ | ✓ | ✓ |
| Structured output (JSON schema) | ✓ | ✓ | ✓ | ✓ |
| Reasoning effort | ✓ | ✓ | ✓ | ✓ |
| Reasoning/thinking visibility | ✗ | ✗ | ✗ | ✗ |
| Tool calling (caller-executed) | ✗ | ✗ | ✗ | ✗ |
| Image input | ✗ | ✗ | ✗ | ✗ |
| PDF/document input | ✗ | ✗ | ✗ | ✗ |
| Temperature / TopP | ✗ | ✗ | ✗ | ✗ |
| TopK | ✗ | ✗ | ✗ | ✗ |
| Stop sequences | ✗ | ✗ | ✗ | ✗ |
| Web search | ✗ | ✗ | ✗ | ✗ |
| Metadata / end-user ID | ✗ | ✗ | ✗ | ✗ |
| Conversation continuation by ID | ✗ | ✗ | ✗ | ✗ |
| Token counting | ✗ | ✗ | ✗ | ✗ |

Structured output and reasoning effort are supported on every target; the
remaining optional features are not yet implemented, and later releases flip
cells as each target gains support. `codex/cli` is the `openai` provider with the
`cli` flavor.

## Structured output

Set `Request.OutputSchema` to constrain the response to a JSON document matching
a JSON Schema. The result is returned on `Response.StructuredOutput` (the
JSON-validated bytes), while `Response.Text` keeps the raw model output.

```go
response, err := client.Generate(ctx, llmdriver.Request{
	Messages:  []llmdriver.Message{llmdriver.User("Extract the year from: shipped in 1998.")},
	MaxTokens: 256,
	OutputSchema: &llmdriver.OutputSchema{
		Name:   "extraction",
		Schema: json.RawMessage(`{"type":"object","properties":{"year":{"type":"integer"}},"required":["year"]}`),
	},
})
if err != nil {
	log.Fatal(err)
}
fmt.Println(string(response.StructuredOutput)) // e.g. {"year":1998}
```

`Schema` must be a JSON object; an invalid schema fails with `ErrInvalidRequest`
before any target is invoked. `Name` is required by the OpenAI mapping and
defaulted when empty; other targets ignore it. If the model returns output that
is not valid JSON, the call fails with a normalized `*Error`.

Each target maps the schema natively: `claude/api` via `output_config.format`,
`openai/api` via `text.format` json_schema (strict), `claude/cli` via
`--json-schema`, and `codex/cli` via `--output-schema` (written to a private
0600 temp file that is removed when the call completes). Streaming works the
same way: chunks stream the raw JSON text and `StructuredOutput` is populated on
the final `Response`.

Note: `openai/api` maps with `strict: true`, so a schema it accepts must set
`additionalProperties: false` and mark every property required. The same schema
may succeed on the other three targets and be rejected by OpenAI — write schemas
to OpenAI's strict subset for portable requests.

The example accepts a schema file via `--schema`:

```bash
go run ./examples/switchable \
  --provider claude --flavor api --model claude-sonnet-4-5 \
  --prompt "Extract the year from: shipped in 1998." \
  --schema ./schema.json
```

## Reasoning effort

Set `Request.Effort` to ask the model for more or less reasoning. The neutral
enum is the Anthropic-style superset — `EffortLow`, `EffortMedium`,
`EffortHigh`, `EffortXHigh`, `EffortMax` — and the zero value leaves effort
unset (today's behavior).

```go
response, err := client.Generate(ctx, llmdriver.Request{
	Messages:  []llmdriver.Message{llmdriver.User("Prove there are infinitely many primes.")},
	MaxTokens: 1024,
	Effort:    llmdriver.EffortHigh,
})
```

Values are provider-validated, never remapped: each target rejects a value it
does not accept with `ErrUnsupportedFeature` naming the value and target.
`openai/api` tops out at `EffortHigh`, so `EffortXHigh` and `EffortMax` are
rejected there; the other three accept the full set. Each target maps the level
natively: `claude/api` via `output_config.effort`, `openai/api` via
`reasoning.effort`, `claude/cli` via `--effort`, and `codex/cli` via
`-c model_reasoning_effort`.

The example accepts a level via `--effort`:

```bash
go run ./examples/switchable \
  --provider claude --flavor api --model claude-sonnet-4-5 \
  --prompt "Prove there are infinitely many primes." \
  --effort high
```

## Scope and CLI differences

The shared contract is text-only generation, batch or streaming, with system
text, multi-turn user/assistant messages, normalized text, and available
usage.

The CLI flavors intentionally wrap agent CLIs, not byte-for-byte equivalents of
the hosted APIs:

- API flavors map and enforce `MaxTokens`. CLI flavors validate it as part of
  the portable request but have no reliable equivalent flag.
- Claude CLI runs in single-shot print mode with the default permission mode,
  so a headless run cannot approve agent tool actions. Codex runs with a
  read-only sandbox. Agent CLI behavior can still differ from a hosted model
  endpoint.
- System text is passed through Claude's system-prompt flag and Codex's
  per-invocation developer-instructions setting. Conversation text stays on
  stdin. System text may therefore be visible to local process inspection.
- CLI subprocesses inherit the application's working directory and environment
  so local authentication works. Codex's read-only sandbox prevents writes, not
  reads; isolate the host process when prompts are not trusted.
- Usage fields are populated only when a target reports them.
- Streaming chunk granularity is target-defined, like `MaxTokens`. The hosted
  APIs and `claude -p --output-format stream-json --include-partial-messages`
  produce token-sized deltas. `codex exec --json` publishes agent messages per
  item on some versions, so a whole answer can arrive as one large chunk.

Tool/function calling, images, retries, automatic fallback, and persisted
conversations are not yet implemented; requesting one currently fails with
`ErrUnsupportedFeature` (see the capability matrix above).

## Develop

Default tests are offline: they use local HTTP test servers and injected fake
processes, never provider credentials or real agent CLIs.

```bash
go test ./...
go test -race ./...
go vet ./...
```

The CLI flavors' flags (`claude -p --output-format json|stream-json ...` and
`codex exec --json ...`) cannot be validated by the offline suite because it
injects a fake process runner. An opt-in smoke test launches the real binaries
so a renamed or removed flag surfaces instead of failing only at runtime. It is
excluded by default and skips when a binary or model is missing:

```bash
LLM_DRIVER_CLAUDE_CLI_MODEL=claude-sonnet-4-5 \
LLM_DRIVER_CODEX_CLI_MODEL=gpt-5.6-sol \
  go test -tags integration -run 'CLIIntegration' .
```
