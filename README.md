# llm-driver

`llm-driver` is a small Go library with one text-generation API and five
switchable targets:

| Provider | Flavor | Target | Authentication |
| --- | --- | --- | --- |
| `claude` | `api` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `claude` | `cli` | Local `claude -p` | Existing Claude CLI login |
| `openai` | `api` | OpenAI Responses API | `OPENAI_API_KEY` |
| `openai` | `cli` | Local `codex exec` | Existing Codex CLI login |
| `opencode` | `cli` | Local `opencode run` | Existing opencode login (many upstream providers) |

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

	llmdriver "github.com/lao/llm-driver"
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

The same loop works for all five targets. Chunk granularity is
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

// Local opencode CLI (any model from `opencode models`)
llmdriver.Config{Provider: llmdriver.ProviderOpenCode, Flavor: llmdriver.FlavorCLI, Model: "anthropic/claude-sonnet-4-5"}
```

The request and `Generate` call stay the same. API flavors require an explicit
model; CLI flavors may leave it empty to use the provider CLI's own default
(and the library never silently selects or updates one).

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

## opencode models

`opencode` fronts many upstream providers, so its model is a `provider/model`
id, for example `anthropic/claude-sonnet-4-5` or
`openrouter/anthropic/claude-sonnet-4.5#high` (a `#variant` pins the effort;
see Reasoning effort for how that interacts with `Request.Effort`). `ListModels` runs
`opencode models` and returns every id the installed opencode can actually
run — opencode resolves its own credentials, provider environment variables,
and `opencode.json` config:

```go
models, err := llmdriver.ListModels(ctx)
if err != nil {
	log.Fatal(err)
}
fmt.Println(models) // e.g. []string{"opencode/big-pickle", "anthropic/claude-sonnet-4-5", ...}

client, err := llmdriver.New(llmdriver.Config{
	Provider: llmdriver.ProviderOpenCode,
	Flavor:   llmdriver.FlavorCLI,
	Model:    models[0],
})
```

Passing a model id returned by `ListModels` is optional: `Config.Model` is
always a plain string, so any valid id works whether or not it was discovered.
`ListModels` returns an `*Error` when the `opencode` binary is missing or
fails.

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
`opencode` is CLI-only: its upstream providers and credentials are whatever the
local opencode is configured for.

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

| Feature | claude/api | openai/api | claude/cli | codex/cli | opencode/cli |
| --- | --- | --- | --- | --- | --- |
| System, messages, MaxTokens, streaming | ✓ | ✓ | ✓ | ✓ | ✓ |
| Structured output (JSON schema) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Reasoning effort | ✓ | ✓ | ✓ | ✓ | ✓ |
| Reasoning/thinking visibility | ✗ | ✗ | ✗ | ✗ | ✗ |
| Tool calling (caller-executed) | ✓ | ✓ | ✗ | ✗ | ✗ |
| Image input | ✗ | ✗ | ✗ | ✗ | ✗ |
| PDF/document input | ✗ | ✗ | ✗ | ✗ | ✗ |
| Temperature / TopP | ✓ | ✓ | ✗ | ✗ | ✗ |
| TopK | ✓ | ✗ | ✗ | ✗ | ✗ |
| Stop sequences | ✓ | ✗ | ✗ | ✗ | ✗ |
| Web search | ✗ | ✗ | ✗ | ✗ | ✗ |
| Metadata / end-user ID | ✓ | ✓ | ✗ | ✗ | ✗ |
| Conversation continuation by ID | ✗ | ✗ | ✗ | ✗ | ✗ |
| Token counting | ✗ | ✗ | ✗ | ✗ | ✗ |

Structured output and reasoning effort are supported on every target; the matrix
assumes a base `Config.Model` for `opencode/cli`, which rejects effort with an
empty model or one already carrying a `#variant`. Sampling
params (temperature/top-p) and the end-user ID map on both API flavors, with
top-k and stop sequences on `claude/api` only (the Responses API lacks both).
The remaining optional features are not yet implemented, and later releases flip
cells as each target gains support. `codex/cli` is the `openai` provider with the
`cli` flavor.

The `opencode/cli` structured-output mapping has no JSON-schema flag: it states
the schema in the prompt and validates the returned document is JSON, failing
with `invalid_structured_output` when it is not.

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
0600 temp file that is removed when the call completes). `opencode/cli` has no
schema flag, so it states the schema in the prompt and validates the returned
JSON the same way. Streaming works the same way: chunks stream the raw JSON text
and `StructuredOutput` is populated on the final `Response`.

Note: `openai/api` maps with `strict: true`, so a schema it accepts must set
`additionalProperties: false` and mark every property required. The same schema
may succeed on the other four targets and be rejected by OpenAI — write schemas
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
rejected there; the other targets accept the full set. Each target maps the level
natively: `claude/api` via `output_config.effort`, `openai/api` via
`reasoning.effort`, `claude/cli` via `--effort`, `codex/cli` via
`-c model_reasoning_effort`, and `opencode/cli` as the `#variant` suffix of the
model id (`opencode run --model provider/model#variant`). The opencode CLI
backend targets 2.x; the driver still recognizes the 1.x `name` error field for
compatibility.

opencode 2.x has no way to attach an effort to the CLI's default model, so an
effort with an empty `Config.Model` is rejected with `ErrUnsupportedFeature` —
set a model to use it. If `Config.Model` already contains a `#variant`,
`Request.Effort` is rejected for the same reason; put the desired variant in the
model and leave `Effort` empty. `Client.Supports(FeatureReasoningEffort)` accounts for this and
returns false for `opencode/cli` with an empty or `#variant`-carrying
`Config.Model`. When an effort is
applied, `Response.Model` is the effective id including the `#variant` suffix.
`Config.Model` should be a base `provider/model` id when you also set `Effort`.

**Breaking for opencode 1.x callers:** the driver targets 2.x and no longer
sends `--variant`. An effort with an empty `Config.Model` is now rejected with
`ErrUnsupportedFeature`; set a model and put the desired `#variant` in it to keep
reasoning control. With a model set, the effort is sent as `--model
provider/model#variant`, a form 1.x may not accept (likely a model-not-found
error), so 1.x installs must upgrade.

The example accepts a level via `--effort`:

```bash
go run ./examples/switchable \
  --provider claude --flavor api --model claude-sonnet-4-5 \
  --prompt "Prove there are infinitely many primes." \
  --effort high
```

## Tool calling

Offer caller-executed tools with `Request.Tools` and, optionally, constrain
selection with `Request.ToolChoice` (the zero value is auto; `ToolChoiceNone`,
`ToolChoiceRequired`, and `ToolChoiceTool` with a `Name` are the alternatives).
When the model calls a tool the response carries `CompletionReasonToolUse` and
one `ToolCall` per invocation (parallel calls preserved in order). You run each
call and feed the results back as `ToolResult` messages, echoing the model's
calls so the provider sees the conversation it produced:

```go
tools := []llmdriver.Tool{{
	Name:        "get_weather",
	Description: "Current weather for a city",
	InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
}}

messages := []llmdriver.Message{llmdriver.User("What's the weather in SF?")}
for {
	response, err := client.Generate(ctx, llmdriver.Request{
		Messages:  messages,
		MaxTokens: 1024,
		Tools:     tools,
	})
	if err != nil {
		return err
	}
	if response.CompletionReason != llmdriver.CompletionReasonToolUse {
		fmt.Println(response.Text)
		break
	}

	// Echo the assistant's tool calls, then append each tool's result.
	parts := make([]llmdriver.Part, len(response.ToolCalls))
	for i, call := range response.ToolCalls {
		parts[i] = llmdriver.ToolCallPart{ID: call.ID, Name: call.Name, Input: call.Input}
	}
	messages = append(messages, llmdriver.Message{Role: llmdriver.RoleAssistant, Parts: parts})
	for _, call := range response.ToolCalls {
		result := runTool(call.Name, call.Input) // your code
		messages = append(messages, llmdriver.ToolResult(call.ID, result, false))
	}
}
```

Streaming surfaces a tool call's JSON arguments incrementally as
`KindToolInput` chunks (`Chunk.Kind`); the parsed `ToolCalls` are on the final
`Response`. A `Message` sets exactly one of `Text` or `Parts` — the existing
`User`/`Assistant` helpers stay text-only, and tool echoes and results use
`Parts`. Tool calling is an API-flavor feature: `claude/api` and `openai/api`
map `tools`/`tool_choice` natively, while `claude/cli`, `codex/cli`, and
`opencode/cli` reject `Tools` or tool parts with `ErrUnsupportedFeature`.

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
  read-only sandbox; opencode runs non-interactively without `--auto`. Agent
  CLI behavior can still differ from a hosted model endpoint. An opencode turn
  that ends on a tool call **without returning text** — commonly a permission
  auto-reject — fails with code `tool_calls` (carrying the CLI's stderr
  diagnostic) instead of surfacing as a generic empty response. A turn that
  emits text and then ends on a tool call fails the same way, since opencode
  runs its tools itself and never returns `ToolCalls`. A turn the provider stopped on its output cap
  reports `CompletionReasonMaxTokens` on the response, and carries
  `Reason: "length"` on the error when the cap was reached before any text.
  opencode 2.x omits the terminal `step_finish` for a plain text answer. By
  default, a turn whose completion cannot be verified fails with code
  `unverified_completion`, even if the process exits successfully. Set
  `Config.AcceptUnverifiedOpenCode` to true to accept such answers with
  `CompletionReasonUnverified`; they may be truncated and cannot report
  `CompletionReasonMaxTokens`. A text-only answer without any `step_finish`
  has zero usage. Security-relevant consumers must require
  `CompletionReasonStop` before approval or other actions; consumers accepting
  unverified answers must validate completeness themselves. Usage and a reason
  are present only when opencode emits a `step_finish` (for example between tool calls).
- System text is passed through Claude's system-prompt flag and Codex's
  per-invocation developer-instructions setting. opencode has no system flag, so
  system text is prefixed to the prompt. Conversation text stays on stdin. System
  text may therefore be visible to local process inspection.
- CLI subprocesses inherit the application's working directory and environment
  so local authentication works. Codex's read-only sandbox prevents writes, not
  reads; isolate the host process when prompts are not trusted.
- CLI error messages can include stderr even after exit 0. Messages are
  terminal-sanitized and capped at 500 bytes, but secrets, prompt fragments, and
  paths are not redacted. Filter or suppress error text before logging or
  forwarding it across trust boundaries.
- Usage fields are populated only when a target reports them.
- Streaming chunk granularity is target-defined, like `MaxTokens`. The hosted
  APIs and `claude -p --output-format stream-json --include-partial-messages`
  produce token-sized deltas. `codex exec --json` publishes agent messages per
  item on some versions, and `opencode run --format json` publishes a text part
  per message, so a whole answer can arrive as one large chunk.

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

The CLI flavors' flags (`claude -p --output-format json|stream-json ...`,
`codex exec --json ...`, and `opencode run --format json ...`) cannot be
validated by the offline suite because it injects a fake process runner. An
opt-in smoke test launches the real binaries so a renamed or removed flag
surfaces instead of failing only at runtime. It is excluded by default and skips
when a binary or model is missing:

```bash
LLM_DRIVER_CLAUDE_CLI_MODEL=claude-sonnet-4-5 \
LLM_DRIVER_CODEX_CLI_MODEL=gpt-5.6-sol \
LLM_DRIVER_OPENCODE_CLI_MODEL=anthropic/claude-sonnet-4-5 \
  go test -tags integration -run 'CLIIntegration' .
```
