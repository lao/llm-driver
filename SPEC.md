# Spec: llmwrapper multi-provider API

## Objective

Rename the Go package to `llmwrapper` and expose one small, provider-neutral API
that can switch between four execution targets through configuration only:

| Provider | Flavor | Transport |
| --- | --- | --- |
| `claude` | `api` | Anthropic Messages API through the official Go SDK |
| `claude` | `cli` | Local `claude -p` process |
| `openai` | `api` | OpenAI Responses API through the official Go SDK |
| `openai` | `cli` | Local `codex exec` process |

The primary user is a Go developer who wants to test with a locally
authenticated CLI and later use the corresponding hosted API—or switch
providers—without changing application call sites. Only the `Config` values for
provider, flavor, credentials, and model should change.

This is a clean package rename. The module path remains
`github.com/lao/claudewrap` until the repository is renamed; no compatibility
package or deprecated aliases for the former package are included.

### Public API

The library owns all public types. Neither provider SDK leaks through the public
boundary.

```go
client, err := llmwrapper.New(llmwrapper.Config{
	Provider: llmwrapper.ProviderOpenAI,
	Flavor:   llmwrapper.FlavorCLI,
	Model:    "gpt-5.6-sol",
})
if err != nil {
	return err
}

response, err := client.Generate(ctx, llmwrapper.Request{
	System:    "Answer concisely.",
	MaxTokens: 1024,
	Messages:  []llmwrapper.Message{llmwrapper.User("Hello")},
})
if err != nil {
	return err
}
fmt.Println(response.Text)
```

The intended public surface is deliberately small:

```go
type Config struct {
	Provider Provider
	Flavor   Flavor
	Model    string

	APIKey    string       // optional; API SDK environment defaults still work
	BaseURL   string       // optional proxy/test-server override
	HTTPClient *http.Client

	CLIPath string         // optional; defaults to claude or codex
	CLIArgs []string       // optional argv escape hatch; never shell-expanded
}

func New(Config) (*Client, error)
func (c *Client) Generate(context.Context, Request) (*Response, error)

type Request struct {
	System    string
	Messages  []Message
	MaxTokens int
}

type Message struct {
	Role Role
	Text string
}

func User(string) Message
func Assistant(string) Message

type Response struct {
	ID               string
	Model            string
	Text             string
	Usage            Usage
	CompletionReason CompletionReason
	Provider         Provider
	Flavor           Flavor
}

type CompletionReason string // stop, max_tokens, or refusal when reported

type Usage struct {
	InputTokens              int
	OutputTokens             int
	CachedInputTokens        int
	CacheCreationInputTokens int
	ReasoningTokens          int
}

type Error struct {
	Provider  Provider
	Flavor    Flavor
	Operation string
	Status    int
	Code      string
	Message   string
	Err       error
}
```

`Provider`, `Flavor`, and `Role` are validated string enums. `Config.Model` is
required so model choice remains explicit and never silently changes when a
provider updates an alias. `Request.Messages` must contain at least one
non-empty message and `MaxTokens` must be positive.

API keys may be supplied explicitly. If omitted, the official SDK convention is
used (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`). CLI flavors reuse each local
tool's existing authentication.

### Scope

In scope:

- Non-streaming, text-only generation.
- System instructions and multi-turn user/assistant transcripts.
- A per-request maximum output-token request. API flavors enforce/map it; CLI
  flavors accept the common request but may not provide an equivalent flag.
- Normalized response text, model, provider/flavor identity, optional response
  ID and completion reason, and available token usage.
- Consistent configuration, validation, executable-not-found, process, parsing,
  transport, and API errors.
- Configurable API base URL/HTTP client and CLI path/extra argv for testing,
  proxies, and nonstandard installations.
- A runnable command-line example selecting provider, flavor, model, and prompt.

Out of scope for this PR:

- Streaming.
- Tool/function calling or a tool-execution loop.
- Images, audio, files, web search, structured output, batches, or embeddings.
- Conversation/session persistence.
- Automatic fallback, routing, retries beyond SDK defaults, or model catalogs.
- Renaming the GitHub repository or module path; those changes must be
  coordinated separately.

CLI transports are agent CLIs rather than byte-for-byte API equivalents. The
implementation will use non-interactive, non-persistent modes and pass prompts
through stdin without invoking a shell. Their documented semantic limitations
will be visible in the README.

## Tech Stack

- Go 1.24, matching the current `go.mod`.
- `github.com/anthropics/anthropic-sdk-go` v1.58.1 (existing pinned dependency).
- `github.com/openai/openai-go/v3` v3.44.0 (latest release compatible with the
  approved Go 1.24 baseline; official SDK, Responses API).
- Go standard library for process execution, JSON/JSONL parsing, HTTP test
  servers, flags, and tests.

No framework, dependency-injection library, assertion library, or CLI framework
will be added.

## Commands

```bash
go mod tidy
gofmt -w .
go build ./...
go test ./...
go test -race ./...
go test -cover ./...
go vet ./...
go build ./examples/switchable

# Example: local Codex
go run ./examples/switchable \
  --provider openai \
  --flavor cli \
  --model gpt-5.6-sol \
  --prompt "Explain dependency inversion in one paragraph"

# Example: Anthropic API (reads ANTHROPIC_API_KEY)
go run ./examples/switchable \
  --provider claude \
  --flavor api \
  --model claude-sonnet-4-5 \
  --prompt "Explain dependency inversion in one paragraph"
```

Default tests must not contact a real provider or start a real agent CLI.

## Project Structure

```text
go.mod / go.sum                 module metadata and pinned SDKs
doc.go                          package documentation and short example
client.go                       Client, New, target selection, Generate
config.go                       Config and provider/flavor validation
types.go                        neutral request, response, message, usage types
errors.go                       normalized Error and sentinel errors
backend.go                      small unexported backend interface
backend_anthropic.go            Claude API adapter
backend_openai.go               OpenAI Responses API adapter
backend_claude_cli.go           Claude CLI adapter
backend_codex_cli.go            Codex CLI adapter and JSONL parser
backend_cli.go                  shared safe process runner/prompt rendering
*_test.go                       colocated unit and contract tests
examples/switchable/main.go     provider/flavor/model command-line example
README.md                       install, quick start, matrix, limitations
SPEC.md                         this approved source of truth
tasks/plan.md                   approved implementation plan
tasks/todo.md                   dependency-ordered task checklist
```

Adapters remain in one package because the package is small and they depend on
the same neutral types. Provider SDK imports are confined to their corresponding
adapter files.

## Code Style

Use idiomatic Go, explicit validation, short functions, and provider-specific
mapping helpers. Prefer direct code over generic factories or abstractions that
only have one caller.

```go
func (c *Client) Generate(ctx context.Context, request Request) (*Response, error) {
	if err := request.validate(); err != nil {
		return nil, err
	}

	response, err := c.backend.generate(ctx, request)
	if err != nil {
		return nil, err
	}
	response.Provider = c.config.Provider
	response.Flavor = c.config.Flavor
	return response, nil
}
```

Conventions:

- Export only caller-facing concepts; keep adapter and runner details private.
- Use `ProviderClaude`, `ProviderOpenAI`, `FlavorAPI`, and `FlavorCLI` constants.
- Wrap operational failures with provider, flavor, operation, status/exit code,
  and cause while preserving `errors.Is`/`errors.As` behavior.
- Build command argv directly and send user content through stdin. Never build a
  shell command string.
- Every exported declaration receives concise Go documentation.
- `gofmt` is the formatting authority.

## Testing Strategy

Tests use Go's `testing` package and stay beside the implementation.

1. Configuration tests cover all four valid provider/flavor pairs, invalid
   enum values, missing models, and incompatible/invalid option values.
2. Request tests cover empty transcripts, empty message text, invalid roles,
   non-positive token limits, and valid multi-turn requests.
3. API adapter tests use `httptest.Server` and the SDK base-URL override. They
   verify request mapping, authentication/config wiring, response mapping,
   usage, and normalized HTTP/API errors without external network access.
4. CLI adapter tests inject a fake command runner. They verify exact executable,
   argv, stdin transcript, Claude JSON parsing, Codex JSONL event parsing, usage,
   missing executables, non-zero exits, malformed output, and provider-reported
   failures.
5. A table-driven contract suite sends the same neutral request through all four
   adapters and asserts the same normalized text/usage/provider/flavor shape.
6. The switchable example must compile. Its argument parsing and validation will
   be kept thin enough to rely on library tests plus a build check.
7. Final verification runs build, unit tests, race tests, coverage reporting,
   vet, formatting checks, and a search for stale package/module identifiers.

No coverage percentage is mandated; meaningful branch and error-path coverage
is required. A high number that only exercises mapping happy paths is not a
substitute for the cases above.

## Boundaries

- **Always:** validate public inputs; keep public types provider-neutral; use
  official SDKs for API flavors; keep default tests offline and deterministic;
  pass CLI prompts through stdin; run the full verification suite before
  commits and before opening the PR; update this spec if a decision changes.
- **Ask first:** add compatibility shims; add streaming/tools or another
  provider; add non-official dependencies; change CI/workflow files; rename the
  GitHub repository; make a real billable API/CLI call.
- **Never:** commit credentials or local auth state; interpolate prompts into a
  shell command; silently ignore invalid provider/flavor values; remove failing
  tests to make verification green; overwrite unrelated user changes.

## Success Criteria

- The module path is `github.com/lao/claudewrap`, the root package is
  `llmwrapper`, docs/examples use the fetchable import path, and no active code
  declares the former package name.
- `New(Config)` accepts exactly the four documented provider/flavor pairs and
  returns clear validation errors for invalid configurations.
- One unchanged `Generate` call works with each target when only `Config`
  changes.
- Anthropic API requests use Messages; OpenAI API requests use Responses;
  Claude CLI uses `claude -p`; Codex CLI uses `codex exec`.
- All targets return the normalized response type and contextual normalized
  errors.
- The switchable example selects provider, flavor, and model from command-line
  flags and documents the required environment/local authentication.
- README and Go package docs include installation, one shared-API example, the
  four-target matrix, and honest CLI/API limitations.
- `go build ./...`, `go test ./...`, `go test -race ./...`, `go vet ./...`, and
  the formatting/stale-identifier checks pass locally without real credentials.
- The approved spec, plan, implementation, tests, and documentation are
  committed to a feature branch, pushed, and represented by a GitHub pull
  request against `main`.

## Open Questions

None. The four-target matrix and clean-break API were approved on 2026-07-31.
The module path remains `github.com/lao/claudewrap`; any future module-path
change must be coordinated with the GitHub repository rename so external
consumers can fetch it.
