# claudewrap — Specification

> A Go library that mirrors the Anthropic **Messages API** surface behind a single
> interface, with **two interchangeable backends** selected at runtime:
> `sdk` (HTTP, via the official `anthropic-sdk-go`) and `cli` (shells out to a
> local `claude` Claude Code CLI). Same API, two flavors.

- **Module:** `github.com/lao/claudewrap`
- **Location:** `/Users/lucas/Projects/claudewrap`
- **Status:** spec — awaiting approval before full implementation
- **Go version:** 1.23

---

## 1. Objective

### Problem
Go developers who use Claude must choose up front between (a) the paid Anthropic
HTTP API (needs an API key, bills per token) and (b) the local Claude Code CLI
(uses an existing subscription/local auth, no key in code). Switching means
rewriting call sites.

### Goal
One Go interface, mirroring the Anthropic **Messages API**, backed by either
transport. Write the call once; pick the backend at construction time.

```go
c, _ := claude.New(claude.WithSDK(apiKey)) // HTTP to Anthropic
// or
c, _ := claude.New(claude.WithCLI())        // exec local `claude -p`

resp, err := c.Messages.New(ctx, claude.MessageParams{
    Model:     "claude-opus-4-8",
    MaxTokens: 1024,
    Messages:  []claude.Message{claude.UserMessage("Hello")},
})
```

### Target users
- Go engineers wanting to develop/test against the local CLI and ship against the API (or vice versa) without code changes.
- Cost-sensitive tools that prefer the CLI/subscription locally and the API in CI/prod.
- Anyone wanting a thin, typed, backend-agnostic Claude client in Go.

### Non-goals (v0.1)
- Not a full re-export of every `anthropic-sdk-go` feature (batches, files, MCP, managed agents). Scope = Messages: create + streaming, text + basic tool blocks.
- Not an agent framework (no built-in tool-execution loop). Higher-level `Query()` helper is explicitly deferred.
- CLI backend is **not** a byte-for-byte Messages endpoint — the CLI is agentic. See §7 CLI mapping & limitations.

---

## 2. Design decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | API surface | **Messages API mirror** — `Messages.New` + `Messages.NewStreaming` |
| 2 | Types | **Library-owned neutral types** — both backends map into them; `anthropic-sdk-go` types never appear in the public API |
| 3 | Flavor selection | **Runtime**, via functional options (`WithSDK` / `WithCLI`); both backends compiled in |
| 4 | Module path | `github.com/lao/claudewrap` |

---

## 3. Public API

### Construction
```go
func New(opts ...Option) (*Client, error)

// exactly one backend option is required
func WithSDK(apiKey string, o ...SDKOption) Option   // HTTP backend
func WithCLI(o ...CLIOption) Option                  // local `claude` CLI backend

// backend-scoped tuning
func WithSDKBaseURL(url string) SDKOption            // point at a test server / proxy
func WithSDKHTTPClient(h *http.Client) SDKOption
func WithCLIPath(path string) CLIOption              // default: "claude" on PATH
func WithCLIExtraArgs(args ...string) CLIOption      // escape hatch, appended verbatim

// client-wide
func WithDefaultModel(model string) Option           // default "claude-opus-4-8"
```

### Messages service
```go
type Client struct { Messages *MessagesService }

func (s *MessagesService) New(ctx context.Context, p MessageParams) (*Response, error)
func (s *MessagesService) NewStreaming(ctx context.Context, p MessageParams) (*Stream, error)
```

### Neutral types
```go
type Role string // "user" | "assistant"

type MessageParams struct {
    Model       string       // falls back to client default if ""
    MaxTokens   int          // required
    System      string       // optional system prompt
    Messages    []Message    // required, ≥1
    Temperature *float64     // optional
    StopSeqs    []string     // optional
    Tools       []Tool       // optional (best-effort; see limitations)
    Metadata    map[string]string
}

type Message struct {
    Role    Role
    Content []ContentBlock
}

type ContentBlock struct {
    Type  BlockType // "text" | "tool_use" | "tool_result"
    Text  string    // for text
    // tool fields populated for tool_use / tool_result
    ToolName  string
    ToolInput json.RawMessage
    ToolUseID string
}

type Response struct {
    ID         string
    Model      string
    Role       Role
    Content    []ContentBlock
    StopReason StopReason // "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" | "refusal"
    Usage      Usage
    Backend    string     // "sdk" | "cli" — which flavor served this
}

type Usage struct {
    InputTokens        int
    OutputTokens       int
    CacheReadTokens    int
    CacheCreationTokens int
}

// helpers
func UserMessage(text string) Message
func AssistantMessage(text string) Message
func TextBlock(text string) ContentBlock
func (r *Response) Text() string // concatenation of text blocks
```

### Streaming
```go
type Stream struct { /* ... */ }
func (s *Stream) Next() bool
func (s *Stream) Event() StreamEvent
func (s *Stream) Err() error
func (s *Stream) Response() (*Response, error) // accumulated final message
func (s *Stream) Close() error

type StreamEvent struct {
    Type  EventType // "text_delta" | "message_start" | "message_stop" | ...
    Text  string    // for text_delta
    Index int
}
```

### Errors
```go
type Error struct {
    Backend string // "sdk" | "cli"
    Op      string // "messages.new", ...
    Status  int    // HTTP status (sdk) or process exit code (cli), 0 if n/a
    Type    string // api error type (sdk) when available
    Message string
    Err     error  // wrapped cause
}
func (e *Error) Error() string
func (e *Error) Unwrap() error

var ErrNoBackend      = errors.New("claudewrap: no backend configured")
var ErrCLINotFound    = errors.New("claudewrap: claude CLI not found on PATH")
```

---

## 4. Architecture

```
                 ┌───────────────────────────┐
   caller ─────▶ │  claudewrap.Client        │  neutral types only
                 │    .Messages.New/Streaming │
                 └─────────────┬─────────────┘
                               │ backend interface (unexported)
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌──────────────────┐      ┌────────────────────────┐
        │  sdkBackend      │      │  cliBackend            │
        │  anthropic-sdk-go│      │  exec `claude -p`      │
        │  → HTTP           │      │  --output-format json  │
        └──────────────────┘      └────────────────────────┘
```

- The **backend interface** is unexported and lives in the root package:
  ```go
  type backend interface {
      messages(ctx context.Context, p MessageParams) (*Response, error)
      streamMessages(ctx context.Context, p MessageParams) (*Stream, error)
      name() string
  }
  ```
- Both implementations live in the **root package** (single package, no import
  cycle with the public types). `anthropic-sdk-go` is imported **only** in
  `backend_sdk.go`; nothing else touches it.
- The CLI backend’s process invocation goes through an injectable
  `commandRunner` func so tests never spawn a real process.

---

## 5. Project structure

```
claudewrap/
├── go.mod
├── go.sum
├── LICENSE
├── README.md
├── SPEC.md
├── .gitignore
├── doc.go              // package doc
├── claude.go           // Client, New(), option wiring, backend selection
├── options.go          // Option / SDKOption / CLIOption + config struct
├── types.go            // Message, ContentBlock, Response, Usage, enums, helpers
├── messages.go         // MessagesService, MessageParams
├── stream.go           // Stream, StreamEvent, accumulation
├── errors.go           // Error, sentinel errors
├── backend.go          // unexported backend interface
├── backend_sdk.go      // sdkBackend  (anthropic-sdk-go adapter)
├── backend_cli.go      // cliBackend  (claude CLI adapter + commandRunner)
├── examples/
│   ├── sdk/main.go
│   └── cli/main.go
├── claude_test.go
├── types_test.go
├── backend_sdk_test.go // httptest server, base-URL override
├── backend_cli_test.go // fake commandRunner
└── contract_test.go    // same assertions run against both backends
```

---

## 6. Commands

```bash
go build ./...              # compile
go test ./...               # unit tests (no network, no real CLI)
go test -tags integration ./...  # hits real API / real `claude` (opt-in, see §8)
go vet ./...
gofmt -l .                  # must be empty
golangci-lint run           # if installed
```

A `Makefile` provides: `make build test lint fmt vet examples`.

---

## 7. CLI backend mapping & limitations

`claude -p <prompt> --output-format json --model <model>` returns a JSON object
with the result text, `usage`, and cost/session metadata. Mapping:

| Neutral concept | CLI mapping |
|-----------------|-------------|
| `MessageParams.System` | `--append-system-prompt <system>` |
| `MessageParams.Messages` | Transcript flattened into the single `-p` prompt; the trailing user turn drives the request. Prompt passed via **args/stdin — never an interpolated shell string** |
| `MessageParams.Model` | `--model <model>` |
| streaming | `--output-format stream-json --verbose`, parsed line-by-line |
| `Response.Content` | assistant text → one text block |
| `Response.Usage` | mapped from the JSON `usage` object |
| `Response.StopReason` | best-effort (`end_turn`; `max_tokens` if truncated) |

**Documented v0.1 limitations (surfaced in godoc + README):**
- The CLI is agentic and may itself use tools/files; it is **not** a raw
  completion endpoint. Multi-turn fidelity is approximate — full transcript is
  flattened into one prompt.
- `MaxTokens`, `Temperature`, `StopSeqs`, and custom `Tools` may be ignored by
  the CLI where it has no equivalent flag; the library maps what it can and
  documents what it drops.
- Prompt caching fields in `Usage` are populated only when the CLI reports them.

Higher-fidelity multi-turn via `--input-format stream-json` on stdin is a
**post-v0.1** enhancement (listed in §9), not this scope.

---

## 8. Testing strategy

- **Unit, deterministic, default `go test`:**
  - `sdkBackend`: `httptest.Server` returning canned Messages responses;
    constructed with `WithSDKBaseURL` pointed at the server. No real network.
  - `cliBackend`: injected fake `commandRunner` returning canned CLI JSON /
    error exit codes. No real process.
  - Table-driven mapping tests: neutral → backend request, backend response → neutral.
- **Contract tests (`contract_test.go`):** one shared set of assertions run
  against both backends (each with its fake) to prove identical neutral-type
  output for equivalent inputs.
- **Integration (`-tags integration`, opt-in, skipped by default):**
  - SDK: requires `ANTHROPIC_API_KEY`; one cheap `claude-haiku-4-5` round-trip.
  - CLI: requires a real `claude` on PATH; skipped with a clear message if absent.
- Coverage target: ≥85% on mapping/backends. Error paths (bad status, non-zero
  exit, malformed JSON, missing binary) are explicitly tested.

---

## 9. Deferred / future (not v0.1)

- High-level `Query()` agent helper.
- Built-in tool-execution loop.
- Full multi-turn CLI via `--input-format stream-json`.
- Additional surfaces: batches, files, token counting, MCP.
- Compile-time build-tag flavor selection (currently runtime-only).

---

## 10. Boundaries

**Always**
- Keep the public API in **library-owned neutral types**; never leak
  `anthropic-sdk-go` types across the package boundary.
- Route both backends through the single unexported `backend` interface.
- Wrap all errors with `%w` inside a `*claudewrap.Error` carrying backend + op.
- Pass CLI prompts via `exec.Command` arg slices / stdin — **no shell string interpolation**.
- Default the model to `claude-opus-4-8` when unset.
- godoc on every exported symbol; `gofmt`-clean; context as first arg.
- Keep dependencies minimal: stdlib + `anthropic-sdk-go` only.

**Ask first**
- Adding any dependency beyond `anthropic-sdk-go`.
- Changing the shape of a public type after it is written.
- Implementing the deferred items in §9.
- Anything that runs the real CLI or real API as part of default `go test`.
- Publishing/pushing to a real remote or changing the module path.

**Never**
- Log, print, or embed API keys or prompt contents beyond what the caller asked.
- Build a CLI command by string-concatenating untrusted prompt text into a shell.
- Silently truncate inputs — surface an error instead.
- Auto-`go get` unpinned or run network installs without asking.
- Invent `claude` CLI flags that don’t exist — verify against the installed binary.
```

