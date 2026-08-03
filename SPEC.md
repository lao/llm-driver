# Spec: llmwrapper streaming support

## Objective

Add streaming text generation to `llmwrapper` so a caller can consume
incremental text deltas — and then the same normalized final `Response` that
`Generate` returns — from any of the four existing targets without changing
call sites when the target changes:

| Provider | Flavor | Streaming transport |
| --- | --- | --- |
| `claude` | `api` | Anthropic Messages streaming (SSE) via the official Go SDK |
| `claude` | `cli` | `claude -p --output-format stream-json --include-partial-messages` |
| `openai` | `api` | OpenAI Responses streaming (SSE) via the official Go SDK |
| `openai` | `cli` | `codex exec --json` JSONL events read incrementally |

The primary user is the same Go developer as the base library: they prototype
against a locally authenticated CLI, then switch `Config` to a hosted API. A
streaming call written once must work against all four targets; only chunk
granularity may differ, and that difference is documented, not hidden.

The base non-streaming API (`New`, `Generate`, `Request`, `Response`, `Error`)
is stable and must not change shape or behavior.

### Public API additions

The library keeps owning all public types. Neither provider SDK leaks through
the boundary. The streaming surface mirrors both official SDKs' iterator style:

```go
stream, err := client.GenerateStream(ctx, llmwrapper.Request{
	System:    "Answer concisely.",
	MaxTokens: 1024,
	Messages:  []llmwrapper.Message{llmwrapper.User("Hello")},
})
if err != nil {
	return err
}
defer stream.Close()

for stream.Next() {
	fmt.Print(stream.Current().Text)
}

response, err := stream.Response()
if err != nil {
	return err
}
fmt.Println(response.Usage.OutputTokens)
```

New public surface (exact, deliberately small):

```go
// GenerateStream validates the request and starts one streaming generation.
func (c *Client) GenerateStream(context.Context, Request) (*Stream, error)

// Chunk is one incremental piece of response text.
type Chunk struct {
	Text string
}

// Stream iterates text chunks and terminates in a normalized Response.
type Stream struct{ /* unexported */ }

func (s *Stream) Next() bool             // advance; false at end or on error
func (s *Stream) Current() Chunk         // valid after Next() returned true
func (s *Stream) Err() error             // terminal stream error, if any
func (s *Stream) Response() (*Response, error) // final accumulated response
func (s *Stream) Close() error           // release resources; idempotent, safe concurrently with Next
```

Contract:

- `GenerateStream` performs the same `Request` validation as `Generate` and
  returns the same normalized `*Error` values for invalid requests and
  startup failures (bad config pairs are impossible past `New`).
- `Next`/`Current`/`Err` follow the standard Go iterator idiom (as in both
  official SDKs and `bufio.Scanner`): `Next` returns false at normal end or
  on error; `Err` distinguishes the two. `Chunk.Text` is never empty.
- `Response()` returns the same fields `Generate` would have produced: text
  equal to the concatenation of all yielded chunks, plus model, usage,
  completion reason, ID, provider, and flavor as reported by the target.
  Calling it before the stream is exhausted returns a normalized `*Error`;
  after a stream error it returns that error.
- `Close()` aborts an in-flight stream (terminating a CLI subprocess or the
  HTTP response body), is idempotent, and never panics. A stream abandoned
  after `Close` without exhaustion leaves no goroutine or process behind.
- Context cancellation terminates the stream; `Err()` then reports a
  normalized error wrapping `context.Canceled`/`DeadlineExceeded`.
- Mid-stream provider failures (SSE error event, CLI `turn.failed`, nonzero
  exit, malformed output) surface through `Err()`/`Response()` as the same
  normalized `*Error` shape used by `Generate`, including provider, flavor,
  operation `"generate_stream"`, and status/code when available.
- `Generate` behavior is byte-for-byte unchanged.

### Per-target semantics

- **Anthropic API**: SDK Messages streaming; chunks from `text_delta` events;
  usage/stop reason accumulated from message start/delta events.
- **OpenAI API**: SDK Responses streaming; chunks from output-text delta
  events; usage/ID/completion state from the completed-response event.
- **Claude CLI**: `-p --output-format stream-json --include-partial-messages
  --verbose` (stream-json requires verbose in print mode); chunks from
  `stream_event` content-block text deltas; final `result` event supplies
  usage, session ID, and error results exactly as the batch parser does.
- **Codex CLI**: `exec --json` events parsed line-by-line as the process
  writes them; chunks from agent-message items (delta events when the
  installed CLI emits them, otherwise per completed item). Usage and thread
  ID from `turn.completed`/`thread.started`.

Chunk granularity is target-defined: hosted APIs and Claude CLI produce
token-ish deltas; Codex CLI may produce few, large chunks. The README states
this as a semantic difference of the CLI flavors, like `MaxTokens`.

CLI streaming requires reading stdout incrementally, so the existing batch
`commandRunner` is complemented by a streaming runner in `backend_cli.go`
that exposes the subprocess stdout as an `io.ReadCloser` with the same
safety properties as today: argv built directly, prompt on stdin, no shell,
bounded line sizes, process-group termination on cancel/close, `WaitDelay`.
Fake streaming runners keep default tests offline.

### Scope

In scope:

- Streaming text-only generation on all four provider/flavor targets.
- The `Stream`/`Chunk` public API above; final accumulated `Response` parity
  with `Generate`.
- Streaming-capable process running for CLI targets with clean cancellation
  and no zombie processes.
- Normalized mid-stream and startup errors.
- A `--stream` flag on `examples/switchable` printing chunks as they arrive.
- README and package documentation updates (streaming moves from the
  out-of-scope list to a documented feature with honest CLI caveats).

Out of scope for this PR:

- Tool/function calling, images, structured output, retries, fallback,
  conversation persistence (unchanged from base spec).
- Streaming usage/heartbeat metadata beyond the final `Response`.
- Changing the non-streaming path, module path, or CI workflows.
- New dependencies: only Go stdlib plus the already-pinned SDKs.

## Tech Stack

- Go 1.24 (current `go.mod`).
- `github.com/anthropics/anthropic-sdk-go` v1.58.1 — `Messages.NewStreaming`
  and its `ssestream` package (already a dependency).
- `github.com/openai/openai-go/v3` v3.44.0 — `Responses.NewStreaming`
  (already a dependency).
- Go standard library (`os/exec`, `bufio`, `encoding/json`, `iter` not
  required) for CLI streaming and tests. No new dependencies.

## Commands

```bash
go mod tidy
gofmt -l .            # must print nothing
go build ./...
go vet ./...
go test ./...
go test -race ./...
go test -cover ./...
go build ./examples/switchable

# Streaming example: local Claude CLI
go run ./examples/switchable \
  --provider claude --flavor cli --model claude-sonnet-4-5 \
  --stream --prompt "Explain dependency inversion in one paragraph"
```

Default tests must not contact a real provider or start a real agent CLI.

## Project Structure

New/changed files only; everything else is untouched:

```text
stream.go                       Stream, Chunk, accumulation, GenerateStream plumbing
backend.go                      backend interface gains a streaming method
backend_anthropic.go            + Messages streaming adapter
backend_openai.go               + Responses streaming adapter
backend_claude_cli.go           + stream-json incremental parser
backend_codex_cli.go            + incremental JSONL event streaming
backend_cli.go                  + streaming subprocess runner (pipe-based)
stream_test.go                  Stream contract and accumulation tests
*_test.go                       per-adapter streaming tests beside existing ones
contract_test.go                + streaming contract cases across all four targets
examples/switchable/main.go     + --stream flag
README.md / doc.go              streaming docs and caveats
SPEC.md                         this spec
tasks/plan.md / tasks/todo.md   implementation plan and checklist
```

Adapters stay in one package; provider SDK imports stay confined to their
adapter files.

## Code Style

Unchanged from the base spec: idiomatic Go, explicit validation, short
functions, direct code over single-caller abstractions, documented exports,
`gofmt` authoritative. Streaming-specific additions:

- One shared chunk-accumulator used by all four adapters so `Response`
  parity with `Generate` is written once.
- No goroutine leaks: every goroutine started by a stream exits on `Close`,
  context cancellation, or exhaustion; verify with `-race` tests.
- Wrap operational failures exactly as `Generate` does, with operation
  `"generate_stream"`.

## Testing Strategy

Beside the implementation, `testing` package only:

1. Stream contract tests: chunk order, `Current` stability, `Err` vs normal
   end, `Response` before exhaustion, `Response` text == concatenated chunks,
   double `Close`, `Close` mid-stream, context cancellation.
2. Anthropic/OpenAI adapter tests: `httptest.Server` emitting SSE fixtures
   through the SDK base-URL override — happy path, usage mapping, mid-stream
   error events, malformed SSE, HTTP-level errors.
3. Claude CLI streaming tests: fake streaming runner emitting stream-json
   lines incrementally — deltas, final result mapping, error results,
   malformed lines, early process exit, argv assertion (including
   `--include-partial-messages`).
4. Codex CLI streaming tests: fake streaming runner emitting JSONL — item
   deltas/completions, `turn.failed`, `error`, missing completion, oversized
   lines, argv assertion.
5. Contract suite: the same streaming request through all four adapters
   asserts identical normalized chunk/response shape.
6. Process hygiene: fake and real (`os/exec` with a helper binary or `sh`-free
   equivalent) tests proving Close/cancel terminates the subprocess and no
   goroutines leak (`-race`, goroutine-count guard).
7. Example compiles; README commands stay accurate.

Meaningful error-path coverage is required; happy-path-only coverage is not
acceptance.

## Boundaries

- **Always:** validate public inputs; keep public types provider-neutral;
  keep default tests offline/deterministic; prompts via stdin, never a shell;
  keep `Generate` behavior unchanged; run the full verification suite before
  commits and the PR; update this spec if a decision changes.
- **Ask first:** new dependencies; changing the batch `commandRunner`
  contract; adding streaming beyond text deltas; touching CI files; real
  billable API/CLI calls.
- **Never:** commit credentials; leak provider SDK types publicly; leave
  orphan subprocesses/goroutines; remove failing tests to go green; break
  the existing public API.

## Success Criteria

- One unchanged `GenerateStream` loop works against all four targets when
  only `Config` changes.
- `stream.Response()` equals what `Generate` would report (text, usage,
  completion reason, IDs) for the same underlying provider output.
- Anthropic uses Messages streaming; OpenAI uses Responses streaming; Claude
  CLI uses stream-json with partial messages; Codex CLI parses JSONL
  incrementally — all verified by offline tests.
- Close/cancel terminates CLI subprocesses and leaks no goroutines under
  `-race`.
- `examples/switchable --stream` prints incrementally and compiles in CI-less
  local verification.
- `gofmt`, `go build`, `go vet`, `go test`, `go test -race` all pass
  offline.
- Spec, plan, implementation, tests, and docs land on a feature branch with
  a GitHub PR against `main`.

## Open Questions

None blocking. Codex CLI delta-event support varies by installed version;
the parser accepts both delta events and whole-item completion, and the
integration smoke test (opt-in, as today) validates real-binary flags.
