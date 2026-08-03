# Plan: llmwrapper streaming support

**Status:** approved 2026-08-03 (autonomous run)

**Source of truth:** [`SPEC.md`](../SPEC.md) (streaming). Base API stays
byte-for-byte unchanged.

Each slice is vertical (one complete path), lands with its tests, and must
leave `gofmt -l .` empty and `go build ./... && go vet ./... && go test ./...`
green before the next slice starts.

## Dependency graph

```
S1 core Stream + backend seam ──► S2 Anthropic API stream (first full path)
S1 ─► S3 OpenAI API stream
S1 ─► S4 streaming CLI runner ──► S5 Claude CLI stream
                               └─► S6 Codex CLI stream
S2,S3,S5,S6 ─► S7 streaming contract suite
S7 ─► S8 example --stream + docs
S8 ─► S9 final verification
```

## Slices

### S1 — Core streaming surface + backend seam (proved by S2)

Files: `stream.go`, `stream_test.go`, `backend.go`, `client.go` (additive only).

- Add to `backend`: `generateStream(context.Context, Request) (chunkStream, error)`
  where `chunkStream` is a small unexported interface: `recv() (Chunk, error)`
  (typed done sentinel), `finish() (*Response, error)`, `close() error`.
  Adapters do provider mapping; `Stream` does accumulation once.
- Public `Stream` (Next/Current/Err/Response/Close per SPEC) wraps a
  `chunkStream`: accumulates text, suppresses empty chunks, stamps
  Provider/Flavor/default Model exactly like `Client.Generate`, normalizes
  errors with operation `"generate_stream"`.
- `Client.GenerateStream`: same request validation as `Generate`.
- Unit tests with a scripted fake `chunkStream`: chunk order, `Response`
  before exhaustion errors, `Response.Text == concat(chunks)`, empty-chunk
  suppression, mid-stream error in `Err()` and `Response()`, double `Close`,
  `Close` mid-stream stops iteration.

Acceptance: `go test ./... -run 'Stream'` green; all four backends compile
(temporary "not implemented" streaming stubs on CLI backends allowed only
until S4–S6 replace them within this same PR — none survive to review).

### S2 — Anthropic API streaming

Files: `backend_anthropic.go`, `backend_anthropic_test.go`.

- `Messages.NewStreaming` with the same request mapping as `generate`.
- `content_block_delta`/`text_delta` → chunks; usage, stop reason, message ID
  accumulated from message events (SDK `Message.Accumulate` helper acceptable
  if it keeps mapping smaller).
- Normalize SSE/API errors like the batch path (status, code, message).
- Tests: `httptest.Server` SSE fixtures — happy path with usage, multi-block
  text, mid-stream error event, malformed SSE, non-200 start.

Checkpoint: first full streaming path works end-to-end offline — reassess
seam design before replicating to three more adapters.

### S3 — OpenAI API streaming

Files: `backend_openai.go`, `backend_openai_test.go`.

- `Responses.NewStreaming`; chunks from `response.output_text.delta`; final
  `response.completed` event supplies ID/usage/status via the batch mapping
  helpers.
- Same test matrix as S2 with Responses SSE fixtures.

### S4 — Streaming CLI subprocess runner

Files: `backend_cli.go`, `backend_cli_test.go` (reuse existing process-group
files; extend only if a gap is found).

- Add `streamingCommandRunner`: like `commandRunner` but returns stdout as an
  `io.ReadCloser` plus a `wait()` reporting exit code/stderr after EOF/close.
  Default impl: `exec.CommandContext`, existing `configureCommand`
  process-group setup, `WaitDelay`, stdin bytes, bounded stderr buffer, kill
  on close/cancel.
- Shared line reader with the existing 16 MiB bound for both CLI adapters.
- Tests: fake-runner injection; real-process hygiene test (helper-process
  pattern from liveness tests) proving close/cancel terminates the child;
  oversized line → normalized error.

Acceptance: no goroutine/process leak under `go test -race -run 'CLI'`.

### S5 — Claude CLI streaming

Files: `backend_claude_cli.go`, `backend_claude_cli_test.go`.

- argv: `-p --output-format stream-json --include-partial-messages --verbose
  --permission-mode default --model <m>` (+ system flag, + user args), same
  stdin transcript as batch.
- Parse stream-json lines: `stream_event` content_block_delta text → chunks;
  final `result` line reuses existing result mapping (success/error, usage,
  session ID). Missing result line, malformed line, unsuccessful result, and
  early exit map to normalized errors.
- Tests mirror batch tests plus incremental delivery (chunks observable
  before the fake process finishes), argv assertion, error results.

### S6 — Codex CLI streaming

Files: `backend_codex_cli.go`, `backend_codex_cli_test.go`.

- argv unchanged from batch (`exec --json ...`); parsing becomes incremental:
  reuse `codexCLIEvent`; chunks from agent-message delta events when present,
  else from `item.completed` agent_message; `turn.failed`/`error`/
  missing-completion handling identical to batch.
- Text seen via deltas must not be duplicated by the item's final
  `item.completed` (track emitted length per item).
- Tests: delta path, completed-only path, mixed, failure events, nonzero exit
  with diagnostic (batch parity), argv.

Checkpoint: all four adapters stream — run full race suite before contract
work.

### S7 — Streaming contract suite

Files: `contract_test.go`.

- Table-driven: same neutral request through all four adapters (SSE fixtures
  + fake streaming runners); assert identical normalized shape: chunk concat
  == `Response.Text`, provider/flavor stamped, usage populated, operation
  `"generate_stream"` on injected failures.

### S8 — Example + docs

Files: `examples/switchable/main.go`, `README.md`, `doc.go`.

- `--stream` flag: print chunks as they arrive, then the usage summary the
  batch path prints; default path unchanged.
- README: move streaming out of the out-of-scope list; add a streaming
  example and CLI granularity caveats; `doc.go` short streaming mention.

### S9 — Final verification (checkpoint before ship)

```bash
gofmt -l .                    # empty
go build ./...
go vet ./...
go test ./...
go test -race ./...
go test -cover ./...
go build ./examples/switchable
```

Provider SDK imports occur only in their adapter files. CLI implementations
depend on a shared unexported command runner and prompt renderer. Public code
never imports a provider SDK or exposes subprocess types.

## Notes for the implementer

- Consult the pinned SDK sources in the module cache for exact streaming
  event type names (anthropic-sdk-go v1.58.1, openai-go/v3 v3.44.0) instead
  of guessing; both ship `ssestream` helpers.
- Do not modify the batch `commandRunner` contract or existing tests.
- Codex delta event names vary by CLI version; the parser must tolerate
  unknown event types silently, as the batch parser does.
