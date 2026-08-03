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
The implementation is a clean break. It does not retain deprecated constructors,
Messages-service aliases, content-block types, or the old package name. The
existing module path remains unchanged.

## Baseline and Migration Constraints

- The branch starts from a working two-target prototype: Anthropic SDK and
  Claude CLI, both non-streaming.
- The existing public API is intentionally replaced, not wrapped.
- Renaming every root Go package declaration is one unavoidable atomic,
  mechanical change. It spans more files than a normal task but contains no
  behavioral redesign and receives its own review checkpoint.
- Default tests stay offline: API tests use `httptest.Server`; CLI tests use an
  injected process runner.
- The GitHub remote and the committed Go module path both remain
  `github.com/lao/llmwrapper`, matching the repository so external consumers can
  fetch it; the root package is also named `llmwrapper`.

## Architecture

```text
caller
  |
  v
Client.Generate(ctx, Request)
  |-- validate Config at construction
  |-- validate Request before adapter invocation
  |
  v
backend.generate(ctx, Request)  (unexported interface)
  |
  +-- claude/api  ---> anthropicBackend ---> Anthropic Messages API
  +-- claude/cli  ---> claudeCLIBackend --> `claude -p` JSON
  +-- openai/api  ---> openAIBackend -----> OpenAI Responses API
  `-- openai/cli  ---> codexCLIBackend ----> `codex exec --json` JSONL
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

```
gofmt -l .                    # empty

Provider SDK imports occur only in their adapter files. CLI implementations
depend on a shared unexported command runner and prompt renderer. Public code
never imports a provider SDK or exposes subprocess types.

### Target mapping

| Target | Request mapping | Result mapping |
| --- | --- | --- |
| Claude/API | Messages API system block, structured turns, `max_tokens`, configured model | Text blocks joined; message ID/model; neutral completion reason; input/output/cache usage |
| OpenAI/API | Responses API instructions, structured input turns, `max_output_tokens`, configured model | Output text/refusal; response ID/model; neutral completion reason; input/output/cached/reasoning usage |
| Claude/CLI | Prompt transcript on stdin; `-p --output-format json --permission-mode default --model`; system via CLI flag; single-shot print mode | JSON result/session ID/usage; non-success subtype becomes normalized error |
| OpenAI/CLI | Transcript on stdin; system via per-invocation `developer_instructions`; `codex exec --json --sandbox read-only --skip-git-repo-check --model ... -` | Final completed agent-message item; turn-completed usage; failed/error events become normalized errors |

CLI `MaxTokens` has no reliable equivalent and is accepted but not translated.
This limitation is documented, not silently presented as exact parity.

## Dependency-Ordered Increments

### Increment 1 — Package identity and neutral contract

Atomically rename the root package while retaining the module path, remove the
old public surface, and install the approved neutral types, validation, errors,
client, and backend selection.
Use a tiny fake backend to prove validation occurs before delegation and that
response provider/flavor metadata is normalized centrally.

Dependencies: none.

Acceptance checkpoint:

- The root package compiles only as `llmwrapper`; `go.mod` retains the fetchable
  `github.com/lao/llmwrapper` module path.
- All four provider/flavor pairs resolve to an internal target; invalid enums,
  blank models, and invalid requests fail before adapter work.
- `errors.Is`/`errors.As` work for validation and wrapped operational errors.
- No compatibility API remains.

Verification:

```bash
go test ./... -run 'Test(Config|Request|Client|Error)'
go test ./...
go vet ./...
```

### Increment 2 — Claude API vertical slice

Adapt the existing Anthropic implementation to the new contract. Map system
instructions, multi-turn text, model, and max tokens into Messages; normalize
text, IDs, model, all supported usage fields, and SDK errors.

Dependencies: Increment 1.

Acceptance checkpoint:

- A neutral `Generate` call reaches an `httptest.Server` through the official
  Anthropic SDK and returns the approved neutral response.
- Request body and error mapping tests cover happy and non-2xx paths.
- No Anthropic type appears in an exported declaration.

Verification:

```bash
go test ./... -run 'TestAnthropic'
go test ./...
go vet ./...
```

### Increment 3 — OpenAI API vertical slice

Add the official OpenAI Go SDK and implement the Responses API adapter. Map the
same neutral system/transcript/model/token request and normalize output text,
IDs, usage (including cached and reasoning tokens), and SDK errors.

Dependencies: Increment 1. It may be implemented after Increment 2 without
coupling to the Anthropic adapter.

Acceptance checkpoint:

- A neutral `Generate` call reaches an `httptest.Server` through OpenAI
  Responses and returns the same neutral response shape.
- Tests assert the request uses Responses rather than Chat Completions.
- OpenAI SDK unions and errors are confined to the adapter.

Verification:

```bash
go test ./... -run 'TestOpenAI'
go test ./...
go vet ./...
```

### Checkpoint A — Both API flavors

- Both API slices pass independently and together.
- Request/response/error mappings receive a focused diff review.
- `go test -race ./...` is green before process adapters are added.

### Increment 4 — Shared process boundary and Claude CLI slice

Extract a shared safe runner and transcript renderer, then migrate the current
Claude CLI implementation. Prompts remain stdin bytes; configuration contributes
argv only; no shell string is ever constructed.

Dependencies: Increment 1.

Acceptance checkpoint:

- Exact executable, argv, and stdin are tested for single-turn, system, and
  multi-turn requests.
- Success JSON, provider-reported error JSON, malformed JSON, missing binary,
  launch failure, and non-zero exit are normalized.
- The default runner is the only code that touches `os/exec`.

Verification:

```bash
go test ./... -run 'Test(ClaudeCLI|Prompt|Command)'
go test ./...
go vet ./...
```

### Increment 5 — Codex CLI vertical slice

Implement non-interactive Codex execution using JSONL output. Parse completed
agent-message items as response text and turn completion as usage. Treat failed
turns, error events, missing final messages, malformed JSONL, process failures,
and missing executables as contextual errors.

Dependencies: Increment 4 for the runner/renderer; Increment 1 for target
selection.

Acceptance checkpoint:

- Exact `codex exec` argv and stdin are covered by fake-runner tests.
- Multiple progress/reasoning events do not leak into final response text.
- Cached input and reasoning output usage map to neutral fields.
- Parsing has no dependency on a locally installed Codex binary.

Verification:

```bash
go test ./... -run 'TestCodexCLI'
go test ./...
go vet ./...
```

### Checkpoint B — Both CLI flavors

- Claude and Codex fake-process suites cover success and all material error
  paths.
- No test invokes a real executable.
- Shell-injection safety is evident from the runner signature and argv/stdin
  assertions.
- `go test -race ./...` is green.

### Increment 6 — Four-target contract suite

Build one table-driven suite over all four target factories. Give every fake
transport an equivalent upstream result and assert identical normalized text,
model, usage, provider/flavor metadata, validation behavior, and error context.

Dependencies: Increments 2–5.

Acceptance checkpoint:

- Every target passes the same request/response assertions.
- Only documented transport differences are excluded, with comments explaining
  why.
- Construction tests prove all and only the four approved pairs.

Verification:

```bash
go test ./... -run 'TestContract'
go test -race ./...
go test -cover ./...
```

### Increment 7 — Switchable example and documentation

Add `examples/switchable`, package docs, and a concise README. The example parses
`--provider`, `--flavor`, `--model`, `--prompt`, and `--max-tokens`, constructs
one `Config`, and makes one unchanged `Generate` call. Document API-key
environment variables, local CLI auth, the target matrix, and CLI semantic
limitations.

Dependencies: Increment 6.

Acceptance checkpoint:

- The example builds and `--help` explains all required flags.
- README quick starts cover at least one API and one CLI configuration while
  sharing the same request code.
- Every exported declaration has Go documentation.

Verification:

```bash
go build ./examples/switchable
go run ./examples/switchable --help
go test ./...
go vet ./...
```

### Increment 8 — Cleanup, final audit, and PR

Remove obsolete files/types/tests, tidy dependencies, format, and audit every
success criterion against current evidence. Review the complete diff, commit
cohesive changes, push `codex/llmwrapper`, and open a PR against `main` that
references the spec and reports verification results.

Dependencies: Increments 1–7.

Acceptance checkpoint:

- No active code or documentation declares the old package identifier, uses the
  uncommitted `github.com/lao/llmwrapper` module path, or retains removed API names.
- All required commands pass from a clean working tree without credentials.
- The remote branch and GitHub PR exist and match the verified commit.

Verification:

```bash
test -z "$(gofmt -l .)"
go mod tidy
git diff --exit-code -- go.mod go.sum
go build ./...
go vet ./...
go test ./...
go test -race ./...
go test -cover ./...          # error paths covered, not only happy paths
go build ./examples/switchable
git grep -n "claudewrap" -- '*.go' '*.md'   # no stale identifiers in active code
rg -n 'package claudewrap|github.com/lao/claudewrap' --glob '*.go' --glob 'go.mod' --glob '*.md' --glob '!tasks/**'
rg -n '^module github.com/lao/llmwrapper$' go.mod
git diff --check origin/main...HEAD
```

## Notes for the implementer

- Consult the pinned SDK sources in the module cache for exact streaming
  event type names (anthropic-sdk-go v1.58.1, openai-go/v3 v3.44.0) instead
  of guessing; both ship `ssestream` helpers.
- Do not modify the batch `commandRunner` contract or existing tests.
- Codex delta event names vary by CLI version; the parser must tolerate
  unknown event types silently, as the batch parser does.
## Sequencing and Parallelization

- Increment 1 is strictly first because every adapter depends on the neutral
  contract.
- API adapters (2 and 3) are logically independent after Increment 1.
- CLI work begins with the shared runner in Increment 4; Codex then reuses it in
  Increment 5.
- Contract tests, docs/example, and final cleanup remain sequential because they
  integrate every preceding slice.
- Work will be executed sequentially in this session. The logical independence
  above is documented for review and future maintenance, not used to introduce
  coordination overhead.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Public rename/API replacement creates a wide diff | High | Separate mechanical identity changes from adapter behavior in commits and review the stale-identifier search |
| OpenAI SDK Responses unions are easy to map incorrectly | High | Pin Go-1.24-compatible v3.44.0, inspect actual types with `go doc`, and assert captured wire JSON through `httptest` |
| SDK base-URL semantics differ | Medium | Give each adapter its own test-server setup and assert request path/method |
| CLI output schemas drift between installed versions | High | Parse only documented fields/events, ignore unknown JSON fields, and cover malformed/missing terminal events |
| CLI tools are agentic, unlike hosted text endpoints | Medium | Use non-interactive single-shot modes, read-only Codex sandbox, stdin prompts, and explicit README limitations |
| Usage fields do not align one-to-one | Medium | Keep a small superset of neutral counters and leave unavailable counters zero |
| CLI max tokens cannot be enforced consistently | Low | Accept the shared request, document the limitation, and never claim exact transport parity |
| Module path must match the canonical repository to stay fetchable | Medium | Keep the fetchable `github.com/lao/llmwrapper` module path in code/docs, matching both the repository and the `llmwrapper` package name |

## Verification Checkpoints

At every increment:

```bash
gofmt -w .
go test ./...
go vet ./...
git diff --check
```

At Checkpoints A and B, and before the PR:

```bash
go build ./...
go test -race ./...
go test -cover ./...
```

No implementation increment is complete when a required command fails.

## Open Questions

None. Implementation begins only after this plan is approved and the detailed
task checklist is generated and approved.
