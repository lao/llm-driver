# Implementation Plan: llmwrapper multi-provider API

**Status:** approved on 2026-07-31

**Source of truth:** [`SPEC.md`](../SPEC.md)

## Overview

Replace the current Claude-specific constructor and Anthropic-shaped public API
with a small `llmwrapper.Config` + `Client.Generate` contract. Four internal
adapters implement the same request/response boundary: Anthropic API, Claude
CLI, OpenAI Responses API, and Codex CLI. The work proceeds as vertical slices,
proving one target end-to-end at a time while keeping SDK and process details
private.

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
- The GitHub remote remains `github.com/lao/claudewrap` until the separate
  repository setting is renamed. The committed Go module path remains
  `github.com/lao/claudewrap`; only the root package is renamed to `llmwrapper`.

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

### Public boundary

- `Config` selects a required `Provider`, `Flavor`, and `Model`.
- `Client.Generate` accepts `Request{System, Messages, MaxTokens}`.
- `Response` exposes only neutral text, identity, completion reason,
  provider/flavor, and `Usage`.
- `Error` records provider, flavor, operation, HTTP status/process exit code,
  provider code, message, and wrapped cause.
- `ErrInvalidConfig`, `ErrInvalidRequest`, and `ErrExecutableNotFound` allow
  stable `errors.Is` checks without multiplying target-specific sentinels.

### Dependency direction

```text
neutral types + validation + errors
             |
             v
      Client + backend interface
       /        |         |       \
Anthropic API OpenAI API Claude CLI Codex CLI
       \        |         |       /
             contract tests
                   |
             example + docs
```

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
  `github.com/lao/claudewrap` module path.
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
go test ./...
go test -race ./...
go test -cover ./...
go vet ./...
go build ./examples/switchable
rg -n 'package claudewrap|github.com/lao/llmwrapper' --glob '*.go' --glob 'go.mod' --glob '*.md' --glob '!tasks/**'
rg -n '^module github.com/lao/claudewrap$' go.mod
git diff --check origin/main...HEAD
```

The stale-identifier `rg` command must produce no matches; the module-path
command must match the single declaration in `go.mod`.

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
| Package and module names intentionally differ until repository rename | Medium | Keep the fetchable `github.com/lao/claudewrap` module path in code/docs and distinguish it from package `llmwrapper` in verification |

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
