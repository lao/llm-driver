# Plan: llm-driver SDK generation parity

**Status:** draft — pending approval

**Source of truth:** [`SPEC.md`](../SPEC.md) (SDK generation parity).
Supersedes the shipped streaming plan (git history of this file).

Delivery unit is one phase = one PR (spec roadmap). Inside a phase, slices
are vertical: each lands with its tests and leaves `gofmt -l .` empty and
`go build ./... && go vet ./... && go test ./... && go test -race ./...`
green. Phases 1–5 are the committed core; 6–9 are sketched and get their own
detailed plans when reached.

## Dependency graph

```
P1 capability framework
 ├─► P2 structured output (all 4 targets)
 ├─► P3 reasoning effort + richer results (all 4 targets)
 ├─► P4 sampling params + UserID (API flavors)
 └─► P5 tool calling (API flavors; introduces Message.Parts)
        └─► P6 multimodal input (extends Parts)
P1..P5 ─► core checkpoint (parity core done)
P7 client robustness   (independent of P2–P6; needs only P1)
P8 conversation continuation (needs P1)
P9 web search          (needs P1)
```

P2–P4 are mutually independent after P1 and could be parallelized across
sessions; P5 must precede P6 (both shape `Message.Parts`).

## Architecture decisions

- **One capability table** (`internal/core/capabilities.go`) drives request
  validation, `Client.Supports`, the README matrix, and the contract tests.
  Feature gates never appear as scattered `if flavor ==` checks in adapters.
- **Feature detection from the request**: `featuresOf(Request) []Feature`
  computes which features a request exercises; validation intersects that
  with the target's row. Adding a feature = one constant, one detection case,
  four table cells.
- **Granular features**: `FeatureTopK` and `FeatureStopSequences` are
  separate from `FeatureSamplingParams` because openai/api lacks both (the
  Responses API has no stop sequences).
- **`Message.Parts` arrives in P5**, not P6: tool results need content parts
  anyway (`ToolResultPart`), so P5 introduces the union
  (`TextPart`/`ToolCallPart`/`ToolResultPart`) and P6 only appends
  (`ImagePart`/`DocumentPart`). `Message.Text` stays as the fast path;
  exactly one of `Text`/`Parts` is set.
- **Effort values are provider-validated, not translated**: the neutral
  `Effort` enum is the Anthropic-style superset (`low…max`); each adapter
  rejects values its provider lacks (openai/api tops out at `high`) via a
  per-target allowed set in the capability table, so mismatches fail loudly
  instead of being remapped.
- **Streaming + structured output**: chunks stay raw text deltas of the JSON
  document; `Response.StructuredOutput` is populated at `Finish` from the
  accumulated text. No new chunk kind needed in P2.

## Task list

### Phase 1 — Capability framework (PR 1)

#### T1.1 Capability table + sentinel (S)

**Description:** Add `Feature` string type with constants for every spec
inventory row, `ErrUnsupportedFeature` sentinel, and the per-target table
mapping (provider, flavor) → supported feature set. Only
already-implemented features (system/messages/max-tokens/streaming) are
present in all four rows initially; later phases fill cells they implement.

- Acceptance:
  - [ ] Table covers all four targets and every `Feature` constant.
  - [ ] `errors.Is(err, ErrUnsupportedFeature)` works; error text names the
        feature and target.
- Verification: `go test ./internal/core -run 'Capabilit|Unsupported'`.
- Dependencies: none.
- Files: `internal/core/capabilities.go`, `internal/core/errors.go`,
  `internal/core/capabilities_test.go`.

#### T1.2 Request feature detection + gate in Generate paths (S)

**Description:** `featuresOf(Request)` + a `ValidateRequestFor(target)` step
run by `Generate` and `GenerateStream` after existing validation. With
today's request fields it always passes (no new fields exist yet), proven by
a test that a future-feature stub is rejected.

- Acceptance:
  - [ ] Both entry points gate through the table.
  - [ ] Existing test suite untouched and green (behavioral no-op today).
- Verification: full suite + `-race`.
- Dependencies: T1.1.
- Files: `internal/core/types.go`, `internal/core/backend.go`, `client.go`,
  `client_test.go`.

#### T1.3 Public surface + docs (S)

**Description:** Root aliases (`Feature`, constants, `ErrUnsupportedFeature`),
`Client.Supports(Feature) bool`, README capability matrix section, `doc.go`
note.

- Acceptance:
  - [ ] `Supports` answers straight from the table.
  - [ ] README matrix rows match table contents (asserted by a test walking
        the table, or manually cross-checked and noted in PR).
- Verification: `go build ./... && go vet ./... && go test ./...`.
- Dependencies: T1.1.
- Files: `llmdriver.go`, `client.go`, `README.md`, `doc.go`.

### Checkpoint — Phase 1
- [ ] Full suite + `-race` green; behavior of existing requests unchanged.
- [ ] Human review of `Feature` granularity before adapters build on it.

### Phase 2 — Structured output, all four targets (PR 2)

#### T2.1 Core types + validation (S)

**Description:** `OutputSchema{Name string, Schema json.RawMessage}`,
`Request.OutputSchema *OutputSchema`, `Response.StructuredOutput
json.RawMessage`, `FeatureStructuredOutput` detection, validation (schema
must parse as a JSON object; name required for openai mapping — default it).
Root aliases.

- Acceptance:
  - [ ] Invalid schema JSON → `ErrInvalidRequest`; unsupported target →
        `ErrUnsupportedFeature` (none yet — all four will support it, cell
        flipped per adapter task).
- Verification: `go test ./... -run 'Schema|StructuredOutput'`.
- Dependencies: P1.
- Files: `internal/core/types.go`, `internal/core/capabilities.go`,
  `llmdriver.go`, `internal/core/request_test.go`.

#### T2.2 Anthropic API mapping (S)

**Description:** Map `OutputSchema` to the SDK's structured-output request
parameter (`output_config.format` json_schema). Consult the pinned
anthropic-sdk-go in the module cache first; if the pinned version lacks the
parameter, stop and ask (SDK bump = ask-first boundary). Populate
`StructuredOutput` from the response text after validating it is JSON.
Batch + streaming.

- Acceptance:
  - [ ] Request JSON asserted via httptest fixture; response carries both
        `Text` (raw) and `StructuredOutput` (same bytes, JSON-validated).
  - [ ] Non-JSON model output → normalized error, operation preserved.
- Verification: `go test ./... -run 'Anthropic.*Schema'` + race.
- Dependencies: T2.1.
- Files: `internal/backends/anthropic.go`, `anthropic_test.go` (root),
  `internal/core/capabilities.go` (flip cell).

#### T2.3 OpenAI API mapping (S)

**Description:** `text.format` json_schema with `strict: true`. Same
acceptance shape as T2.2 with Responses fixtures, batch + streaming.

- Dependencies: T2.1. Files: `internal/backends/openai.go`,
  `openai_test.go`, capability cell.

#### T2.4 Claude CLI mapping (S)

**Description:** Append `--json-schema <inline-json>` to argv; parse result
text as the JSON document. Integration smoke extended to assert the flag is
accepted by the real binary (open question 1 in SPEC resolves here — if the
pinned CLI rejects inline JSON, fall back to a temp file and update SPEC).

- Acceptance:
  - [ ] argv assertion; fake-runner batch + stream tests; malformed output →
        normalized error.
- Dependencies: T2.1. Files: `internal/backends/claude_cli.go`,
  `claude_cli_test.go`, `cli_integration_test.go`, capability cell.

#### T2.5 Codex CLI mapping (S)

**Description:** Write schema to a temp file (0600, scratch dir, removed on
close — including early-error paths), pass `--output-schema <path>`. Batch +
stream tests, argv assertion, temp-file lifecycle test, integration smoke.

- Dependencies: T2.1. Files: `internal/backends/codex_cli.go`,
  `codex_cli_test.go`, `cli_integration_test.go`, capability cell.

#### T2.6 Contract suite + docs (S)

**Description:** Table-driven contract test: same `OutputSchema` request
through all four targets asserts identical normalized shape (batch and
stream). README: structured output section + matrix update; example gains
`--schema` flag reading a schema file.

- Dependencies: T2.2–T2.5. Files: `contract_test.go`,
  `examples/switchable/main.go`, `README.md`.

### Checkpoint — Phase 2
- [ ] All four targets produce `StructuredOutput` offline; full suite +
      `-race` green; PR review before P3.

### Phase 3 — Reasoning effort + richer results (PR 3)

#### T3.1 Effort type + per-target allowed sets (S)

**Description:** `Effort` enum (`low`/`medium`/`high`/`xhigh`/`max`),
`Request.Effort`, feature detection, per-target allowed-value sets in the
capability table (openai/api: `low…high`; others: full set — verify against
pinned SDKs/CLI help during implementation). Out-of-set value →
`ErrUnsupportedFeature` naming the value.

- Files: `internal/core/types.go`, `capabilities.go`, `llmdriver.go`, tests.

#### T3.2 Adapter mappings (M)

**Description:** anthropic/api `output_config.effort`; openai/api
`reasoning.effort`; claude/cli `--effort <v>`; codex/cli
`-c model_reasoning_effort=<v>`. Request-mapping assertions per adapter,
batch + stream.

- Dependencies: T3.1. Files: four adapter files + their tests (touched
  mechanically; ~8 files but one line-item each — acceptable as it is one
  concern).

#### T3.3 Richer completion mapping (S)

**Description:** Add `CompletionReasonToolUse` and
`CompletionReasonContentFilter` constants now (used by P5; OpenAI
`incomplete_details.reason: content_filter` and Anthropic `stop_reason`
mappings wired where already observable). Document that unknown provider
reasons pass through empty as today.

- Dependencies: none beyond P1. Files: `internal/core/types.go`, adapter
  mapping helpers, tests.

#### T3.4 Contract + docs (S)

**Description:** Contract row for effort across all four; README matrix +
effort section; example `--effort` flag.

### Checkpoint — Phase 3
- [ ] Effort verified against real binaries via opt-in smoke; suite green;
      review.

### Phase 4 — Sampling params + UserID, API flavors (PR 4)

#### T4.1 Core fields (S)

**Description:** `Temperature *float64`, `TopP *float64`, `TopK *int`,
`StopSequences []string`, `UserID string` on `Request`; distinct features
(`FeatureSamplingParams`, `FeatureTopK`, `FeatureStopSequences`,
`FeatureUserID`); range validation (temperature bounds are
provider-specific — validate only "non-negative, finite", pass through
otherwise per SPEC note ¹).

- Files: `internal/core/types.go`, `capabilities.go`, `llmdriver.go`, tests.

#### T4.2 Anthropic mapping (S)

**Description:** temperature/top_p/top_k/stop_sequences/`metadata.user_id`.
Provider rejections (e.g. newest models reject temperature) surface as the
existing normalized API error — asserted with a 400 fixture.
`stop_reason: stop_sequence` keeps mapping to `CompletionReasonStop`
(documented).

- Files: `internal/backends/anthropic.go`, `anthropic_test.go`, cells.

#### T4.3 OpenAI mapping (S)

**Description:** temperature/top_p/`safety_identifier`. `FeatureTopK` and
`FeatureStopSequences` stay unsupported (cells remain ✗) — contract test
asserts `ErrUnsupportedFeature`.

- Files: `internal/backends/openai.go`, `openai_test.go`, cells.

#### T4.4 Contract + docs (S)

**Description:** Contract rows: supported params on both API flavors; CLI
flavors reject each param; openai rejects TopK/StopSequences. README matrix.

### Checkpoint — Phases 1–4
- [ ] Full suite + `-race` + example build green; human review before P5
      (P5 reshapes `Message`).

### Phase 5 — Tool calling, API flavors (PR 5)

#### T5.1 Core tool types + Message.Parts (M)

**Description:** `Tool{Name, Description, InputSchema json.RawMessage}`,
`ToolChoice` (auto/none/required/named), `ToolCall{ID, Name, Input
json.RawMessage}` on `Response`, `Part` union with
`TextPart`/`ToolCallPart`/`ToolResultPart`, `Message.Parts` (exactly one of
`Text`/`Parts`; assistant tool-call echoes and user tool results expressed
as parts), helper `ToolResult(id string, content string, isErr bool)
Message`. Validation: tool names unique/non-empty, schemas valid JSON,
`ToolChoice` named tool must exist, parts non-empty.

- Acceptance:
  - [ ] Existing `Text`-only messages unchanged (backward-compat tests).
  - [ ] `FeatureTools` detection covers tools, tool choice, and part usage.
- Verification: `go test ./internal/core -run 'Tool|Part'`.
- Files: `internal/core/types.go`, `capabilities.go`, `llmdriver.go`,
  `internal/core/request_test.go`.

#### T5.2 Anthropic mapping (M)

**Description:** tools + tool_choice into SDK request; `tool_use` response
blocks → `ToolCalls` + `CompletionReasonToolUse`; `ToolResultPart`/
`ToolCallPart` messages → SDK content blocks. Streaming: `input_json_delta`
→ `Chunk{Kind: KindToolInput}` (additive `Chunk.Kind`, zero value
`KindText` — existing consumers unaffected); accumulated tool calls on
`Finish`.

- Acceptance:
  - [ ] Full round-trip fixture: request → tool_use → tool_result follow-up
        request asserted; parallel tool calls preserved in order.
- Files: `internal/backends/anthropic.go`, `anthropic_test.go`,
  `backend_anthropic_stream_test.go`, cells.

#### T5.3 OpenAI mapping (M)

**Description:** function tools + tool_choice into Responses request;
`function_call` output items → `ToolCalls`; function_call_output input items
from `ToolResultPart`; streaming `function_call_arguments.delta` →
`KindToolInput`. Same acceptance shape as T5.2.

- Files: `internal/backends/openai.go`, `openai_test.go`,
  `backend_openai_stream_test.go`, cells.

#### T5.4 Contract, CLI rejection, docs (S)

**Description:** Contract rows: identical normalized tool round-trip on both
API flavors; both CLI flavors reject `Tools`/parts with
`ErrUnsupportedFeature`. README tool-calling section with loop example +
matrix; `doc.go` note.

- Dependencies: T5.2, T5.3. Files: `contract_test.go`, `stream_test.go`,
  `README.md`, `doc.go`.

### Checkpoint — Core parity done (after P5)
- [ ] Every P1–P5 feature: implemented or `ErrUnsupportedFeature`, no third
      state (contract suite is the proof).
- [ ] `Supports` agrees with runtime behavior (table-driven test).
- [ ] Pre-parity requests byte-identical on all four targets.
- [ ] Human review; decide P6–P9 order confirmation.

### Phase 6–9 — sketches (detailed plans when reached)

- **P6 multimodal input:** `ImagePart` (bytes+media type or path) /
  `DocumentPart`; claude/api + openai/api blocks; codex/cli `--image` temp
  file; claude/cli rejects. Depends on P5 parts.
- **P7 client robustness:** `Config.MaxRetries`/`Config.Timeout`; SDK
  option mapping for API flavors; shared backoff wrapper for CLI flavors
  (retry only on retryable classes; never mid-stream); request IDs onto
  `Error` where available.
- **P8 conversation continuation:** `Request.ContinueFrom` — openai/api
  `previous_response_id` (+`store`), claude/cli `--resume <session>`;
  claude/api + codex/cli reject; interaction with `Messages` validation
  (openai: ContinueFrom + new turn only).
- **P9 web search:** `WebSearchOption`; anthropic server tool, openai
  built-in tool, claude/cli `--allowedTools WebSearch`, codex/cli
  `--search`; widens CLI subprocess abilities → ask-first gate before
  implementation (SPEC boundary).

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Pinned SDKs lack a needed param (structured output shape, effort) | High | First task of each adapter slice inspects the module cache; SDK bump is ask-first and isolated to that PR |
| Claude CLI `--json-schema` shape drift (inline vs file) | Med | Resolved in T2.4 via integration smoke; temp-file fallback documented |
| Effort value sets differ per provider | Med | Per-target allowed sets in capability table; loud rejection, no remapping |
| `Message.Parts` breaks existing callers | High | `Text` stays primary; exactly-one validation; backward-compat tests in T5.1 |
| Tool streaming accumulation complexity | Med | `Chunk.Kind` additive with zero-value `KindText`; accumulation logic lives once in core stream, mirroring S1 seam |
| Codex temp-file leak on error paths | Low | Lifecycle owned by the ChunkStream/Generate call; test asserts removal on every exit path |

## Open questions

1. P2 `StructuredOutput` duplication with `Text` (SPEC open question 3) —
   settle at Phase 2 review.
2. Parallelizing P2/P3/P4 across sessions after P1 — allowed by the graph;
   default is sequential unless requested.

## Notes for the implementer

- Consult pinned SDK sources in the module cache for exact param/type names
  (anthropic-sdk-go, openai-go/v3) instead of guessing; bump versions only
  with approval.
- Never scatter flavor checks in adapters — unsupported requests must be
  rejected in core validation before an adapter sees them; adapters may
  assume validated input.
- Every phase updates the README capability matrix and `SPEC.md` inventory
  if a cell's truth changes (e.g. T2.4 fallback).
- Default tests stay offline; each new CLI flag also lands in the opt-in
  integration smoke.
