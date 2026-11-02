# Spec: llm-driver SDK generation parity

Supersedes the streaming spec (shipped; see git history of this file).

## Objective

Bring `llm-driver` to feature parity with the official SDKs' *generation*
surface — the capabilities of a single `messages.create` (Anthropic) /
`responses.create` (OpenAI) call — across all existing targets, without
breaking the current public API.

An `opencode` provider (`cli` flavor) joined the target set after the phases
below: the local `opencode run` CLI, which fronts many upstream providers. It
carries the same generation surface as the other CLI targets (baseline plus
structured output and reasoning effort). Because `opencode run` has no
JSON-schema flag, its structured output states the schema in the prompt and
validates the returned document, failing `invalid_structured_output` on a
non-JSON answer. `ListModels` runs `opencode models` to discover the
`provider/model` ids the installed opencode can use, and `Config.Model` accepts
any such id. opencode 2.x encodes reasoning effort as the model id's `#variant`
suffix (`Response.Model` carries the effective id); an effort with an empty
model or a model that already names a variant is rejected with
`ErrUnsupportedFeature`, because 2.x has no default-model form and cannot
override an inline variant. A plain text turn may omit the terminal
`step_finish`; a completed process plus text is the finish signal, and it
reports `CompletionReasonUnverified` so callers can tell the finish was not
verified. A tool-only turn reports `tool_calls`.

Scope decision (approved): **generation parity only.** Platform APIs that a
CLI can never serve (embeddings, batches, files CRUD, audio, images
generation, fine-tuning, vector stores, admin) are permanently out of scope.

The user is the same Go developer as today: write a call once, switch
`Config` between the four targets. New rule for the expanded surface: when a
target cannot honor a requested feature, the library **fails loudly** — it
never silently degrades.

### Parity inventory and target support

Feature set = union of the two SDKs' generation parameters, normalized to
provider-neutral types. Support per target ("bridge" = mapped through a CLI
flag; "✗" = `ErrUnsupportedFeature`):

| Feature | claude/api | openai/api | claude/cli | codex/cli |
| --- | --- | --- | --- | --- |
| System, messages, MaxTokens, streaming | ✓ (today) | ✓ (today) | ✓ (today) | ✓ (today) |
| Structured output (JSON schema) | ✓ `output_config.format` | ✓ `text.format` json_schema strict | ✓ `--json-schema` | ✓ `--output-schema` |
| Reasoning effort | ✓ `output_config.effort` | ✓ `reasoning.effort` | ✓ `--effort` | ✓ `-c model_reasoning_effort` |
| Reasoning/thinking visibility | ✓ thinking config | ✓ `reasoning.summary` | ✗ | ✗ |
| Tool calling (caller-executed) | ✓ tools + tool_choice | ✓ tools + tool_choice | ✗ (MCP bridge = stretch) | ✗ |
| Image input | ✓ content blocks | ✓ `input_image` | ✗ | ✓ `--image` |
| PDF/document input | ✓ document blocks | ✓ `input_file` | ✗ | ✗ |
| Temperature / TopP / TopK | pass-through¹ | pass-through¹ (no TopK) | ✗ | ✗ |
| Stop sequences | ✓ | ✗ (absent from Responses API) | ✗ | ✗ |
| Web search (provider-executed) | ✓ server tool | ✓ built-in tool | ✓ `--allowedTools WebSearch` | ✓ `--search` |
| Metadata / end-user ID | ✓ `metadata.user_id` | ✓ `safety_identifier` | ✗ | ✗ |
| Conversation continuation by ID | ✗² | ✓ `previous_response_id` | ✓ `--resume` | ✗ (resume is a separate subcommand) |
| Token counting | ✓ `count_tokens` | ✗ | ✗ | ✗ |
| Retries, timeouts | library-level, all four targets | | | |

¹ Providers are moving away from sampling params (Anthropic rejects them on
newest models; OpenAI ignores them on reasoning models). The library passes
them through and surfaces provider rejections as normalized errors; it does
not maintain a model→param compatibility table.
² Anthropic conversations are already expressible via the `Messages`
transcript; no ID-based continuation exists in the Messages API.

### Public API additions (shape, exact names settled per phase)

All types stay provider-neutral in `internal/core` with root aliases, as
today. Additions are strictly backward compatible — existing fields and
methods keep byte-for-byte behavior.

```go
// Capability query — answers "will this target accept this request feature?"
type Feature string // FeatureStructuredOutput, FeatureTools, FeatureImageInput, ...
func (c *Client) Supports(Feature) bool

// New sentinel: requested feature not available on the selected target.
var ErrUnsupportedFeature error

// Request grows optional fields (zero value = today's behavior):
type Request struct {
    // existing: System, Messages, MaxTokens
    Effort          Effort          // "", low, medium, high, xhigh, max (normalized)
    OutputSchema    *OutputSchema   // name + JSON Schema (json.RawMessage)
    Tools           []Tool          // name, description, JSON Schema input
    ToolChoice      ToolChoice      // auto (default) / none / required / named
    WebSearch       *WebSearchOption
    Temperature     *float64
    TopP            *float64
    TopK            *int
    StopSequences   []string
    UserID          string          // abuse-tracking end-user id
    ContinueFrom    string          // provider response / CLI session id
}

// Message grows content parts; Text stays as the text-only fast path.
type Message struct {
    Role  Role
    Text  string  // exactly one of Text / Parts
    Parts []Part  // TextPart, ImagePart (bytes+media type or path), DocumentPart
}

// Response grows:
type Response struct {
    // existing fields unchanged
    ToolCalls  []ToolCall      // id, name, raw JSON input
    StructuredOutput json.RawMessage // present when OutputSchema was set
}
// New CompletionReasonToolUse. Caller loops by appending ToolResult
// messages (helper: llmdriver.ToolResult(id, content, isErr)).

// Streaming: Chunk gains a Kind (default KindText, existing behavior);
// KindThinking and KindToolInput deltas are additive.

// Config grows: MaxRetries *int, Timeout time.Duration (library-level
// retry/backoff for CLI flavors; mapped to SDK options for API flavors).
```

Contract:

- `Generate`/`GenerateStream` validate the request against the selected
  target's capability table **before** invocation; unsupported features
  return `ErrUnsupportedFeature` naming the feature and target. No silent
  drops, ever.
- A request using only today's fields behaves identically to today.
- Provider SDK types still never leak through the public API.

## Roadmap (phased, ~1 PR each, dependency-ordered)

1. **Capability framework** — `Feature`, `Supports`, `ErrUnsupportedFeature`,
   per-target validation table, README capability matrix. Everything else
   hangs off this.
2. **Structured output** — all four targets. `OutputSchema` in,
   `StructuredOutput` out; CLI flag mapping (`--json-schema`,
   `--output-schema` temp file); malformed-output error path.
3. **Reasoning effort + richer results** — `Effort` on all four targets;
   expanded `CompletionReason` values (tool_use, content_filter) and `Usage`
   passthrough already-normalized fields where targets report them.
4. **Sampling params** — `Temperature`, `TopP`, `TopK`, `StopSequences`,
   `UserID`; API flavors only, pass-through semantics (note ¹ above).
5. **Tool calling** — API flavors: `Tools`, `ToolChoice`, parallel calls,
   `ToolCalls`/`ToolResult` round-trip, streaming `KindToolInput` deltas.
   CLI flavors: `ErrUnsupportedFeature`.
6. **Multimodal input** — `Parts` with images (claude/api, openai/api,
   codex/cli via `--image` temp file) and PDFs (API flavors).
7. **Client robustness** — `MaxRetries`/`Timeout` on `Config`, uniform
   backoff for CLI flavors, mapped to SDK retry options for API flavors;
   request IDs on `Error` where available.
8. **Conversation continuation** — `ContinueFrom` (openai/api
   `previous_response_id`, claude/cli `--resume`); `Response.ID` already
   supplies the handle.
9. **Web search** — provider-executed search on all four targets, opt-in.
10. **Stretch (separate approval each):** Claude CLI tool bridge via an
    in-process MCP server; token counting (`claude/api` only); prompt-cache
    control knobs.

Phases 1–5 are the committed core. 6–9 land in priority order after. 10 is
not committed by this spec.

## Tech Stack

Unchanged: Go 1.24, `anthropic-sdk-go` (pinned), `openai-go/v3` (pinned),
stdlib only otherwise. SDK version bumps allowed when a phase needs a newer
parameter surface; no other new dependencies without approval.

## Commands

```bash
go mod tidy
gofmt -l .            # must print nothing
go build ./...
go vet ./...
go test ./...
go test -race ./...
go build ./examples/switchable

# Opt-in real-binary smoke test (unchanged mechanism)
LLM_DRIVER_CLAUDE_CLI_MODEL=... LLM_DRIVER_CODEX_CLI_MODEL=... \
  go test -tags integration -run 'CLIIntegration' .
```

Default tests stay offline: httptest servers + fake process runners.

## Project Structure

Unchanged layout. Each phase touches its adapter files plus:

```text
internal/core/capabilities.go    per-target feature table (new, phase 1)
internal/core/types.go           Request/Response/Part/Tool growth
internal/backends/*.go           per-target mapping of each feature
docs/SPEC.md                     this spec (living document)
docs/tasks/                      per-phase plan + todo, as today
```

## Code Style

Unchanged from prior specs: idiomatic Go, explicit validation, documented
exports, `gofmt` authoritative, no single-caller abstractions. Additions:

- Capability checks live in one table, not scattered `if flavor ==` checks.
- Feature mapping code sits in the owning adapter file; shared normalization
  (e.g. tool-call accumulation) written once in `internal/core` or a shared
  backend helper.

## Testing Strategy

`testing` package only, offline by default, per phase:

- Contract tests: same request through every *supporting* target asserts one
  normalized result shape; every *non-supporting* target asserts
  `ErrUnsupportedFeature` (table-driven off the capability table, so docs,
  validation, and tests cannot drift apart).
- Adapter tests: request-mapping assertions (SDK request JSON / CLI argv) +
  response parsing from fixtures, happy path and error paths.
- Streaming tests for every phase touching stream shape (tool deltas,
  structured output accumulation).
- Process hygiene and `-race` guarantees maintained for CLI changes.
- Opt-in integration smoke extends to new CLI flags (`--json-schema`,
  `--output-schema`, `--effort`, `--image`, `--search`, `--resume`).

## Boundaries

- **Always:** validate before invoking targets; fail loudly on unsupported
  features; keep public types provider-neutral; keep default tests offline;
  prompts/schemas via stdin or temp files, never shell interpolation; keep
  existing behavior for existing requests; update this spec when a decision
  changes.
- **Ask first:** SDK version bumps; any new dependency; anything in the
  stretch list; widening CLI subprocess permissions (e.g. enabling tools or
  search changes what the CLI may do locally); CI changes.
- **Never:** silent feature degradation; leaking SDK types; committing
  credentials; orphan subprocesses; removing failing tests; breaking the
  existing public API.

## Success Criteria

- Every feature in the parity inventory is either implemented on a target or
  deliberately rejected with `ErrUnsupportedFeature` — no third state.
- `Supports` answers agree exactly with runtime behavior (enforced by the
  table-driven contract tests).
- A request using only pre-parity fields produces byte-identical behavior on
  all four targets.
- Each phase lands green (`gofmt`, build, vet, test, `-race`) as its own PR
  with README capability-matrix updates.
- Phases 1–5 complete = "core parity done"; 6–9 complete = spec objective met.

## Open Questions

1. Claude CLI `--json-schema`: verify the flag accepts inline JSON at the
   pinned CLI version during phase 2 (integration smoke will catch drift).
2. Codex `exec resume` ergonomics are awkward (separate subcommand); phase 8
   may add codex support later if the CLI grows an inline flag — spec marks
   it ✗ for now.
3. Whether `StructuredOutput` should also re-expose as `Text` (both, per
   current draft: `Text` keeps raw text, `StructuredOutput` the parsed raw
   JSON) — settle in phase 2 review.
