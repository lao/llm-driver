# Todo: SDK generation parity

Source: [`plan.md`](plan.md) — one phase = one PR. Every task ends with
`gofmt -l .` empty, `go build ./... && go vet ./... && go test ./... &&
go test -race ./...` green.

## Phase 1 — Capability framework (PR 1)

- [ ] T1.1 `Feature` constants, `ErrUnsupportedFeature`, per-target table
      (`internal/core/capabilities.go`) + tests
- [ ] T1.2 `featuresOf(Request)` + gate in `Generate`/`GenerateStream`
      (no-op for existing requests, proven)
- [ ] T1.3 Root aliases, `Client.Supports`, README matrix, `doc.go`
- [ ] Checkpoint: suite + `-race` green; review Feature granularity

## Phase 2 — Structured output (PR 2)

- [ ] T2.1 `OutputSchema`, `Request.OutputSchema`,
      `Response.StructuredOutput`, validation, aliases
- [ ] T2.2 anthropic/api `output_config.format` mapping (inspect pinned SDK
      first; bump = ask) — batch + stream + fixtures
- [ ] T2.3 openai/api `text.format` json_schema strict — batch + stream
- [ ] T2.4 claude/cli `--json-schema` argv + parse + integration smoke
      (resolves inline-vs-file question; update SPEC if fallback)
- [ ] T2.5 codex/cli `--output-schema` temp file (0600, removed on all exit
      paths) + integration smoke
- [ ] T2.6 Contract suite rows, example `--schema` flag, README
- [ ] Checkpoint: all four targets green offline; PR review

## Phase 3 — Reasoning effort + richer results (PR 3)

- [ ] T3.1 `Effort` enum, `Request.Effort`, per-target allowed sets
      (verify openai ceiling against pinned SDK)
- [ ] T3.2 Mappings: `output_config.effort` / `reasoning.effort` /
      `--effort` / `-c model_reasoning_effort` + per-adapter tests
- [ ] T3.3 `CompletionReasonToolUse`, `CompletionReasonContentFilter`
      constants + observable mappings
- [ ] T3.4 Contract row, example `--effort`, README
- [ ] Checkpoint: opt-in smoke on real binaries; review

## Phase 4 — Sampling params + UserID (PR 4)

- [ ] T4.1 `Temperature`/`TopP`/`TopK`/`StopSequences`/`UserID` fields,
      granular features, minimal validation
- [ ] T4.2 anthropic/api mapping incl. `metadata.user_id`; 400-rejection
      fixture (pass-through semantics)
- [ ] T4.3 openai/api mapping (`safety_identifier`); TopK + StopSequences
      stay ✗ with contract-asserted rejection
- [ ] T4.4 Contract rows (CLI rejects all), README matrix
- [ ] Checkpoint P1–P4: full verification; human review before P5

## Phase 5 — Tool calling (PR 5)

- [ ] T5.1 `Tool`/`ToolChoice`/`ToolCall`, `Part` union
      (Text/ToolCall/ToolResult), `Message.Parts` (exactly one of
      Text/Parts), `ToolResult` helper, backward-compat tests
- [ ] T5.2 anthropic/api round-trip (tool_use blocks, parallel calls,
      `input_json_delta` → `Chunk.Kind: KindToolInput`)
- [ ] T5.3 openai/api round-trip (function_call items,
      function_call_output, arg deltas)
- [ ] T5.4 Contract rows + CLI rejection + README tool-loop example
- [ ] Checkpoint — core parity done: no third state between implemented and
      `ErrUnsupportedFeature`; `Supports` == runtime; review + confirm
      P6–P9 order

## Phases 6–9 (plan in detail when reached)

- [ ] P6 multimodal input (needs P5 parts)
- [ ] P7 client robustness (retries/timeouts) — independent after P1
- [ ] P8 conversation continuation (`ContinueFrom`)
- [ ] P9 web search — ask-first gate (widens CLI subprocess abilities)
