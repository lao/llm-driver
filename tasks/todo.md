# llmwrapper streaming checklist

**Status:** approved 2026-08-03 (autonomous run)

**Spec:** [`SPEC.md`](../SPEC.md) · **Plan:** [`tasks/plan.md`](plan.md)

Dependency-ordered. Each task: failing test first, minimum code to pass,
refactor green, run its verification before checking the box.

- [x] S1: `stream.go` core — `Stream`/`Chunk` public API, `chunkStream` seam,
      `Client.GenerateStream`, scripted-fake contract tests
- [x] S2: Anthropic Messages streaming adapter + SSE fixture tests
      — checkpoint: reassess seam before replicating
- [x] S3: OpenAI Responses streaming adapter + SSE fixture tests
- [x] S4: streaming CLI subprocess runner (pipe stdout, wait, kill on
      close/cancel) + hygiene tests under `-race`
- [ ] S5: Claude CLI stream-json adapter (`--include-partial-messages`) +
      incremental fake-runner tests
- [ ] S6: Codex CLI incremental JSONL adapter (delta dedup) + tests
      — checkpoint: full race suite
- [ ] S7: four-target streaming contract suite in `contract_test.go`
- [ ] S8: `examples/switchable --stream`, README, `doc.go`
- [ ] S9: final verification: gofmt, build, vet, test, race, cover, example
      build, stale-identifier grep
