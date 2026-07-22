# claudewrap — TODO

Vertical slices. Each task ends green on `go build ./... && go vet ./... && gofmt -l . && go test ./...`.
See `tasks/plan.md` for acceptance criteria + verification detail.

## Phase A — SDK non-streaming
- [x] **T1** Foundation: `types.go`, `errors.go`, `doc.go` + `types_test.go`
- [x] **T2** Backend iface + `New`/options + `sdkBackend.messages` + `Messages.New`; add pinned `anthropic-sdk-go` dep; httptest test
- [ ] ⛳ **CHECKPOINT 1** — SDK flavor end-to-end; review public API shape ← **HERE**

## Phase B — CLI non-streaming + parity
- [ ] **T3** `cliBackend.messages` + injectable `commandRunner` + fake-runner tests
- [ ] **T4** `contract_test.go` — non-streaming parity across both backends
- [ ] ⛳ **CHECKPOINT 2** — both flavors reach parity (non-streaming)

## Phase C — Streaming
- [ ] **T5** `stream.go` + `sdkBackend.streamMessages` + `NewStreaming`; SSE httptest test
- [ ] **T6** `cliBackend.streamMessages` (stream-json) + streaming contract test; full suite green
- [ ] ⛳ **CHECKPOINT 3** — streaming parity

## Phase D — Ship polish
- [ ] **T7** examples (sdk/cli), README, Makefile, `-tags integration` tests, godoc pass
- [ ] ⛳ **CHECKPOINT 4** — ship-ready v0.1

## Deferred (post-v0.1, per SPEC §9)
- [ ] `Query()` high-level agent helper
- [ ] tool-execution loop
- [ ] CLI multi-turn via `--input-format stream-json`
- [ ] batches / files / token counting / MCP surfaces
- [ ] build-tag flavor selection
