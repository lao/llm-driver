# claudewrap — Implementation Plan

Derived from `SPEC.md`. Greenfield; single Go package (root) + `internal` helpers.
Work is sliced **vertically** — each task delivers one complete caller→backend→neutral-type
path, testable on its own. No horizontal "write all types, then all backends" layering.

---

## Dependency graph

```
types.go ─┬─────────────────────────────────────────────┐
          │                                              │
errors.go ┤                                              │
          ▼                                              ▼
      backend.go (iface) ──┬── backend_sdk.go        stream.go
          │                └── backend_cli.go            │
          ▼                        │                     │
   options.go / claude.go (New, backend select)          │
          ▼                        │                     │
      messages.go (MessagesService: New, NewStreaming) ──┘
          │
          ├── backend_sdk_test.go (httptest)
          ├── backend_cli_test.go (fake commandRunner)
          ├── contract_test.go (both backends, parity)
          └── examples/, README, Makefile, doc.go
```

Hard edges:
- Everything depends on `types.go` + `errors.go`.
- `claude.go`/`options.go` depend on `backend.go` (iface) — but **not** on the
  concrete backends’ internals (constructed via unexported factories in the same package → no cycle).
- `anthropic-sdk-go` imported **only** in `backend_sdk.go`.
- Streaming (`stream.go`) is independent of non-streaming and lands after the
  non-streaming path is proven.

---

## Phases, checkpoints & vertical slices

### Phase A — SDK non-streaming path (first end-to-end value)

**T1. Foundation types + errors**
- Files: `types.go`, `errors.go`, `doc.go`.
- Neutral types per SPEC §3: `Role`, `BlockType`, `StopReason`, `EventType`,
  `Message`, `ContentBlock`, `MessageParams`, `Response`, `Usage`, `Tool`.
- Helpers: `UserMessage`, `AssistantMessage`, `TextBlock`, `(*Response).Text()`.
- `Error` type (`Backend`, `Op`, `Status`, `Type`, `Message`, `Err`), `Error()`,
  `Unwrap()`; sentinels `ErrNoBackend`, `ErrCLINotFound`.
- **Acceptance:** types compile; helpers + `Text()` covered by `types_test.go`.
- **Verify:** `go build ./...`; `go test -run Types ./...`; `go vet ./...`.

**T2. Backend interface + Client/options + SDK backend (messages only)**
- Files: `backend.go`, `options.go`, `claude.go`, `backend_sdk.go`, `messages.go`.
- `backend` iface: `messages`, `streamMessages` (stub returns "not implemented"
  for now), `name`.
- `New(...Option)`: validate exactly one backend option set → else `ErrNoBackend`.
- `WithSDK(key, ...SDKOption)`, `WithSDKBaseURL`, `WithSDKHTTPClient`,
  `WithDefaultModel`.
- `sdkBackend.messages`: map `MessageParams` → `anthropic.MessageNewParams`,
  call `client.Messages.New`, map `anthropic.Message` → neutral `Response`
  (content text blocks, `Usage`, `StopReason`, `Backend:"sdk"`). Model defaults
  to `claude-opus-4-8`. Wrap errors in `*Error`.
- `MessagesService.New` delegates to backend.
- Add `anthropic-sdk-go` require to `go.mod` (pinned).
- **Acceptance:** `c,_ := New(WithSDK("k", WithSDKBaseURL(ts.URL)))` then
  `Messages.New` returns a correct neutral `Response` from a canned httptest body;
  error status → `*Error` with `Status` + `Backend:"sdk"`.
- **Verify:** `go test -run SDK ./...` (httptest, no network); `go vet ./...`; `gofmt -l .` empty.

> **CHECKPOINT 1 — SDK flavor works end-to-end.** One real transport proven
> against a fake server, neutral types only. Human review of the public API shape
> before the second backend locks it in.

---

### Phase B — CLI non-streaming path + parity

**T3. CLI backend (messages only)**
- File: `backend_cli.go`.
- `WithCLI(...CLIOption)`, `WithCLIPath`, `WithCLIExtraArgs`.
- Injectable `commandRunner func(ctx, name string, args []string, stdin []byte) (stdout, stderr []byte, exit int, err error)`; default = real `exec.CommandContext`.
- `cliBackend.messages`: build args (`-p`, `--output-format json`,
  `--model`, `--append-system-prompt`), flatten transcript → prompt via **stdin**
  (no shell string), parse CLI JSON → neutral `Response` (`Backend:"cli"`).
- Missing binary → `ErrCLINotFound`; non-zero exit / bad JSON → `*Error` with
  `Status`=exit code, `Backend:"cli"`.
- **Acceptance:** fake runner returning canned CLI JSON → correct neutral
  `Response`; exit=1 → `*Error`; missing binary → `ErrCLINotFound`.
- **Verify:** `go test -run CLI ./...` (fake runner, no process); `go vet`; `gofmt -l .`.

**T4. Cross-backend contract test (non-streaming)**
- File: `contract_test.go`.
- One assertion set run against both backends (each with its fake) proving
  identical neutral output for an equivalent prompt (content text, role, usage mapping).
- **Acceptance:** same input → equivalent neutral `Response` from both (allowing
  `Backend` field + documented CLI-dropped params to differ).
- **Verify:** `go test -run Contract ./...`.

> **CHECKPOINT 2 — Both flavors reach parity on non-streaming.** Confirms the
> neutral-type boundary holds across two very different transports.

---

### Phase C — Streaming, both backends

**T5. Stream abstraction + SDK streaming**
- Files: `stream.go`, extend `backend_sdk.go`, `messages.go`.
- `Stream` (`Next`, `Event`, `Err`, `Response`, `Close`), `StreamEvent`.
- `sdkBackend.streamMessages`: drive `anthropic-sdk-go` streaming, accumulate to
  final `Response`, emit `text_delta` events.
- `MessagesService.NewStreaming` delegates.
- **Acceptance:** httptest SSE body → ordered `text_delta` events + accumulated
  `Response.Text()` equal to full text.
- **Verify:** `go test -run StreamSDK ./...`.

**T6. CLI streaming + streaming contract**
- Files: extend `backend_cli.go`, extend `contract_test.go`.
- `cliBackend.streamMessages`: `--output-format stream-json --verbose`, parse
  JSONL lines from fake runner’s piped stdout → `text_delta` events + final `Response`.
- Streaming contract test across both backends.
- **Acceptance:** fake stream-json lines → same event/text sequence shape as SDK;
  streaming contract passes.
- **Verify:** `go test -run 'StreamCLI|StreamContract' ./...`; then full `go test ./...`.

> **CHECKPOINT 3 — Streaming parity.** Full unit suite green on both flavors,
> non-streaming + streaming.

---

### Phase D — Ship polish

**T7. Examples, docs, tooling, integration tests**
- Files: `examples/sdk/main.go`, `examples/cli/main.go`, `README.md`,
  `Makefile`, integration tests behind `//go:build integration`.
- README: install, both-flavor quickstart, CLI limitations (SPEC §7), backend
  selection table.
- Integration: SDK needs `ANTHROPIC_API_KEY` (one `claude-haiku-4-5` call); CLI
  skips with a clear message if `claude` absent.
- **Acceptance:** examples compile (`go build ./examples/...`); `make test lint`
  green; `gofmt -l .` empty; every exported symbol has godoc.
- **Verify:** `go build ./...`; `go test ./...`; `go vet ./...`; `gofmt -l .`;
  `golangci-lint run` (if installed); `go test -tags integration ./...` (opt-in, may skip).

> **CHECKPOINT 4 — Ship-ready v0.1.** Full green, examples build, docs complete.

---

## Global verification (run after every task)
```
go build ./...  &&  go vet ./...  &&  gofmt -l .  &&  go test ./...
```
`gofmt -l .` must print nothing. Default `go test` never hits network or spawns `claude`.

## Risks & mitigations
- **CLI JSON schema drift** — pin behavior to fields actually emitted by the
  installed `claude`; verify flags/shape against the real binary before T3 lands;
  keep mapping in one function.
- **anthropic-sdk-go API differences** — write `backend_sdk.go` first, let the
  compiler drive exact type/field names (per skill guidance); do not guess.
- **Import cycle** — avoided by keeping both backends in the root package.
- **Prompt injection into shell** — args/stdin only; never a shell string (enforced in T3, checked in review).
```

