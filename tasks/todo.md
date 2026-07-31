# llmwrapper implementation checklist

**Status:** approved on 2026-07-31

**Spec:** [`SPEC.md`](../SPEC.md)

**Plan:** [`tasks/plan.md`](plan.md)

Tasks are dependency-ordered. Each implementation task starts with a focused
failing test, adds the minimum production code to pass, refactors under the
green test suite, and runs its verification commands before the checkbox is
updated.

## Task 1: Cut over package identity and public contract

**Description:** Atomically rename the module/root package and replace the old
Claude-specific API with the approved provider-neutral `Config`,
`Client.Generate`, request/response/usage types, validation, contextual errors,
and backend interface. Install internal target factories so every valid
provider/flavor pair constructs while adapters are filled in by later tasks.

This is the one approved sizing exception: Go requires every file in the root
directory to change package identity together, and the clean-break API makes the
legacy tests invalid in the same cutover. Mechanical declaration edits and
legacy deletions must be kept separate from new behavioral logic in the diff.

**Acceptance criteria:**

- [x] `go.mod`, every root Go package declaration, exported API, sentinel, and
      error prefix use `llmwrapper`; no compatibility API remains.
- [x] Exactly the four approved provider/flavor pairs and nonblank models are
      accepted; invalid config/request values fail before adapter invocation.
- [x] A fake backend proves `Generate` delegates once and centrally stamps
      provider/flavor metadata while preserving `errors.Is`/`errors.As`.

**Verification:**

- [x] `go test ./... -run 'Test(Config|Request|Client|Error|Message|Usage)'`
- [x] `go test ./...`
- [x] `go vet ./...`
- [x] `git diff --check`

**Dependencies:** None.

**Files likely touched:**

- `go.mod`
- Root `*.go` package declarations (mechanical rename)
- `types.go`, `errors.go`, `backend.go`, `config.go`, `client.go`
- `types_test.go`, `client_test.go`
- Removal of superseded `claude.go`, `options.go`, `messages.go`, and legacy
  tests/adapters that cannot compile against the clean-break API

**Estimated scope:** Large mechanical migration exception; one semantic
contract, reviewed as its own checkpoint.

## Checkpoint: Neutral contract

- [x] Package identity and API match the approved spec.
- [x] Default build/tests are green without provider credentials.
- [x] Diff review confirms the large file count is mechanical rather than hidden
      feature work.

## Task 2: Implement the Claude API adapter

**Description:** Restore Claude API support behind the new backend interface
using the official Anthropic SDK and Messages API. Map the neutral system
instruction, structured transcript, configured model, and max tokens; normalize
text, response identity, usage, and API errors.

**Acceptance criteria:**

- [x] Captured HTTP requests contain the expected Messages path/body and
      multi-turn roles without exposing Anthropic types publicly.
- [x] Successful text/cache usage maps to the neutral response with
      `ProviderClaude`/`FlavorAPI`.
- [x] Non-2xx and transport failures become contextual `*Error` values with
      status/code/cause preserved.

**Verification:**

- [x] `go test ./... -run 'TestAnthropic'`
- [x] `go test ./...`
- [x] `go vet ./...`
- [x] `git diff --check`

**Dependencies:** Task 1.

**Files likely touched:**

- `backend_anthropic.go`
- `backend_anthropic_test.go`
- `client.go`

**Estimated scope:** Medium (3 files).

## Task 3: Implement the OpenAI Responses API adapter

**Description:** Add the official OpenAI Go SDK and implement the OpenAI API
target through Responses. Map the same neutral request, including system
instructions, multi-turn roles, configured model, and maximum output tokens;
normalize text, response identity, usage, and SDK errors.

**Acceptance criteria:**

- [x] Captured requests use `/responses` with the expected structured input,
      instructions, model, and output-token limit.
- [x] Successful output text plus cached/reasoning usage maps to the neutral
      response with `ProviderOpenAI`/`FlavorAPI`.
- [x] Non-2xx and transport failures preserve HTTP status, provider code, and
      wrapped cause without leaking SDK types.

**Verification:**

- [x] `go test ./... -run 'TestOpenAI'`
- [x] `go test ./...`
- [x] `go vet ./...`
- [x] `git diff --check`

**Dependencies:** Task 1; executed after Task 2 for a simple review sequence.

**Files likely touched:**

- `go.mod`
- `go.sum`
- `backend_openai.go`
- `backend_openai_test.go`
- `client.go`

**Estimated scope:** Medium (5 files).

## Checkpoint: API flavors

- [x] Both API adapters pass independently and together.
- [x] `go test -race ./...`
- [x] Provider SDK imports occur only in their own adapters/tests.
- [x] Captured-wire tests prove Messages vs. Responses endpoint selection.

## Task 4: Implement the shared runner and Claude CLI adapter

**Description:** Add the safe shared command boundary and transcript renderer,
then implement the Claude CLI target using non-interactive JSON output. User
content is passed only through stdin; CLI configuration is direct argv; process
execution stays injectable.

**Acceptance criteria:**

- [x] Tests assert exact executable/argv/stdin for system, single-turn, and
      multi-turn requests, including configured path/extra arguments.
- [x] Claude success JSON maps response text/session/usage with
      `ProviderClaude`/`FlavorCLI`.
- [x] Missing executable, launch failure, non-zero exit, malformed JSON, and
      non-success subtype map to stable sentinels/contextual errors.

**Verification:**

- [x] `go test ./... -run 'Test(ClaudeCLI|Prompt|Command)'`
- [x] `go test ./...`
- [x] `go vet ./...`
- [x] `git diff --check`

**Dependencies:** Task 1; executed after the API checkpoint.

**Files likely touched:**

- `backend_cli.go`
- `backend_cli_test.go`
- `backend_claude_cli.go`
- `backend_claude_cli_test.go`
- `client.go`

**Estimated scope:** Medium (5 files).

## Task 5: Implement the Codex CLI adapter

**Description:** Add the OpenAI CLI target using `codex exec --json`. Parse JSONL
events, select only completed agent-message text, collect turn-completed usage,
and normalize terminal and process failures.

**Acceptance criteria:**

- [x] Tests assert `codex exec` non-interactive/read-only/ephemeral argv and the
      rendered request on stdin, including custom CLI configuration.
- [x] Progress/reasoning events are ignored; the final agent message and
      cached/reasoning usage map with `ProviderOpenAI`/`FlavorCLI`.
- [x] Failed turns, error events, missing final messages, malformed JSONL,
      missing executable, and non-zero exits return contextual errors.

**Verification:**

- [x] `go test ./... -run 'TestCodexCLI'`
- [x] `go test ./...`
- [x] `go vet ./...`
- [x] `git diff --check`

**Dependencies:** Task 4.

**Files likely touched:**

- `backend_codex_cli.go`
- `backend_codex_cli_test.go`
- `client.go`

**Estimated scope:** Medium (3 files).

## Checkpoint: CLI flavors

- [x] Both CLI adapters pass independently and together.
- [x] `go test -race ./...`
- [x] No default test starts `claude` or `codex`.
- [x] Only the shared default runner imports `os/exec`; prompts never enter a
      shell command string.

## Task 6: Prove the four-target contract

**Description:** Replace target-specific parity assumptions with one
table-driven contract suite. Each fake upstream emits equivalent content and
usage; the unchanged `Generate` call must return the same neutral shape with
only expected provider/flavor identity differences.

**Acceptance criteria:**

- [x] All four targets pass identical text, model, usage, validation, and
      error-context assertions.
- [x] Construction tests prove all and only the approved provider/flavor matrix.
- [x] Any excluded transport behavior is documented inline and matches the spec.

**Verification:**

- [x] `go test ./... -run 'TestContract'`
- [x] `go test -race ./...`
- [x] `go test -cover ./...`
- [x] `git diff --check`

**Dependencies:** Tasks 2–5.

**Files likely touched:**

- `contract_test.go`
- `client_test.go`
- `errors_test.go`

**Estimated scope:** Medium (3 files).

## Task 7: Add the switchable command-line example

**Description:** Create a small example program that parses provider, flavor,
model, prompt, and token limit flags, constructs one `Config`, and issues one
unchanged `Generate` call. Keep argument parsing in a testable helper without
adding a CLI framework.

**Acceptance criteria:**

- [x] `--provider`, `--flavor`, `--model`, and `--prompt` are required/validated;
      `--max-tokens` has a documented positive default.
- [x] Valid flags create the expected config/request; invalid values produce a
      concise error and non-zero exit.
- [x] `--help` describes API-key environment variables and local CLI auth.

**Verification:**

- [x] `go test ./examples/switchable`
- [x] `go build ./examples/switchable`
- [x] `go run ./examples/switchable --help`
- [x] `git diff --check`

**Dependencies:** Task 6.

**Files likely touched:**

- `examples/switchable/main.go`
- `examples/switchable/main_test.go`

**Estimated scope:** Small (2 files).

## Task 8: Write package and user documentation

**Description:** Add a concise README and replace Claude-specific package docs.
Show the same request code with configuration-only target switching, installation
instructions, the four-target matrix, authentication, and honest CLI/API
limitations.

**Acceptance criteria:**

- [x] README includes installation, target matrix, API/CLI quick starts, example
      command, authentication, and limitations without claiming exact CLI/API
      parity.
- [x] `doc.go` demonstrates the approved API and every exported declaration has
      a concise Go doc comment.
- [x] Documentation consistently uses `llmwrapper` and the new module path.

**Verification:**

- [x] `go test ./...`
- [x] `go vet ./...`
- [x] `go doc .`
- [x] `git diff --check`

**Dependencies:** Task 7.

**Files likely touched:**

- `README.md`
- `doc.go`
- Exported declarations missing comments, if found by the documentation pass

**Estimated scope:** Small to medium (2–4 files).

## Checkpoint: Feature complete

- [x] Contract, adapter, and example tests pass together.
- [x] `go build ./...`
- [x] `go test -race ./...`
- [x] `go test -cover ./...`
- [x] README/spec/API agree on the exact four targets and scope.

## Task 9: Run the completion and cleanliness audit

**Description:** Tidy/format the final tree and prove every spec success
criterion from current evidence. Fix any gaps found, rerun the complete gate,
and record completed checklist items only after their commands pass.

**Acceptance criteria:**

- [ ] No active package/module identifier or removed API survives, and dependency
      metadata is tidy.
- [ ] Build, unit, race, coverage, vet, example, formatting, and diff checks all
      pass without credentials or real CLIs.
- [ ] A requirement-by-requirement audit finds evidence for every success
      criterion and no unexplained scope divergence.

**Verification:**

- [ ] `test -z "$(gofmt -l .)"`
- [ ] `go mod tidy && git diff --exit-code -- go.mod go.sum`
- [ ] `go build ./...`
- [ ] `go test ./...`
- [ ] `go test -race ./...`
- [ ] `go test -cover ./...`
- [ ] `go vet ./...`
- [ ] `go build ./examples/switchable`
- [ ] `rg -n 'package claudewrap|github.com/lao/claudewrap' --glob '*.go' --glob 'go.mod' --glob '*.md'` produces no output
- [ ] `git diff --check origin/main...HEAD`

**Dependencies:** Tasks 1–8.

**Files likely touched:**

- `go.mod`
- `go.sum`
- `tasks/todo.md`
- Any directly evidenced correction required by a failing gate

**Estimated scope:** Small if prior checkpoints were honored.

## Task 10: Review, push, and create the pull request

**Description:** Review the complete branch diff for correctness, simplicity,
security, and scope; create cohesive commits if needed; push the verified branch;
and open a GitHub PR against `main` that links the spec and reports exact test
results plus the repository-rename caveat.

**Acceptance criteria:**

- [ ] Working tree is clean and branch commits contain only in-scope changes.
- [ ] `origin/codex/llmwrapper` points at the fully verified local commit.
- [ ] A GitHub PR against `main` exists with summary, testing evidence, spec
      reference, and repository rename follow-up.

**Verification:**

- [ ] `git status --short --branch`
- [ ] `git diff --check origin/main...HEAD`
- [ ] `git log --oneline origin/main..HEAD`
- [ ] `gh pr view --json number,url,title,baseRefName,headRefName,state`

**Dependencies:** Task 9.

**Files likely touched:** None beyond final checklist status.

**Estimated scope:** Small.

## Final checkpoint

- [ ] All ten tasks and all checkpoints are complete.
- [ ] The complete verification suite is green at the pushed commit.
- [ ] The pull request URL is recorded in the final handoff.
