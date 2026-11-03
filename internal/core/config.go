package core

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

// Provider identifies the model provider selected by a Config.
type Provider string

const (
	// ProviderClaude selects Anthropic Claude.
	ProviderClaude Provider = "claude"
	// ProviderOpenAI selects OpenAI.
	ProviderOpenAI Provider = "openai"
	// ProviderOpenCode selects the local opencode agent CLI. opencode fronts
	// many upstream providers, so its model is a "provider/model" id obtained
	// from `opencode models` (see ListModels).
	ProviderOpenCode Provider = "opencode"
)

// Flavor identifies how a provider is invoked.
type Flavor string

const (
	// FlavorAPI invokes a provider's hosted API through its official Go SDK.
	FlavorAPI Flavor = "api"
	// FlavorCLI invokes the provider's locally authenticated command-line tool.
	FlavorCLI Flavor = "cli"
)

// ClaudeOptions carries Claude CLI flags for a CLI-flavored client. It is
// ignored by API flavors and by the OpenAI provider. The zero value preserves
// the library's default invocation.
type ClaudeOptions struct {
	// PermissionMode overrides the --permission-mode value. Empty selects the
	// CLI's own default ("default"), which keeps a headless run from approving
	// agent tool actions.
	PermissionMode string
	// ReplaceSystemPrompt passes system text through --system-prompt instead of
	// --append-system-prompt, replacing the CLI's built-in prompt rather than
	// appending to it.
	ReplaceSystemPrompt bool
}

// CodexOptions carries Codex CLI flags for a CLI-flavored client. It is ignored
// by API flavors and by the Claude provider. The zero value preserves the
// library's default invocation.
type CodexOptions struct {
	// SandboxMode overrides the --sandbox value. Empty selects the library's
	// default ("read-only").
	SandboxMode string
	// OmitSandbox drops the --sandbox flag so a permissions profile supplied
	// through Config.CLIArgs can apply instead. It is mutually exclusive with
	// SandboxMode.
	OmitSandbox bool
}

// Config selects a provider, transport flavor, and model for a Client.
type Config struct {
	// Provider selects Claude, OpenAI, or opencode.
	Provider Provider
	// Flavor selects the hosted API or locally authenticated CLI.
	Flavor Flavor
	// Model is the explicit provider model name. API flavors require it; CLI
	// flavors may leave it empty to use the provider CLI's own default model.
	Model string

	// APIKey overrides the API flavor's environment-based credential.
	APIKey string
	// BaseURL optionally overrides the API endpoint for a proxy or test server.
	BaseURL string
	// HTTPClient optionally overrides the API flavor's HTTP client.
	HTTPClient *http.Client

	// CLIPath optionally overrides the default claude, codex, or opencode executable.
	CLIPath string
	// CLIArgs appends direct arguments to the selected CLI command.
	CLIArgs []string
	// CLIEnv is the CLI-flavored child process environment, each entry
	// "KEY=VALUE". A nil slice inherits the parent environment (the default); a
	// non-nil slice — including an empty one — is used verbatim, so a caller can
	// fail closed with an allowlist.
	CLIEnv []string
	// CLIWorkDir is the CLI-flavored child process working directory. Empty
	// inherits the parent's.
	CLIWorkDir string

	// AcceptUnverifiedOpenCode allows OpenCode responses without a verified finish.
	// Such responses may be truncated and have CompletionReasonUnverified.
	AcceptUnverifiedOpenCode bool

	// Claude carries Claude CLI flags. Ignored by API flavors and OpenAI.
	Claude ClaudeOptions
	// Codex carries Codex CLI flags. Ignored by API flavors and Claude.
	Codex CodexOptions
}

func (c Config) validate() error {
	if c.Provider != ProviderClaude && c.Provider != ProviderOpenAI && c.Provider != ProviderOpenCode {
		return fmt.Errorf("%w: provider must be %q, %q, or %q", ErrInvalidConfig, ProviderClaude, ProviderOpenAI, ProviderOpenCode)
	}
	if c.Flavor != FlavorAPI && c.Flavor != FlavorCLI {
		return fmt.Errorf("%w: flavor must be %q or %q", ErrInvalidConfig, FlavorAPI, FlavorCLI)
	}
	if c.Provider == ProviderOpenCode && c.Flavor != FlavorCLI {
		return fmt.Errorf("%w: opencode only supports the CLI flavor", ErrInvalidConfig)
	}
	// CLI flavors may omit Model: the provider CLI then runs on its own
	// account default. API flavors require an explicit model.
	if c.Flavor == FlavorAPI && strings.TrimSpace(c.Model) == "" {
		return fmt.Errorf("%w: model is required", ErrInvalidConfig)
	}
	if c.Model != "" && strings.TrimSpace(c.Model) == "" {
		return fmt.Errorf("%w: model cannot be whitespace", ErrInvalidConfig)
	}
	if c.Flavor == FlavorAPI && c.BaseURL != "" {
		u, err := url.Parse(c.BaseURL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return fmt.Errorf("%w: base URL must be an absolute HTTP(S) URL", ErrInvalidConfig)
		}
	}
	if c.Flavor == FlavorCLI && c.CLIPath != "" && strings.TrimSpace(c.CLIPath) == "" {
		return fmt.Errorf("%w: CLI path cannot be whitespace", ErrInvalidConfig)
	}
	if c.Codex.OmitSandbox && c.Codex.SandboxMode != "" {
		return fmt.Errorf("%w: Codex OmitSandbox and SandboxMode are mutually exclusive", ErrInvalidConfig)
	}
	return nil
}
