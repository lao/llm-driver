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

// Config selects a provider, transport flavor, and model for a Client.
type Config struct {
	// Provider selects Claude, OpenAI, or opencode.
	Provider Provider
	// Flavor selects the hosted API or locally authenticated CLI.
	Flavor Flavor
	// Model is the explicit provider model name.
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
}

func (c Config) validate() error {
	if c.Provider != ProviderClaude && c.Provider != ProviderOpenAI && c.Provider != ProviderOpenCode {
		return fmt.Errorf("%w: provider must be %q, %q, or %q", ErrInvalidConfig, ProviderClaude, ProviderOpenAI, ProviderOpenCode)
	}
	if c.Flavor != FlavorAPI && c.Flavor != FlavorCLI {
		return fmt.Errorf("%w: flavor must be %q or %q", ErrInvalidConfig, FlavorAPI, FlavorCLI)
	}
	if strings.TrimSpace(c.Model) == "" {
		return fmt.Errorf("%w: model is required", ErrInvalidConfig)
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
	return nil
}
