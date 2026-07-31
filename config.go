package llmwrapper

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
	Provider Provider
	Flavor   Flavor
	Model    string

	APIKey     string
	BaseURL    string
	HTTPClient *http.Client

	CLIPath string
	CLIArgs []string
}

func (c Config) validate() error {
	if c.Provider != ProviderClaude && c.Provider != ProviderOpenAI {
		return fmt.Errorf("%w: provider must be %q or %q", ErrInvalidConfig, ProviderClaude, ProviderOpenAI)
	}
	if c.Flavor != FlavorAPI && c.Flavor != FlavorCLI {
		return fmt.Errorf("%w: flavor must be %q or %q", ErrInvalidConfig, FlavorAPI, FlavorCLI)
	}
	if strings.TrimSpace(c.Model) == "" {
		return fmt.Errorf("%w: model is required", ErrInvalidConfig)
	}
	if c.BaseURL != "" {
		u, err := url.Parse(c.BaseURL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return fmt.Errorf("%w: base URL must be an absolute HTTP(S) URL", ErrInvalidConfig)
		}
	}
	if c.CLIPath != "" && strings.TrimSpace(c.CLIPath) == "" {
		return fmt.Errorf("%w: CLI path cannot be whitespace", ErrInvalidConfig)
	}
	return nil
}
