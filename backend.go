package llmwrapper

import "context"

type backend interface {
	generate(context.Context, Request) (*Response, error)
}

func buildBackend(config Config) (backend, error) {
	switch {
	case config.Provider == ProviderClaude && config.Flavor == FlavorAPI:
		return newAnthropicBackend(config), nil
	case config.Provider == ProviderClaude && config.Flavor == FlavorCLI:
		return newClaudeCLIBackend(config), nil
	case config.Provider == ProviderOpenAI && config.Flavor == FlavorAPI:
		return newOpenAIBackend(config), nil
	case config.Provider == ProviderOpenAI && config.Flavor == FlavorCLI:
		return newCodexCLIBackend(config), nil
	default:
		return nil, &Error{
			Provider:  config.Provider,
			Flavor:    config.Flavor,
			Operation: "client.new",
			Err:       ErrInvalidConfig,
		}
	}
}
