package llmwrapper

import "context"

type backend interface {
	generate(context.Context, Request) (*Response, error)
}

type pendingBackend struct {
	config Config
}

func (b *pendingBackend) generate(context.Context, Request) (*Response, error) {
	return nil, &Error{
		Provider:  b.config.Provider,
		Flavor:    b.config.Flavor,
		Operation: "generate",
		Message:   "backend is not implemented",
	}
}

func buildBackend(config Config) (backend, error) {
	switch {
	case config.Provider == ProviderClaude && config.Flavor == FlavorAPI:
		return newAnthropicBackend(config), nil
	case config.Provider == ProviderClaude && config.Flavor == FlavorCLI:
		return &pendingBackend{config: config}, nil
	case config.Provider == ProviderOpenAI && config.Flavor == FlavorAPI:
		return newOpenAIBackend(config), nil
	case config.Provider == ProviderOpenAI && config.Flavor == FlavorCLI:
		return &pendingBackend{config: config}, nil
	default:
		return nil, &Error{
			Provider:  config.Provider,
			Flavor:    config.Flavor,
			Operation: "client.new",
			Err:       ErrInvalidConfig,
		}
	}
}
