package llmdriver

import (
	"context"

	"github.com/lao/llm-driver/internal/backends"
	"github.com/lao/llm-driver/internal/core"
)

// Client sends provider-neutral generation requests through one configured
// provider and flavor.
type Client struct {
	provider Provider
	flavor   Flavor
	model    string
	backend  core.Backend
}

// New constructs a Client for one provider, flavor, and model.
func New(config Config) (*Client, error) {
	return newClient(config, nil)
}

func newClient(config Config, runner core.CommandRunner) (*Client, error) {
	if err := core.ValidateConfig(config); err != nil {
		return nil, &Error{
			Provider:  config.Provider,
			Flavor:    config.Flavor,
			Operation: "client.new",
			Err:       err,
		}
	}

	target, err := backends.New(config, runner)
	if err != nil {
		return nil, err
	}
	return &Client{
		provider: config.Provider,
		flavor:   config.Flavor,
		model:    config.Model,
		backend:  target,
	}, nil
}

// Generate validates and sends one text request through the configured target.
func (c *Client) Generate(ctx context.Context, request Request) (*Response, error) {
	if err := core.ValidateRequest(request); err != nil {
		return nil, &Error{
			Provider:  c.provider,
			Flavor:    c.flavor,
			Operation: "generate",
			Err:       err,
		}
	}

	response, err := c.backend.Generate(ctx, request)
	if err != nil {
		return nil, err
	}
	if response == nil {
		return nil, &Error{
			Provider:  c.provider,
			Flavor:    c.flavor,
			Operation: "generate",
			Message:   "backend returned a nil response",
		}
	}

	response.Provider = c.provider
	response.Flavor = c.flavor
	if response.Model == "" {
		response.Model = c.model
	}
	return response, nil
}
