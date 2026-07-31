package llmwrapper

import "context"

// Client sends provider-neutral generation requests through one configured
// provider and flavor.
type Client struct {
	provider Provider
	flavor   Flavor
	model    string
	backend  backend
}

// New constructs a Client for one provider, flavor, and model.
func New(config Config) (*Client, error) {
	return newClient(config, nil)
}

func newClient(config Config, runner commandRunner) (*Client, error) {
	if err := config.validate(); err != nil {
		return nil, &Error{
			Provider:  config.Provider,
			Flavor:    config.Flavor,
			Operation: "client.new",
			Err:       err,
		}
	}

	target, err := buildBackend(config, runner)
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
	if err := request.validate(); err != nil {
		return nil, &Error{
			Provider:  c.provider,
			Flavor:    c.flavor,
			Operation: "generate",
			Err:       err,
		}
	}

	response, err := c.backend.generate(ctx, request)
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
