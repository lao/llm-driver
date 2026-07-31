package llmwrapper

import "context"

// Client sends provider-neutral generation requests through one configured
// provider and flavor.
type Client struct {
	config  Config
	backend backend
}

// New constructs a Client for one provider, flavor, and model.
func New(config Config) (*Client, error) {
	if err := config.validate(); err != nil {
		return nil, &Error{
			Provider:  config.Provider,
			Flavor:    config.Flavor,
			Operation: "client.new",
			Err:       err,
		}
	}

	config.CLIArgs = append([]string(nil), config.CLIArgs...)
	target, err := buildBackend(config)
	if err != nil {
		return nil, err
	}
	return &Client{config: config, backend: target}, nil
}

// Generate validates and sends one text request through the configured target.
func (c *Client) Generate(ctx context.Context, request Request) (*Response, error) {
	if err := request.validate(); err != nil {
		return nil, &Error{
			Provider:  c.config.Provider,
			Flavor:    c.config.Flavor,
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
			Provider:  c.config.Provider,
			Flavor:    c.config.Flavor,
			Operation: "generate",
			Message:   "backend returned a nil response",
		}
	}

	response.Provider = c.config.Provider
	response.Flavor = c.config.Flavor
	if response.Model == "" {
		response.Model = c.config.Model
	}
	return response, nil
}
