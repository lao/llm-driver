package claudewrap

// Client is a backend-agnostic Claude client. Construct it with New and one
// backend option, then call methods on its service fields.
type Client struct {
	// Messages exposes the Anthropic Messages API surface.
	Messages *MessagesService

	cfg *config
}

// New constructs a Client. Exactly one backend option (WithSDK or WithCLI) is
// required; otherwise New returns ErrNoBackend.
func New(opts ...Option) (*Client, error) {
	c := &config{defaultModel: DefaultModel}
	for _, o := range opts {
		o(c)
	}

	var b backend
	var err error
	switch c.backendKind {
	case "sdk":
		b = newSDKBackend(c)
	case "cli":
		b, err = newCLIBackend(c)
	default:
		return nil, ErrNoBackend
	}
	if err != nil {
		return nil, err
	}

	return &Client{
		Messages: &MessagesService{backend: b},
		cfg:      c,
	}, nil
}

// Backend reports which flavor this client uses: "sdk" or "cli".
func (c *Client) Backend() string { return c.Messages.backend.name() }
