package claudewrap

import "context"

// MessagesService mirrors the Anthropic Messages API.
type MessagesService struct {
	backend backend
}

// New sends a non-streaming Messages request and returns a neutral Response.
func (s *MessagesService) New(ctx context.Context, p MessageParams) (*Response, error) {
	if err := p.validate(); err != nil {
		return nil, err
	}
	return s.backend.messages(ctx, p)
}

// validate enforces required fields before a request reaches any backend.
func (p MessageParams) validate() error {
	if p.MaxTokens <= 0 {
		return &Error{Op: "messages.new", Message: "MaxTokens must be greater than 0"}
	}
	if len(p.Messages) == 0 {
		return &Error{Op: "messages.new", Message: "at least one message is required"}
	}
	return nil
}
