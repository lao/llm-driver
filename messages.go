package claudewrap

import "context"

// MessagesService mirrors the Anthropic Messages API.
type MessagesService struct {
	backend backend
}

// New sends a non-streaming Messages request and returns a neutral Response.
func (s *MessagesService) New(ctx context.Context, p MessageParams) (*Response, error) {
	return s.backend.messages(ctx, p)
}
