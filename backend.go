package claudewrap

import "context"

// backend is the unexported strategy both flavors implement. The public API
// deals only in neutral types; concrete backends map to/from their transport.
type backend interface {
	// messages performs a non-streaming request.
	messages(ctx context.Context, p MessageParams) (*Response, error)
	// name reports the flavor: "sdk" or "cli".
	name() string
}
