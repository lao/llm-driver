package claudewrap

import "net/http"

// config holds resolved construction settings for a Client.
type config struct {
	backendKind  string // "sdk" | "cli"
	defaultModel string

	// sdk backend
	sdkAPIKey  string
	sdkBaseURL string
	sdkHTTP    *http.Client
}

// Option configures a Client. Exactly one backend option (WithSDK or WithCLI)
// must be supplied to New.
type Option func(*config)

// SDKOption tunes the sdk (HTTP) backend. Pass to WithSDK.
type SDKOption func(*config)

// WithDefaultModel sets the model used when MessageParams.Model is empty.
// Defaults to DefaultModel.
func WithDefaultModel(model string) Option {
	return func(c *config) { c.defaultModel = model }
}

// WithSDK selects the HTTP backend backed by the official anthropic-sdk-go,
// authenticating with apiKey.
func WithSDK(apiKey string, o ...SDKOption) Option {
	return func(c *config) {
		c.backendKind = "sdk"
		c.sdkAPIKey = apiKey
		for _, opt := range o {
			opt(c)
		}
	}
}

// WithSDKBaseURL overrides the API base URL (e.g. a proxy or a test server).
func WithSDKBaseURL(url string) SDKOption {
	return func(c *config) { c.sdkBaseURL = url }
}

// WithSDKHTTPClient sets the underlying HTTP client for the sdk backend.
func WithSDKHTTPClient(h *http.Client) SDKOption {
	return func(c *config) { c.sdkHTTP = h }
}
