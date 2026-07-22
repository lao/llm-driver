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

	// cli backend
	cliPath      string
	cliExtraArgs []string
	cliRunner    commandRunner
}

// Option configures a Client. Exactly one backend option (WithSDK or WithCLI)
// must be supplied to New.
type Option func(*config)

// SDKOption tunes the sdk (HTTP) backend. Pass to WithSDK.
type SDKOption func(*config)

// CLIOption tunes the cli backend. Pass to WithCLI.
type CLIOption func(*config)

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

// WithCLI selects the backend that shells out to a local Claude Code CLI.
func WithCLI(o ...CLIOption) Option {
	return func(c *config) {
		c.backendKind = "cli"
		for _, opt := range o {
			opt(c)
		}
	}
}

// WithCLIPath sets the claude binary to invoke. Defaults to "claude" on PATH.
func WithCLIPath(path string) CLIOption {
	return func(c *config) { c.cliPath = path }
}

// WithCLIExtraArgs appends verbatim arguments to every CLI invocation. Escape
// hatch for flags the library does not model; use with care.
func WithCLIExtraArgs(args ...string) CLIOption {
	return func(c *config) { c.cliExtraArgs = append(c.cliExtraArgs, args...) }
}

// withCLIRunner injects a command runner. Unexported: for tests only.
func withCLIRunner(r commandRunner) CLIOption {
	return func(c *config) { c.cliRunner = r }
}
