package backends

import (
	"net/http"

	"github.com/lao/llmwrapper/internal/core"
)

type apiBackendConfig struct {
	model      string
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

type cliBackendConfig struct {
	model        string
	path         string
	args         []string
	runner       core.CommandRunner
	streamRunner streamingCommandRunner
}

// withCLIDefaults supplies the executable and runners a CLI backend falls back
// to when the caller left them unset, and copies the caller's extra arguments.
func withCLIDefaults(config cliBackendConfig, defaultPath string) cliBackendConfig {
	if config.path == "" {
		config.path = defaultPath
	}
	if config.runner == nil {
		config.runner = defaultCommandRunner
	}
	if config.streamRunner == nil {
		config.streamRunner = defaultStreamingCommandRunner
	}
	config.args = append([]string(nil), config.args...)
	return config
}

// New builds the backend selected by the config. A nil runner selects the
// default CLI command runner for CLI flavors.
func New(config core.Config, runner core.CommandRunner) (core.Backend, error) {
	apiConfig := apiBackendConfig{
		model:      config.Model,
		apiKey:     config.APIKey,
		baseURL:    config.BaseURL,
		httpClient: config.HTTPClient,
	}
	cliConfig := cliBackendConfig{
		model:  config.Model,
		path:   config.CLIPath,
		args:   config.CLIArgs,
		runner: runner,
	}

	switch {
	case config.Provider == core.ProviderClaude && config.Flavor == core.FlavorAPI:
		return newAnthropicBackend(apiConfig), nil
	case config.Provider == core.ProviderClaude && config.Flavor == core.FlavorCLI:
		return newClaudeCLIBackend(cliConfig), nil
	case config.Provider == core.ProviderOpenAI && config.Flavor == core.FlavorAPI:
		return newOpenAIBackend(apiConfig), nil
	case config.Provider == core.ProviderOpenAI && config.Flavor == core.FlavorCLI:
		return newCodexCLIBackend(cliConfig), nil
	default:
		return nil, &core.Error{
			Provider:  config.Provider,
			Flavor:    config.Flavor,
			Operation: "client.new",
			Err:       core.ErrInvalidConfig,
		}
	}
}
