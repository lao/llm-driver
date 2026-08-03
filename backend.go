package llmwrapper

import (
	"context"
	"net/http"
)

type backend interface {
	generate(context.Context, Request) (*Response, error)
	generateStream(context.Context, Request) (chunkStream, error)
}

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
	runner       commandRunner
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

func buildBackend(config Config, runner commandRunner) (backend, error) {
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
	case config.Provider == ProviderClaude && config.Flavor == FlavorAPI:
		return newAnthropicBackend(apiConfig), nil
	case config.Provider == ProviderClaude && config.Flavor == FlavorCLI:
		return newClaudeCLIBackend(cliConfig), nil
	case config.Provider == ProviderOpenAI && config.Flavor == FlavorAPI:
		return newOpenAIBackend(apiConfig), nil
	case config.Provider == ProviderOpenAI && config.Flavor == FlavorCLI:
		return newCodexCLIBackend(cliConfig), nil
	default:
		return nil, &Error{
			Provider:  config.Provider,
			Flavor:    config.Flavor,
			Operation: "client.new",
			Err:       ErrInvalidConfig,
		}
	}
}
