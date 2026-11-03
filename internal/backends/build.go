package backends

import (
	"net/http"

	"github.com/lao/llm-driver/internal/core"
)

type apiBackendConfig struct {
	model      string
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

type cliBackendConfig struct {
	acceptUnverified bool
	model            string
	path             string
	args             []string
	env              []string
	dir              string
	claude           core.ClaudeOptions
	codex            core.CodexOptions
	runner           core.CommandRunner
	streamRunner     streamingCommandRunner
}

// withCLIDefaults supplies the executable and runners a CLI backend falls back
// to when the caller left them unset, and copies the caller's extra arguments.
// The default runners carry the caller's environment and working directory so
// the injected-runner seam (used by tests) keeps its signature.
func withCLIDefaults(config cliBackendConfig, defaultPath string) cliBackendConfig {
	if config.path == "" {
		config.path = defaultPath
	}
	if config.runner == nil {
		config.runner = commandRunnerFor(config.env, config.dir)
	}
	if config.streamRunner == nil {
		config.streamRunner = streamingRunnerFor(config.env, config.dir)
	}
	config.args = append([]string(nil), config.args...)
	if config.env != nil {
		config.env = append([]string(nil), config.env...)
	}
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
		acceptUnverified: config.AcceptUnverifiedOpenCode,
		model:            config.Model,
		path:             config.CLIPath,
		args:             config.CLIArgs,
		env:              config.CLIEnv,
		dir:              config.CLIWorkDir,
		claude:           config.Claude,
		codex:            config.Codex,
		runner:           runner,
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
	case config.Provider == core.ProviderOpenCode && config.Flavor == core.FlavorCLI:
		return newOpenCodeCLIBackend(cliConfig), nil
	default:
		return nil, &core.Error{
			Provider:  config.Provider,
			Flavor:    config.Flavor,
			Operation: "client.new",
			Err:       core.ErrInvalidConfig,
		}
	}
}
