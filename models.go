package llmdriver

import (
	"context"

	"github.com/lao/llm-driver/internal/backends"
	"github.com/lao/llm-driver/internal/core"
)

// ListModels returns the model ids the local opencode CLI can use, in the order
// `opencode models` prints them.
//
// Each id has the form "provider/model" and is passed verbatim as Config.Model
// with ProviderOpenCode and FlavorCLI. Because opencode resolves its own
// provider credentials, the list reflects what the installed opencode can
// actually run. A missing or failing opencode binary returns an *Error.
func ListModels(ctx context.Context) ([]string, error) {
	return listModels(ctx, "", nil)
}

// listModels is ListModels with an injectable executable path and runner for
// tests; an empty path and nil runner select the defaults.
func listModels(ctx context.Context, path string, runner core.CommandRunner) ([]string, error) {
	return backends.OpenCodeModels(ctx, path, runner)
}
