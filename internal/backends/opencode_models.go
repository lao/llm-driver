package backends

import (
	"context"
	"strings"

	"github.com/lao/llm-driver/internal/core"
)

// OpenCodeModels runs `opencode models` and returns the provider/model ids it
// prints, in the order opencode printed them. An empty path selects "opencode"
// on PATH; a nil runner selects the default command runner. Lines that are not a
// launchable provider/model id (blank lines, error text, option-like values) are
// dropped.
func OpenCodeModels(ctx context.Context, path string, runner core.CommandRunner) ([]string, error) {
	if path == "" {
		path = "opencode"
	}
	if runner == nil {
		runner = defaultCommandRunner
	}
	stdout, stderr, exitCode, err := runner(ctx, path, []string{"models"}, nil)
	if err != nil {
		return nil, &core.Error{
			Provider:  core.ProviderOpenCode,
			Flavor:    core.FlavorCLI,
			Operation: "models",
			Message:   strings.TrimSpace(string(stderr)),
			Err:       err,
		}
	}
	if exitCode != 0 {
		message := strings.TrimSpace(string(stderr))
		if message == "" {
			message = "opencode models failed"
		}
		return nil, &core.Error{
			Provider:  core.ProviderOpenCode,
			Flavor:    core.FlavorCLI,
			Operation: "models",
			Status:    exitCode,
			Message:   message,
		}
	}
	return parseOpenCodeModels(stdout), nil
}

// parseOpenCodeModels keeps the provider/model ids from `opencode models`
// stdout. Blank lines, error text, and anything not launchable are dropped;
// #variant suffixes and nested provider/a/b ids are preserved verbatim, because
// --model takes the id as-is.
func parseOpenCodeModels(stdout []byte) []string {
	var models []string
	for _, line := range strings.Split(string(stdout), "\n") {
		line = strings.TrimSpace(line)
		if openCodeModelID(line) {
			models = append(models, line)
		}
	}
	return models
}

// openCodeModelID reports whether a line is one provider/model id safe to pass
// to --model: a non-empty segment either side of the first "/", built only from
// characters real ids use and not option-like.
func openCodeModelID(line string) bool {
	provider, model, found := strings.Cut(line, "/")
	return found && provider != "" && model != "" && safeCLIOption(line)
}

// safeCLIOption mirrors the option-injection guard muxcontrol applies before a
// discovered id reaches a launch command.
func safeCLIOption(value string) bool {
	if value == "" || strings.HasPrefix(value, "-") {
		return false
	}
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
		case r == '-' || r == '.' || r == '_' || r == '/' || r == ':' ||
			r == '[' || r == ']' || r == '#' || r == '@':
		default:
			return false
		}
	}
	return true
}
