package llmwrapper

import (
	"errors"
	"testing"
)

func TestConfigValidateAcceptsFourTargets(t *testing.T) {
	t.Parallel()

	for _, provider := range []Provider{ProviderClaude, ProviderOpenAI} {
		for _, flavor := range []Flavor{FlavorAPI, FlavorCLI} {
			config := Config{Provider: provider, Flavor: flavor, Model: "test-model"}
			if err := config.validate(); err != nil {
				t.Errorf("validate(%s/%s): %v", provider, flavor, err)
			}
		}
	}
}

func TestConfigValidateRejectsInvalidValues(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		config Config
	}{
		{name: "provider", config: Config{Provider: "anthropic", Flavor: FlavorAPI, Model: "model"}},
		{name: "flavor", config: Config{Provider: ProviderClaude, Flavor: "sdk", Model: "model"}},
		{name: "model", config: Config{Provider: ProviderClaude, Flavor: FlavorAPI, Model: "  "}},
		{name: "base URL", config: Config{Provider: ProviderClaude, Flavor: FlavorAPI, Model: "model", BaseURL: "://bad"}},
		{name: "CLI path", config: Config{Provider: ProviderClaude, Flavor: FlavorCLI, Model: "model", CLIPath: "  "}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			err := test.config.validate()
			if !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("validate() error = %v, want ErrInvalidConfig", err)
			}
		})
	}
}

func TestConfigValidateIgnoresFieldsForOtherFlavor(t *testing.T) {
	t.Parallel()

	tests := []Config{
		{Provider: ProviderClaude, Flavor: FlavorCLI, Model: "model", BaseURL: "://unused"},
		{Provider: ProviderOpenAI, Flavor: FlavorAPI, Model: "model", CLIPath: "  "},
	}
	for _, config := range tests {
		if err := config.validate(); err != nil {
			t.Errorf("validate(%s/%s): %v", config.Provider, config.Flavor, err)
		}
	}
}
