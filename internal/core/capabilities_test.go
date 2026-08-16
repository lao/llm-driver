package core

import (
	"errors"
	"strings"
	"testing"
)

func allTargets() []target {
	return []target{
		{ProviderClaude, FlavorAPI},
		{ProviderClaude, FlavorCLI},
		{ProviderOpenAI, FlavorAPI},
		{ProviderOpenAI, FlavorCLI},
	}
}

func TestCapabilityTableCoversEveryTargetAndFeature(t *testing.T) {
	t.Parallel()

	for _, tg := range allTargets() {
		if _, ok := featureSupport[tg]; !ok {
			t.Fatalf("capability table missing target %s/%s", tg.provider, tg.flavor)
		}
	}
	if len(featureSupport) != len(allTargets()) {
		t.Fatalf("capability table has %d targets, want %d", len(featureSupport), len(allTargets()))
	}

	seen := map[Feature]bool{}
	for _, feature := range allFeatures {
		if feature == "" {
			t.Fatal("allFeatures contains an empty feature")
		}
		if seen[feature] {
			t.Fatalf("allFeatures lists %q twice", feature)
		}
		seen[feature] = true
	}
}

// Structured output and reasoning effort are implemented on every target. Every
// other gated feature is still rejected everywhere.
func TestSupportsMatchesImplementedFeatures(t *testing.T) {
	t.Parallel()

	for _, tg := range allTargets() {
		for _, feature := range allFeatures {
			want := feature == FeatureStructuredOutput || feature == FeatureReasoningEffort
			if got := Supports(tg.provider, tg.flavor, feature); got != want {
				t.Errorf("Supports(%s/%s, %q) = %v, want %v", tg.provider, tg.flavor, feature, got, want)
			}
		}
	}
}

// TestEffortAllowedSetsPerTarget pins the per-target allowed effort values: the
// value is provider-validated, never remapped. openai/api tops out at high;
// the other three accept the full Anthropic-style superset.
func TestEffortAllowedSetsPerTarget(t *testing.T) {
	t.Parallel()

	full := []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax}
	tests := []struct {
		tg      target
		accepts []Effort
		rejects []Effort
	}{
		{target{ProviderClaude, FlavorAPI}, full, nil},
		{target{ProviderClaude, FlavorCLI}, full, nil},
		{target{ProviderOpenAI, FlavorCLI}, full, nil},
		{target{ProviderOpenAI, FlavorAPI}, []Effort{EffortLow, EffortMedium, EffortHigh}, []Effort{EffortXHigh, EffortMax}},
	}
	for _, test := range tests {
		for _, value := range test.accepts {
			request := Request{MaxTokens: 16, Messages: []Message{User("hi")}, Effort: value}
			if err := ValidateRequestFor(test.tg.provider, test.tg.flavor, request); err != nil {
				t.Errorf("%s/%s effort %q = %v, want accepted", test.tg.provider, test.tg.flavor, value, err)
			}
		}
		for _, value := range test.rejects {
			request := Request{MaxTokens: 16, Messages: []Message{User("hi")}, Effort: value}
			err := ValidateRequestFor(test.tg.provider, test.tg.flavor, request)
			if !errors.Is(err, ErrUnsupportedFeature) {
				t.Errorf("%s/%s effort %q = %v, want ErrUnsupportedFeature", test.tg.provider, test.tg.flavor, value, err)
			}
			for _, want := range []string{string(value), formatTarget(test.tg.provider, test.tg.flavor)} {
				if err == nil || !strings.Contains(err.Error(), want) {
					t.Errorf("%s/%s effort %q error = %v, missing %q", test.tg.provider, test.tg.flavor, value, err, want)
				}
			}
		}
	}
}

// TestEffortRejectsUnknownValue confirms a value outside every target's set is
// rejected loudly rather than passed through.
func TestEffortRejectsUnknownValue(t *testing.T) {
	t.Parallel()

	request := Request{MaxTokens: 16, Messages: []Message{User("hi")}, Effort: Effort("turbo")}
	for _, tg := range allTargets() {
		if err := ValidateRequestFor(tg.provider, tg.flavor, request); !errors.Is(err, ErrUnsupportedFeature) {
			t.Errorf("%s/%s effort %q = %v, want ErrUnsupportedFeature", tg.provider, tg.flavor, request.Effort, err)
		}
	}
}

func TestValidateRequestForIsNoOpForTodaysRequest(t *testing.T) {
	t.Parallel()

	request := Request{MaxTokens: 16, Messages: []Message{User("hello")}}
	for _, tg := range allTargets() {
		if err := ValidateRequestFor(tg.provider, tg.flavor, request); err != nil {
			t.Errorf("ValidateRequestFor(%s/%s) = %v, want nil", tg.provider, tg.flavor, err)
		}
	}
}

func TestCapabilityGateRejectsUnsupportedFeature(t *testing.T) {
	t.Parallel()

	err := validateFeaturesFor(ProviderClaude, FlavorAPI, []Feature{FeatureTools})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("gate error = %v, want ErrUnsupportedFeature", err)
	}
	for _, want := range []string{string(FeatureTools), "claude/api"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("gate error = %q, missing %q", err.Error(), want)
		}
	}
}

func TestCapabilityGatePassesSupportedFeatures(t *testing.T) {
	t.Parallel()

	if err := validateFeaturesFor(ProviderClaude, FlavorAPI, nil); err != nil {
		t.Fatalf("gate rejected an empty feature set: %v", err)
	}
}
