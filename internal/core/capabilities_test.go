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

// Structured output is the only feature implemented so far, and every target
// supports it. Every other gated feature is still rejected everywhere.
func TestSupportsMatchesImplementedFeatures(t *testing.T) {
	t.Parallel()

	for _, tg := range allTargets() {
		for _, feature := range allFeatures {
			want := feature == FeatureStructuredOutput
			if got := Supports(tg.provider, tg.flavor, feature); got != want {
				t.Errorf("Supports(%s/%s, %q) = %v, want %v", tg.provider, tg.flavor, feature, got, want)
			}
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

	err := validateFeaturesFor(ProviderClaude, FlavorAPI, []Feature{FeatureReasoningEffort})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("gate error = %v, want ErrUnsupportedFeature", err)
	}
	for _, want := range []string{string(FeatureReasoningEffort), "claude/api"} {
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
