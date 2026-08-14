package core

import "fmt"

// Feature identifies an optional generation capability that a target either
// supports or rejects with ErrUnsupportedFeature. The baseline surface (system,
// messages, MaxTokens, streaming) is unconditional and has no Feature: every
// target supports it. Later phases flip cells in the capability table below as
// they implement each feature.
type Feature string

const (
	// FeatureStructuredOutput requests a JSON-schema-constrained response.
	FeatureStructuredOutput Feature = "structured output"
	// FeatureReasoningEffort requests a reasoning-effort level.
	FeatureReasoningEffort Feature = "reasoning effort"
	// FeatureReasoningVisibility requests visible reasoning or thinking output.
	FeatureReasoningVisibility Feature = "reasoning visibility"
	// FeatureTools requests caller-executed tool calling.
	FeatureTools Feature = "tool calling"
	// FeatureImageInput requests image content in the input.
	FeatureImageInput Feature = "image input"
	// FeatureDocumentInput requests PDF or document content in the input.
	FeatureDocumentInput Feature = "document input"
	// FeatureSamplingParams requests temperature or top-p sampling controls.
	FeatureSamplingParams Feature = "sampling params"
	// FeatureTopK requests top-k sampling, which openai/api lacks.
	FeatureTopK Feature = "top-k sampling"
	// FeatureStopSequences requests stop sequences, which openai/api lacks.
	FeatureStopSequences Feature = "stop sequences"
	// FeatureWebSearch requests provider-executed web search.
	FeatureWebSearch Feature = "web search"
	// FeatureUserID requests an abuse-tracking end-user identifier.
	FeatureUserID Feature = "end-user id"
	// FeatureContinuation requests conversation continuation by identifier.
	FeatureContinuation Feature = "conversation continuation"
	// FeatureTokenCounting requests standalone token counting.
	FeatureTokenCounting Feature = "token counting"
)

// allFeatures lists every Feature in parity-inventory order. It backs the
// completeness tests and the README capability matrix.
var allFeatures = []Feature{
	FeatureStructuredOutput,
	FeatureReasoningEffort,
	FeatureReasoningVisibility,
	FeatureTools,
	FeatureImageInput,
	FeatureDocumentInput,
	FeatureSamplingParams,
	FeatureTopK,
	FeatureStopSequences,
	FeatureWebSearch,
	FeatureUserID,
	FeatureContinuation,
	FeatureTokenCounting,
}

type target struct {
	provider Provider
	flavor   Flavor
}

// featureSupport maps each target to the optional features it honors. A feature
// absent from a target's set is rejected with ErrUnsupportedFeature. Later
// phases add the cells they implement.
var featureSupport = map[target]map[Feature]bool{
	{ProviderClaude, FlavorAPI}: {FeatureStructuredOutput: true},
	{ProviderClaude, FlavorCLI}: {FeatureStructuredOutput: true},
	{ProviderOpenAI, FlavorAPI}: {FeatureStructuredOutput: true},
	{ProviderOpenAI, FlavorCLI}: {FeatureStructuredOutput: true},
}

// AllFeatures returns every Feature in parity-inventory order. It lets callers
// (and the README drift test) enumerate the full feature set without importing
// the unexported backing slice.
func AllFeatures() []Feature {
	return append([]Feature(nil), allFeatures...)
}

// Supports reports whether the target honors the feature.
func Supports(provider Provider, flavor Flavor, feature Feature) bool {
	return featureSupport[target{provider, flavor}][feature]
}

// featuresOf returns the optional features the request exercises, in inventory
// order. Each later phase adds one detection case as it adds a field, and the
// gate below picks it up automatically.
func featuresOf(r Request) []Feature {
	var features []Feature
	if r.OutputSchema != nil {
		features = append(features, FeatureStructuredOutput)
	}
	return features
}

// ValidateRequestFor reports whether the target accepts every feature the
// request exercises. Generate and GenerateStream call it after ValidateRequest;
// an unsupported feature returns ErrUnsupportedFeature naming the feature and
// the target. With today's Request fields it always passes.
func ValidateRequestFor(provider Provider, flavor Flavor, r Request) error {
	return validateFeaturesFor(provider, flavor, featuresOf(r))
}

// validateFeaturesFor is the capability gate: it rejects the first feature the
// target does not support. Split from ValidateRequestFor so the gate can be
// tested directly while featuresOf still detects nothing.
func validateFeaturesFor(provider Provider, flavor Flavor, features []Feature) error {
	for _, feature := range features {
		if !Supports(provider, flavor, feature) {
			return fmt.Errorf("%w: %s not supported by %s", ErrUnsupportedFeature, feature, formatTarget(provider, flavor))
		}
	}
	return nil
}
