package core

import (
	"fmt"
	"strings"
)

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

// effortFull is every Effort value; claude/api, claude/cli, codex/cli, and
// opencode/cli accept it (verified against anthropic-sdk-go OutputConfigEffort,
// `claude --effort` help, Codex's pass-through -c config, and opencode's
// pass-through --variant). effortToHigh caps openai/api at EffortHigh: the
// neutral superset is Anthropic-style and OpenAI's Responses reasoning.effort
// tops out at high for these values.
var (
	effortFull   = effortSet(EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax)
	effortToHigh = effortSet(EffortLow, EffortMedium, EffortHigh)
)

func effortSet(values ...Effort) map[Effort]bool {
	set := make(map[Effort]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}

// featureSupport maps each target to the optional features it honors. A feature
// absent from a target's map is rejected with ErrUnsupportedFeature. The value is
// the neutral values a valued feature accepts (never remapped, so a value outside
// the set is rejected); an empty map marks a plain boolean feature with no value
// constraint. Later phases add the cells they implement.
var featureSupport = map[target]map[Feature]map[Effort]bool{
	{ProviderClaude, FlavorAPI}: {
		FeatureStructuredOutput: {},
		FeatureReasoningEffort:  effortFull,
		FeatureTools:            {},
		FeatureSamplingParams:   {},
		FeatureTopK:             {},
		FeatureStopSequences:    {},
		FeatureUserID:           {},
	},
	{ProviderClaude, FlavorCLI}: {
		FeatureStructuredOutput: {},
		FeatureReasoningEffort:  effortFull,
	},
	{ProviderOpenAI, FlavorAPI}: {
		FeatureStructuredOutput: {},
		FeatureReasoningEffort:  effortToHigh,
		FeatureTools:            {},
		FeatureSamplingParams:   {},
		FeatureUserID:           {},
	},
	{ProviderOpenAI, FlavorCLI}: {
		FeatureStructuredOutput: {},
		FeatureReasoningEffort:  effortFull,
	},
	{ProviderOpenCode, FlavorCLI}: {
		FeatureStructuredOutput: {},
		FeatureReasoningEffort:  effortFull,
	},
}

// AllFeatures returns every Feature in parity-inventory order. It lets callers
// (and the README drift test) enumerate the full feature set without importing
// the unexported backing slice.
func AllFeatures() []Feature {
	return append([]Feature(nil), allFeatures...)
}

// OpenCodeAcceptsEffort reports whether Request.Effort can be expressed for the
// model: opencode 2.x encodes effort as the model id's #variant suffix, so the
// model must be set and must not already carry a variant. This is the one
// definition of that model-dependent rule; Client.Supports and the opencode
// backend both call it.
func OpenCodeAcceptsEffort(model string) bool {
	return model != "" && !strings.Contains(model, "#")
}

// Supports reports whether the target honors the feature.
func Supports(provider Provider, flavor Flavor, feature Feature) bool {
	_, ok := featureSupport[target{provider, flavor}][feature]
	return ok
}

// featuresOf returns the optional features the request exercises, in inventory
// order. Each later phase adds one detection case as it adds a field, and the
// gate below picks it up automatically.
func featuresOf(r Request) []Feature {
	var features []Feature
	if r.OutputSchema != nil {
		features = append(features, FeatureStructuredOutput)
	}
	if r.Effort != "" {
		features = append(features, FeatureReasoningEffort)
	}
	if len(r.Tools) > 0 || r.ToolChoice.Mode != ToolChoiceAuto || requestUsesParts(r) {
		features = append(features, FeatureTools)
	}
	if r.Temperature != nil || r.TopP != nil {
		features = append(features, FeatureSamplingParams)
	}
	if r.TopK != nil {
		features = append(features, FeatureTopK)
	}
	if len(r.StopSequences) > 0 {
		features = append(features, FeatureStopSequences)
	}
	if r.UserID != "" {
		features = append(features, FeatureUserID)
	}
	return features
}

// requestUsesParts reports whether any message uses the Parts content form,
// which requires tool support even without Tools or a non-auto ToolChoice. Any
// Parts usage counts, not just tool parts: a target that cannot render Parts
// (the CLI flavors) must reject the request loudly rather than silently drop
// the content that only lives in Parts.
func requestUsesParts(r Request) bool {
	for _, message := range r.Messages {
		if len(message.Parts) > 0 {
			return true
		}
	}
	return false
}

// ValidateRequestFor reports whether the target accepts every feature the
// request exercises. Generate and GenerateStream call it after ValidateRequest;
// an unsupported feature returns ErrUnsupportedFeature naming the feature and
// the target. With today's Request fields it always passes.
func ValidateRequestFor(provider Provider, flavor Flavor, r Request) error {
	if err := validateFeaturesFor(provider, flavor, featuresOf(r)); err != nil {
		return err
	}
	// Valued features: the neutral Effort value must be in the target's allowed
	// set (openai/api tops out at EffortHigh). The feature-support check above
	// already ran, so a supported target has a non-nil allowed set here.
	if r.Effort != "" {
		allowed := featureSupport[target{provider, flavor}][FeatureReasoningEffort]
		if !allowed[r.Effort] {
			return fmt.Errorf("%w: reasoning effort %q not supported by %s", ErrUnsupportedFeature, r.Effort, formatTarget(provider, flavor))
		}
	}
	return nil
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
