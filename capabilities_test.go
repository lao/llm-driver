package llmdriver

import (
	"os"
	"strings"
	"testing"
)

// matrixTarget names a capability-matrix column and the target it maps to.
type matrixTarget struct {
	provider Provider
	flavor   Flavor
}

// matrixColumns is the README matrix column order, left to right.
var matrixColumns = []matrixTarget{
	{ProviderClaude, FlavorAPI},
	{ProviderOpenAI, FlavorAPI},
	{ProviderClaude, FlavorCLI},
	{ProviderOpenAI, FlavorCLI},
}

// matrixRows maps each gated Feature to its README matrix row label.
var matrixRows = []struct {
	feature Feature
	label   string
}{
	{FeatureStructuredOutput, "Structured output (JSON schema)"},
	{FeatureReasoningEffort, "Reasoning effort"},
	{FeatureReasoningVisibility, "Reasoning/thinking visibility"},
	{FeatureTools, "Tool calling (caller-executed)"},
	{FeatureImageInput, "Image input"},
	{FeatureDocumentInput, "PDF/document input"},
	{FeatureSamplingParams, "Temperature / TopP"},
	{FeatureTopK, "TopK"},
	{FeatureStopSequences, "Stop sequences"},
	{FeatureWebSearch, "Web search"},
	{FeatureUserID, "Metadata / end-user ID"},
	{FeatureContinuation, "Conversation continuation by ID"},
	{FeatureTokenCounting, "Token counting"},
}

// TestMatrixRowsCoverEveryFeature ties the README matrix rows to the source of
// truth: every Feature must have exactly one matrix row and vice versa. Without
// this, a later phase could add a Feature but forget its README row and every
// other test would still pass, letting docs silently drift.
func TestMatrixRowsCoverEveryFeature(t *testing.T) {
	t.Parallel()

	inRows := make(map[Feature]bool, len(matrixRows))
	for _, row := range matrixRows {
		if inRows[row.feature] {
			t.Errorf("feature %q listed in more than one matrix row", row.feature)
		}
		inRows[row.feature] = true
	}

	all := AllFeatures()
	if len(matrixRows) != len(all) {
		t.Errorf("matrixRows has %d entries, AllFeatures has %d", len(matrixRows), len(all))
	}
	for _, f := range all {
		if !inRows[f] {
			t.Errorf("feature %q has no README matrix row", f)
		}
	}
}

func clientFor(t *testing.T, mt matrixTarget) *Client {
	t.Helper()
	client, err := New(Config{Provider: mt.provider, Flavor: mt.flavor, Model: "model"})
	if err != nil {
		t.Fatalf("New(%s/%s): %v", mt.provider, mt.flavor, err)
	}
	return client
}

// TestREADMEMatrixMatchesSupports walks the README capability matrix and
// asserts every gated cell equals Client.Supports, so docs and the table
// cannot drift apart.
func TestREADMEMatrixMatchesSupports(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatalf("read README: %v", err)
	}
	readme := string(data)

	for _, row := range matrixRows {
		cells := matrixCells(t, readme, row.label)
		if len(cells) != len(matrixColumns) {
			t.Fatalf("row %q has %d cells, want %d", row.label, len(cells), len(matrixColumns))
		}
		for i, mt := range matrixColumns {
			want := mark(clientFor(t, mt).Supports(row.feature))
			if cells[i] != want {
				t.Errorf("README %q / %s/%s = %q, Supports says %q", row.label, mt.provider, mt.flavor, cells[i], want)
			}
		}
	}
}

func mark(supported bool) string {
	if supported {
		return "✓"
	}
	return "✗"
}

// matrixCells returns the four support cells of the matrix row whose first
// column exactly equals label.
func matrixCells(t *testing.T, readme, label string) []string {
	t.Helper()
	for _, line := range strings.Split(readme, "\n") {
		if !strings.HasPrefix(strings.TrimSpace(line), "|") {
			continue
		}
		fields := splitRow(line)
		if len(fields) >= 1 && fields[0] == label {
			return fields[1:]
		}
	}
	t.Fatalf("README matrix has no row %q", label)
	return nil
}

// splitRow splits a Markdown table row into its trimmed cells.
func splitRow(line string) []string {
	parts := strings.Split(strings.TrimSpace(line), "|")
	// A pipe table row has empty leading and trailing fields around the pipes.
	parts = parts[1 : len(parts)-1]
	cells := make([]string, len(parts))
	for i, p := range parts {
		cells[i] = strings.TrimSpace(p)
	}
	return cells
}

func TestSupportsMatchesImplementedFeatures(t *testing.T) {
	t.Parallel()

	for _, mt := range matrixColumns {
		client := clientFor(t, mt)
		for _, row := range matrixRows {
			// Structured output and reasoning effort are implemented on every
			// target; nothing else is yet.
			want := row.feature == FeatureStructuredOutput || row.feature == FeatureReasoningEffort
			if got := client.Supports(row.feature); got != want {
				t.Errorf("%s/%s Supports(%q) = %v, want %v", mt.provider, mt.flavor, row.feature, got, want)
			}
		}
	}
}
