package llmdriver

import (
	"context"
	"errors"
	"testing"
)

func TestListModelsKeepsProviderModelIDsAndDropsNoise(t *testing.T) {
	t.Parallel()

	stdout := []byte(`opencode/big-pickle

  anthropic/claude-sonnet-4-5  
openrouter/anthropic/claude-sonnet-4.5#high
opencode-go/deepseek-v4.1-flash
Error: Provider not found
not-a-model
provider/model with space
--evil/model
-/x
`)
	runner := &fakeCommandRunner{stdout: stdout}
	models, err := listModels(context.Background(), "opencode", runner.run)
	if err != nil {
		t.Fatalf("listModels(): %v", err)
	}

	want := []string{
		"opencode/big-pickle",
		"anthropic/claude-sonnet-4-5",
		"openrouter/anthropic/claude-sonnet-4.5#high",
		"opencode-go/deepseek-v4.1-flash",
	}
	if len(models) != len(want) {
		t.Fatalf("models = %#v, want %#v", models, want)
	}
	for i := range want {
		if models[i] != want[i] {
			t.Fatalf("models = %#v, want %#v", models, want)
		}
	}
	runner.assertCall(t, "opencode", []string{"models"}, "")
}

func TestListModelsEmptyOrErrorOutputIsEmpty(t *testing.T) {
	t.Parallel()

	for _, stdout := range []string{"", "Error: Provider not found\n", "/\nfoo/\n/bar\n"} {
		runner := &fakeCommandRunner{stdout: []byte(stdout)}
		models, err := listModels(context.Background(), "opencode", runner.run)
		if err != nil {
			t.Fatalf("listModels(%q): %v", stdout, err)
		}
		if len(models) != 0 {
			t.Fatalf("listModels(%q) = %#v, want empty", stdout, models)
		}
	}
}

func TestListModelsErrors(t *testing.T) {
	t.Parallel()

	t.Run("missing executable", func(t *testing.T) {
		t.Parallel()

		runner := &fakeCommandRunner{err: ErrExecutableNotFound}
		_, err := listModels(context.Background(), "opencode", runner.run)
		if !errors.Is(err, ErrExecutableNotFound) {
			t.Fatalf("listModels() error = %v, want ErrExecutableNotFound", err)
		}
	})

	t.Run("nonzero exit", func(t *testing.T) {
		t.Parallel()

		runner := &fakeCommandRunner{stderr: []byte("boom\n"), exitCode: 3}
		_, err := listModels(context.Background(), "opencode", runner.run)
		var target *Error
		if !errors.As(err, &target) {
			t.Fatalf("listModels() error = %T %v, want *Error", err, err)
		}
		if target.Provider != ProviderOpenCode || target.Flavor != FlavorCLI || target.Operation != "models" {
			t.Fatalf("error context = %+v", target)
		}
		if target.Status != 3 || target.Message != "boom" {
			t.Fatalf("status/message = %d/%q, want 3/%q", target.Status, target.Message, "boom")
		}
	})
}
