package claudewrap

import (
	"context"
	"testing"
)

// TestContractNonStreaming asserts that both backends map an equivalent
// upstream reply to an identical neutral Response (aside from the Backend tag).
func TestContractNonStreaming(t *testing.T) {
	// Both fakes reply: text "Hello, world", usage 10/5, stop end_turn.
	sdkSrv, _ := sdkTestServer(t, 200, okBody) // okBody: "Hello, world", 10/5, end_turn
	sdkClient, err := New(WithSDK("k", WithSDKBaseURL(sdkSrv.URL)))
	if err != nil {
		t.Fatalf("sdk New: %v", err)
	}

	const cliBody = `{"type":"result","subtype":"success","is_error":false,` +
		`"result":"Hello, world","stop_reason":"end_turn","session_id":"s",` +
		`"usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}`
	cliClient, err := New(WithCLI(withCLIRunner((&fakeRun{stdout: []byte(cliBody)}).run)))
	if err != nil {
		t.Fatalf("cli New: %v", err)
	}

	params := MessageParams{MaxTokens: 64, Messages: []Message{UserMessage("hi")}}

	cases := []struct {
		name    string
		client  *Client
		backend Backend
	}{
		{"sdk", sdkClient, BackendSDK},
		{"cli", cliClient, BackendCLI},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := tc.client.Messages.New(context.Background(), params)
			if err != nil {
				t.Fatalf("Messages.New: %v", err)
			}
			if resp.Text() != "Hello, world" {
				t.Errorf("Text() = %q", resp.Text())
			}
			if resp.Role != RoleAssistant {
				t.Errorf("Role = %q", resp.Role)
			}
			if resp.StopReason != StopEndTurn {
				t.Errorf("StopReason = %q", resp.StopReason)
			}
			if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
				t.Errorf("Usage = %+v", resp.Usage)
			}
			if resp.Backend != tc.backend {
				t.Errorf("Backend = %q, want %q", resp.Backend, tc.backend)
			}
			if tc.client.Backend() != tc.backend {
				t.Errorf("Client.Backend() = %q, want %q", tc.client.Backend(), tc.backend)
			}
		})
	}
}
