package llmdriver

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

type capturedHTTPRequest struct {
	method string
	path   string
	header http.Header
	body   map[string]any
}

func newAPITestServer(t *testing.T, status int, response string) (*httptest.Server, *capturedHTTPRequest) {
	t.Helper()

	captured := new(capturedHTTPRequest)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		captured.method = request.Method
		captured.path = request.URL.Path
		captured.header = request.Header.Clone()
		if err := json.Unmarshal(body, &captured.body); err != nil {
			t.Errorf("decode request: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, response)
	}))
	t.Cleanup(server.Close)
	return server, captured
}
