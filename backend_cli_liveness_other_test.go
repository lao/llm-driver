//go:build !unix

package llmwrapper

// descendantAlive is a no-op stub on platforms without signal-based liveness
// checks; the descendant-exit assertion runs only on Unix builds.
func descendantAlive(int) bool {
	return false
}
