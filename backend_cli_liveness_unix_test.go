//go:build unix

package llmwrapper

import "syscall"

// descendantAlive reports whether a process with the given PID still exists.
func descendantAlive(pid int) bool {
	return syscall.Kill(pid, 0) == nil
}
