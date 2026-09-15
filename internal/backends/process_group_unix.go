//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package backends

import (
	"errors"
	"os"
	"os/exec"
	"syscall"
)

func configureCommand(command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	command.Cancel = func() error {
		err := syscall.Kill(-command.Process.Pid, syscall.SIGKILL)
		if errors.Is(err, syscall.ESRCH) {
			return os.ErrProcessDone
		}
		return err
	}
}

// reapCommandGroup kills anything still running in the command's process group
// after it exits, so a provider CLI that spawned a background child cannot leave
// it running past the call. It is a no-op when the process never started.
//
// Callers must skip this when the context's Cancel hook already killed the group
// (before Wait returned, while the PID was still valid) to avoid a redundant
// kill on a potentially-recycled PID.
//
// NOTE: After Wait returns the process-group leader's PID is free for reuse.
// If it is recycled as a new group leader before this kill, an unrelated group
// would be hit. The window is extremely narrow; eliminating it entirely would
// require pidfd_send_signal (Linux 5.1+), which is out of scope.
func reapCommandGroup(command *exec.Cmd) {
	if command.Process == nil {
		return
	}
	// Only ESRCH (no such process/group) is expected here — the group exited
	// before we could signal it.  Other errors such as EPERM would indicate
	// the PGID was recycled and the signal reached an unrelated group; we
	// suppress only ESRCH so the distinction is explicit.
	if err := syscall.Kill(-command.Process.Pid, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
		_ = err // no logger available; left for future instrumentation
	}
}
