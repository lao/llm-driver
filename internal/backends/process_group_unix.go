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
// NOTE: After Wait returns the process-group leader's PID is free for reuse.
// If it is recycled as a new group leader before this kill, an unrelated group
// would be hit. The window is extremely narrow and the same pattern exists in
// configureCommand.Cancel.
func reapCommandGroup(command *exec.Cmd) {
	if command.Process == nil {
		return
	}
	_ = syscall.Kill(-command.Process.Pid, syscall.SIGKILL)
}
