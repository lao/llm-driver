//go:build windows

package backends

import (
	"os"
	"os/exec"
	"strconv"
)

// configureCommand terminates the whole provider process tree on
// cancellation. Windows has no process groups equivalent to Unix's, so
// CommandContext would kill only the CLI process and leave any helpers it
// spawned running. taskkill's /T flag terminates the process together with
// its descendants.
func configureCommand(command *exec.Cmd) {
	command.Cancel = func() error {
		if command.Process == nil {
			return os.ErrProcessDone
		}
		kill := exec.Command("taskkill", "/F", "/T", "/PID", strconv.Itoa(command.Process.Pid))
		if err := kill.Run(); err != nil {
			// taskkill exits non-zero when the process already exited.
			if command.ProcessState != nil {
				return os.ErrProcessDone
			}
			return err
		}
		return nil
	}
}

// reapCommandGroup terminates any descendants left after the command exits, so
// a provider CLI cannot leave a background process running past the call.
//
// Callers must skip this when the context's Cancel hook already killed the tree
// (before Wait returned, while the PID was still valid) to avoid a redundant
// kill on a potentially-recycled PID.
//
// NOTE: Because this runs after Wait returns, the PID is free for reuse. On
// Windows, PID recycling can be fast, and if the PID is reassigned between
// Wait and the taskkill invocation an unrelated process tree would be killed.
// The race window is extremely narrow; eliminating it entirely would require a
// Windows job object, which is out of scope.
func reapCommandGroup(command *exec.Cmd) {
	if command.Process == nil {
		return
	}
	_ = exec.Command("taskkill", "/F", "/T", "/PID", strconv.Itoa(command.Process.Pid)).Run()
}
