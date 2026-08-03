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
