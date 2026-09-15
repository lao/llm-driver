//go:build !(aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows)

package backends

import "os/exec"

func configureCommand(*exec.Cmd) {}

// reapCommandGroup is a no-op on platforms without process groups here.
func reapCommandGroup(*exec.Cmd) {}
