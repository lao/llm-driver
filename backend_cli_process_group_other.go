//go:build !(aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows)

package llmwrapper

import "os/exec"

func configureCommand(*exec.Cmd) {}
