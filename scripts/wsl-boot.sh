#!/bin/sh
# WSL2 boot hook: establish cgroup v1, then start the Docker daemon.
#
# Wired in via /etc/wsl.conf:
#
#   [boot]
#   systemd=false
#   command=/usr/local/sbin/wsl-boot.sh
#
# Order is load-bearing. dockerd claims the cgroup controllers on whichever
# hierarchy it finds; if it starts first the controllers are pinned to v2 and
# the v1 mounts fail with EBUSY, which is exactly the state that makes every
# Judge0 submission return "Internal Error".
set -e

# WSL runs [boot] command with a minimal environment. Without this dockerd
# cannot find docker-proxy and dies with "invalid userland-proxy-path".
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

/usr/local/sbin/wsl-cgroup-v1.sh

# systemd is deliberately disabled (it mounts cgroup v2 and cannot be talked
# out of it under WSL), so there is no service manager to start dockerd.
#
# setsid is required, not tidiness: WSL tears down the [boot] command's process
# group once it returns, so a plain `dockerd &` is reaped seconds after it
# finishes initialising — the log shows a clean "Daemon has completed
# initialization" and then the socket is gone. Detaching into its own session
# is what makes it outlive the hook.
if ! pidof dockerd >/dev/null 2>&1; then
    mkdir -p /var/log
    setsid nohup dockerd >>/var/log/dockerd.log 2>&1 < /dev/null &
fi
