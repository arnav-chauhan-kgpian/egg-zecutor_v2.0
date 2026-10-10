#!/bin/sh
# Convert this WSL2 distro's cgroup hierarchy from v2 (unified) to v1 (legacy).
#
# WHY
#   Judge0 1.13.x sandboxes submissions with `isolate`, which is hard-wired to
#   cgroup v1 paths (/sys/fs/cgroup/memory/box-0/). WSL2's init always mounts
#   cgroup2 at /sys/fs/cgroup regardless of the kernel command line or whether
#   systemd is enabled, so on a stock distro every Judge0 submission fails with
#   "Internal Error".
#
#   The kernel supports both layouts; a controller may only live in one
#   hierarchy at a time. With systemd disabled nothing has claimed the
#   controllers yet (root cgroup.subtree_control is empty and there are no
#   child cgroups), so the v2 root can be unmounted and the controllers
#   remounted as individual v1 hierarchies.
#
# WHEN
#   Runs at distro boot via /etc/wsl.conf:
#
#     [boot]
#     systemd=false
#     command=/usr/local/sbin/wsl-cgroup-v1.sh
#
#   It must run BEFORE dockerd starts, otherwise Docker claims the controllers
#   on the v2 hierarchy and the v1 mounts fail with EBUSY.
#
# Idempotent: exits successfully if v1 is already in place.
set -e

CG=/sys/fs/cgroup

# Controllers to expose. `cpu,cpuacct` is mounted as one comounted hierarchy,
# which is the layout isolate and Docker both expect from a v1 host.
CONTROLLERS="memory cpu,cpuacct cpuset blkio devices freezer net_cls,net_prio perf_event pids hugetlb"

if [ "$(stat -fc %T "$CG")" = "tmpfs" ] && [ -d "$CG/memory" ]; then
    echo "cgroup v1 already mounted"
    exit 0
fi

if [ -n "$(cat "$CG/cgroup.subtree_control" 2>/dev/null)" ]; then
    echo "ERROR: v2 controllers are already delegated ($CG/cgroup.subtree_control)." >&2
    echo "       Something claimed them before this ran — check that systemd is" >&2
    echo "       disabled and dockerd has not started yet." >&2
    exit 1
fi

umount "$CG" || {
    echo "ERROR: could not unmount the v2 hierarchy at $CG — it is in use." >&2
    exit 1
}

mount -t tmpfs -o mode=755 cgroup_root "$CG"

for controller in $CONTROLLERS; do
    # A comounted hierarchy is conventionally named after its first controller
    # (cpu,cpuacct -> /sys/fs/cgroup/cpu,cpuacct), which is what isolate and
    # Docker look for.
    dir="$CG/$controller"
    mkdir -p "$dir"
    if mount -t cgroup -o "$controller" cgroup "$dir" 2>/dev/null; then
        # Provide the individual controller names as symlinks, matching the
        # layout systemd produces on a real v1 host.
        case "$controller" in
            *,*)
                first=${controller%%,*}
                second=${controller#*,}
                ln -sfn "$dir" "$CG/$first"
                ln -sfn "$dir" "$CG/$second"
                ;;
        esac
    else
        echo "warning: controller '$controller' unavailable, skipping" >&2
        rmdir "$dir" 2>/dev/null || true
    fi
done

# isolate requires these two above all else; fail loudly rather than let Judge0
# come up and return Internal Error for every submission.
for required in memory cpuacct cpuset; do
    if [ ! -d "$CG/$required" ]; then
        echo "ERROR: required controller '$required' was not mounted." >&2
        exit 1
    fi
done

echo "cgroup v1 hierarchy mounted:"
ls -1 "$CG"
