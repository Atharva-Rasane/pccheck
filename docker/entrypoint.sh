#!/usr/bin/env bash
set -euo pipefail

if [ "${START_SSHD:-0}" = "1" ]; then
    if [ -d /ssh-host ]; then
        install -d -m 0700 /root/.ssh
        cp -a /ssh-host/. /root/.ssh/
        chmod 0700 /root/.ssh
        find /root/.ssh -type f -exec chmod 0600 {} +
        find /root/.ssh -type f -name '*.pub' -exec chmod 0644 {} +
    fi
    ssh-keygen -A
    mkdir -p /run/sshd
    /usr/sbin/sshd
fi

exec "$@"
