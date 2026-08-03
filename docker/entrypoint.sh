#!/usr/bin/env bash
set -euo pipefail

if [ "${START_SSHD:-0}" = "1" ]; then
    install -d -m 0700 /root/.ssh
    if [ -d /ssh-host ]; then
        cp -a /ssh-host/. /root/.ssh/
        chown -R root:root /root/.ssh
        chmod 0700 /root/.ssh
        find /root/.ssh -type f -exec chmod 0600 {} +
        find /root/.ssh -type f -name '*.pub' -exec chmod 0644 {} +
    fi
    cat > /root/.ssh/config <<'EOF'
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
    chown root:root /root/.ssh/config
    chmod 0600 /root/.ssh/config
    ssh-keygen -A
    mkdir -p /run/sshd
    /usr/sbin/sshd
fi

exec "$@"
