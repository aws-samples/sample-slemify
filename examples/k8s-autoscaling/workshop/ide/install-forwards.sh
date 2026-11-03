#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Install the workshop port-forward service on the IDE instance. Run as root
# by the IDE bootstrap (static/workshop-stack.yaml, step InstallPortForwards).
# Idempotent.
#
# Installs:
#   /usr/local/bin/slemify-forward              the per-port supervisor
#   /etc/slemify/forwards.conf                  the port table (demo/forwards.conf)
#   /etc/systemd/system/slemify-forward@.service
#   /etc/systemd/system/slemify-forwards.target
# and enables one slemify-forward@<port> per row of the table.
#
# /etc/slemify/forward.env (region, cluster name, kubeconfig path) is written
# by the bootstrap, not here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_SRC="$HERE/../../demo/forwards.conf"

install -d -m 0755 /etc/slemify
install -m 0644 "$CONF_SRC" /etc/slemify/forwards.conf
install -m 0755 "$HERE/slemify-forward.sh" /usr/local/bin/slemify-forward
install -m 0644 "$HERE/slemify-forward@.service" /etc/systemd/system/slemify-forward@.service
install -m 0644 "$HERE/slemify-forwards.target" /etc/systemd/system/slemify-forwards.target

if [ ! -f /etc/slemify/forward.env ]; then
  echo "WARNING: /etc/slemify/forward.env is missing; the forwards will not start until it exists." >&2
fi

systemctl daemon-reload
ports="$(grep -v '^[[:space:]]*#' /etc/slemify/forwards.conf | awk 'NF >= 5 { print $1 }')"
for port in $ports; do
  systemctl enable "slemify-forward@$port.service"
done
systemctl enable --now slemify-forwards.target
echo "Installed port-forwards for: $(echo "$ports" | tr '\n' ' ')"
