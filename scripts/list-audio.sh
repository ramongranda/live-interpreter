#!/usr/bin/env bash
set -euo pipefail

wpctl status

echo
echo "PipeWire output ports:"
pw-link -o || true

echo
echo "PipeWire input ports:"
pw-link -i || true
