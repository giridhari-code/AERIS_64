#!/usr/bin/env bash
# Back-compat wrapper → official installer
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.sh" "$@"
