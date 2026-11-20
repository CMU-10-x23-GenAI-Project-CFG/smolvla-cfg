#!/bin/sh
set -eu
cd "$(dirname "$0")"
export MODAL_PROFILE="${MODAL_PROFILE:-akshitr}"
exec "${MODAL:-modal}" "$@"
