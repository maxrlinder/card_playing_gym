#!/usr/bin/env bash
# Run `cardgym <game> train` under a restart loop that recycles its process.
#
# usage: tools/train_supervised.sh GAME RUN CONFIG [train options...]
#
# The trainer runs with --recycle-after (default 1000 iterations, or
# CARDGYM_RECYCLE_AFTER). At the first checkpoint past that many iterations it
# exits with status 75 and this loop resumes it from that checkpoint in a
# fresh process. Any other status ends the loop and is passed through.
#
# Why: native heap that PyTorch's MPS backend leaks per operation is only
# returned to the OS when the process exits. An autorelease pool per iteration
# removes most of it in-process; recycling bounds the rest. A restart costs one
# model load and warm-up iteration and loses no iterations.
set -u

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GAME RUN CONFIG [train options...]" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
game="$1"
run_name="$2"
config_path="$3"
shift 3
recycle_after="${CARDGYM_RECYCLE_AFTER:-1000}"

cd "$project_root"
while true; do
  "${project_root}/.venv/bin/python" -m cardgym "$game" train "$run_name" \
    --config "$config_path" --recycle-after "$recycle_after" "$@"
  status=$?
  if [[ $status -ne 75 ]]; then
    exit "$status"
  fi
  printf '[%s] recycled %s; resuming from its latest checkpoint\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_name"
done
