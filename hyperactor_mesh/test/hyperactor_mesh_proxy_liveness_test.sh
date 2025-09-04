#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

BIN="$1"

# Launch client in background; capture its PID
set +e
"$BIN" & BIN_PID=$!
set -e

# Wait for the client to finish
set +e
wait "$BIN_PID"
status=$?
set -e

# Accept 0 (normal) or 143 (SIGTERM)
if [[ $status -ne 0 && $status -ne 143 ]]; then
  echo "FAIL: Test binary exited with unexpected code $status"
  exit 1
fi

# Give children a moment to shut down
sleep 2

# Because the client calls setpgid(0,0) early, its process group ID ==
# its PID.
TARGET_PG="$BIN_PID"

# Find any remaining processes in that group (excluding the client
# itself)
mapfile -t MALINGERING < <(
  ps -eo pid=,pgid= | awk -v pg="$TARGET_PG" -v root="$BIN_PID" '
    $2 == pg && $1 != root { print $1 }'
)

if ((${#MALINGERING[@]} > 0)); then
  echo "FAIL: Found malingering processes in PGID $TARGET_PG"
  # Only call ps if we actually have PIDs
  ps -o pid,pgid,comm,args -p "${MALINGERING[@]}"
  exit 1
fi

echo "PASS: No malingering processes in PGID $TARGET_PG"
