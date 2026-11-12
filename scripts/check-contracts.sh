#!/usr/bin/env bash
# Verify contracts/ integrity, and optionally that it matches the other repo.
#
#   scripts/check-contracts.sh                      # local files vs checksum.txt
#   scripts/check-contracts.sh --against ../web-app # also compare the two checkouts
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$here/contracts/python/checksum.py"

if [[ "${1:-}" == "--against" ]]; then
  other="${2:?usage: scripts/check-contracts.sh --against <path-to-other-repo>}"
  if [[ ! -d "$other/contracts" ]]; then
    echo "no contracts/ in $other" >&2
    exit 66
  fi
  mine="$(python3 "$here/contracts/python/checksum.py" --print-combined)"
  theirs="$(python3 "$other/contracts/python/checksum.py" --print-combined)"
  if [[ "$mine" != "$theirs" ]]; then
    echo >&2
    echo "contract mismatch between repos:" >&2
    echo "  $here -> $mine" >&2
    echo "  $other -> $theirs" >&2
    echo >&2
    diff -r "$here/contracts" "$other/contracts" >&2 || true
    echo >&2
    echo "Sync with: scripts/sync-contracts.sh $other" >&2
    exit 1
  fi
  echo "both repos agree on contract $(cat "$here/contracts/VERSION") ($mine)"
fi
