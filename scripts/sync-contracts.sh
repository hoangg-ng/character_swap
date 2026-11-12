#!/usr/bin/env bash
# Copy contracts/ to the sibling repo so both copies stay byte-identical.
#
#   scripts/sync-contracts.sh ../web-app        # or ../Research
#
# Run this after an intended contract change, once the checksum has been regenerated
# (python contracts/python/checksum.py --write). Commit in both repos.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-}"

if [[ -z "$target" ]]; then
  echo "usage: scripts/sync-contracts.sh <path-to-other-repo>" >&2
  exit 64
fi
if [[ ! -d "$target/.git" ]]; then
  echo "not a git repo: $target" >&2
  exit 66
fi

python3 "$here/contracts/python/checksum.py" >/dev/null || {
  echo "refusing to sync: local contracts/ does not match its own checksum." >&2
  echo "Regenerate first: python3 contracts/python/checksum.py --write" >&2
  exit 65
}

rm -rf "$target/contracts"
cp -R "$here/contracts" "$target/contracts"
find "$target/contracts" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "synced contracts/ -> $target/contracts"
echo "contract version: $(cat "$here/contracts/VERSION")"
echo "combined digest:  $(python3 "$here/contracts/python/checksum.py" --print-combined)"
echo
echo "Now commit in $target as well, or the two repos are out of step."
