#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PATTERN='tech\.kayys\.wayang|WAYANG_[A-Z0-9_]*|wayang\.(inference|multitenancy|home)|(^|[^[:alnum:]_])\.wayang(/|$)|Wayang-Gollek|Wayang/Gollek|Wayang API|wayang-inference'

CANDIDATE_PATHS=(
  framework
  runtime
  observability
  core
  plugins
  runner
  sdk
  spi
  ui/gollek-cli/src/main
  scripts/prepare-llama-source.sh
  scripts/prepare-libtorch-source.sh
)

SCAN_PATHS=()
for p in "${CANDIDATE_PATHS[@]}"; do
  if [ -e "$p" ]; then
    SCAN_PATHS+=("$p")
  fi
done

scan_with_rg() {
  rg -n --color never \
    --glob '!**/target/**' \
    --glob '!**/build/**' \
    --glob '!**/bin/**' \
    --glob '!**/.gradle/**' \
    --glob '!**/src/test/**' \
    --glob '!**/*.md' \
    --glob '!**/README*' \
    -e "$PATTERN" \
    "${SCAN_PATHS[@]}"
}

scan_with_grep() {
  local found=1
  while IFS= read -r -d '' file; do
    if grep -HnE -I "$PATTERN" "$file"; then
      found=0
    fi
  done < <(
    find "${SCAN_PATHS[@]}" \
      \( -path '*/target/*' \
      -o -path '*/build/*' \
      -o -path '*/bin/*' \
      -o -path '*/.gradle/*' \
      -o -path '*/src/test/*' \
      -o -name '*.md' \
      -o -name 'README*' \) -prune \
      -o -type f -print0
  )
  return "$found"
}

if command -v rg >/dev/null 2>&1; then
  scan_command=scan_with_rg
elif command -v grep >/dev/null 2>&1; then
  scan_command=scan_with_grep
else
  echo "ERROR: neither rg nor grep is available for product boundary scanning." >&2
  exit 2
fi

if "$scan_command"; then
  echo
  echo "Gollek product boundary violation: Gollek production code must not depend on Wayang names."
  echo "Use Gollek-owned APIs/config such as gollek.* properties and GOLLEK_* environment variables."
  exit 1
fi

echo "Gollek product boundary OK."
