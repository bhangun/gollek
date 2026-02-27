#!/usr/bin/env bash
set -euo pipefail

LLAMA_SRC="${GOLEK_LLAMA_SOURCE_DIR:-$HOME/.gollek/source/vendor/llama.cpp}"
LLAMA_REF="${GOLEK_LLAMA_REF:-origin/master}"

mkdir -p "$(dirname "$LLAMA_SRC")"

if [ ! -d "$LLAMA_SRC/.git" ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_SRC"
fi

git -C "$LLAMA_SRC" fetch --depth 1 origin
git -C "$LLAMA_SRC" checkout -q "${LLAMA_REF#origin/}" 2>/dev/null || true
git -C "$LLAMA_SRC" reset --hard "$LLAMA_REF"

echo "Prepared llama.cpp source at: $LLAMA_SRC"
