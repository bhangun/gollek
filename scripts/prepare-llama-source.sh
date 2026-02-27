#!/usr/bin/env bash
set -euo pipefail

LLAMA_SRC="${GOLEK_LLAMA_SOURCE_DIR:-$HOME/.gollek/source/vendor/llama.cpp}"
LLAMA_REF="${GOLEK_LLAMA_REF:-origin/master}"

mkdir -p "$(dirname "$LLAMA_SRC")"

if [ ! -d "$LLAMA_SRC/.git" ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_SRC"
fi

git -C "$LLAMA_SRC" fetch --depth 1 origin

TARGET_REF="$LLAMA_REF"
if ! git -C "$LLAMA_SRC" show-ref --verify --quiet "refs/remotes/${TARGET_REF}"; then
  if git -C "$LLAMA_SRC" show-ref --verify --quiet "refs/remotes/origin/main"; then
    TARGET_REF="origin/main"
  elif git -C "$LLAMA_SRC" show-ref --verify --quiet "refs/remotes/origin/master"; then
    TARGET_REF="origin/master"
  else
    TARGET_REF="$(git -C "$LLAMA_SRC" for-each-ref --format='%(refname:short)' refs/remotes/origin | head -n 1)"
  fi
fi

if [ -z "$TARGET_REF" ]; then
  echo "Unable to resolve a remote ref for llama.cpp"
  exit 1
fi

git -C "$LLAMA_SRC" reset --hard "$TARGET_REF"

echo "Prepared llama.cpp source at: $LLAMA_SRC ($TARGET_REF)"
