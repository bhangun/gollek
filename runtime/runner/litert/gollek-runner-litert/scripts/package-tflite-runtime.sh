#!/usr/bin/env bash
set -euo pipefail

OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
  Darwin)
    LIB_NAME="libtensorflowlite_c.dylib"
    PLATFORM="macos"
    ;;
  Linux)
    LIB_NAME="libtensorflowlite_c.so"
    PLATFORM="linux"
    ;;
  *)
    echo "Unsupported OS: ${OS}"
    exit 1
    ;;
esac

ARCH_TAG="${ARCH}"
if [[ "${ARCH}" == "aarch64" ]]; then
  ARCH_TAG="arm64"
fi

LIB_PATH="${LITERT_LIBRARY_PATH:-${1:-}}"
if [[ -z "${LIB_PATH}" ]]; then
  for candidate in \
    "$HOME/.gollek/libs/${LIB_NAME}" \
    "/usr/local/lib/${LIB_NAME}" \
    "/usr/lib/${LIB_NAME}" \
    "./lib/${LIB_NAME}"
  do
    if [[ -f "${candidate}" ]]; then
      LIB_PATH="${candidate}"
      break
    fi
  done
fi

if [[ -z "${LIB_PATH}" || ! -f "${LIB_PATH}" ]]; then
  echo "LiteRT runtime not found. Provide via LITERT_LIBRARY_PATH or copy ${LIB_NAME} into ~/.gollek/libs"
  exit 1
fi

OUT_DIR="${GOLLEK_LITERT_RUNTIME_OUT_DIR:-$(pwd)/dist}"
mkdir -p "${OUT_DIR}"

ASSET_NAME="${GOLLEK_LITERT_RUNTIME_ASSET:-litert-runtime-${PLATFORM}-${ARCH_TAG}.tar.gz}"
OUT_PATH="${OUT_DIR}/${ASSET_NAME}"

echo "Packaging LiteRT runtime"
echo "Library: ${LIB_PATH}"
echo "Output:  ${OUT_PATH}"

tar -czf "${OUT_PATH}" -C "$(dirname "${LIB_PATH}")" "$(basename "${LIB_PATH}")"
echo "✓ Created ${OUT_PATH}"
