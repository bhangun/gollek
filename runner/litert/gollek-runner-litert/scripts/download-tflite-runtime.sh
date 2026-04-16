#!/usr/bin/env bash
set -euo pipefail

OS="$(uname -s)"
ARCH="$(uname -m)"

OUT_DIR="${GOLLEK_TFLITE_LIB_DIR:-$HOME/.gollek/libs}"
mkdir -p "${OUT_DIR}"

RUNTIME_URL="${GOLLEK_TFLITE_RUNTIME_URL:-}"
GITHUB_REPO="${GOLLEK_LITERT_RUNTIME_REPO:-bhangun/gollek}"
RELEASE_TAG="${GOLLEK_LITERT_RUNTIME_RELEASE:-latest}"

case "${OS}" in
  Darwin)
    LIB_NAME="libtensorflowlite_c.dylib"
    LM_LIB_NAME="liblitert_lm.dylib"
    PLATFORM="macos"
    ;;
  Linux)
    LIB_NAME="libtensorflowlite_c.so"
    LM_LIB_NAME="liblitert_lm.so"
    PLATFORM="linux"
    ;;
  *)
    echo "Unsupported OS: ${OS}"
    exit 1
    ;;
esac

if [[ "${ARCH}" != "arm64" && "${ARCH}" != "aarch64" && "${ARCH}" != "x86_64" ]]; then
  echo "Unsupported architecture: ${ARCH}"
  exit 1
fi

ARCH_TAG="${ARCH}"
if [[ "${ARCH}" == "aarch64" ]]; then
  ARCH_TAG="arm64"
fi

echo "LiteRT runtime download helper"
echo "OS: ${OS} (${ARCH})"
echo "Output:"
echo "  Core: ${OUT_DIR}/${LIB_NAME}"
echo "  LLM:  ${OUT_DIR}/${LM_LIB_NAME}"
echo ""

if [[ -f "${OUT_DIR}/${LIB_NAME}" && -f "${OUT_DIR}/${LM_LIB_NAME}" ]]; then
  echo "✓ LiteRT core and LLM runtimes already exist in ${OUT_DIR}/"
  exit 0
fi

extract_and_copy() {
  local archive_path="$1"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  case "${archive_path}" in
    *.zip)
      if ! command -v unzip >/dev/null 2>&1; then
        echo "unzip not found. Install unzip or extract manually."
        exit 1
      fi
      unzip -q "${archive_path}" -d "${tmp_dir}"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "${archive_path}" -C "${tmp_dir}"
      ;;
    *)
      echo "Unknown archive format: ${archive_path}"
      exit 1
      ;;
  esac
  
  # Copy Core Library
  local found_core
  found_core="$(find "${tmp_dir}" -type f -name "${LIB_NAME}" | head -n 1 || true)"
  if [[ -n "${found_core}" ]]; then
    cp "${found_core}" "${OUT_DIR}/${LIB_NAME}"
    echo "✓ Saved ${OUT_DIR}/${LIB_NAME}"
  else
    echo "Could not find ${LIB_NAME} inside archive."
  fi
  
  # Copy LLM Library
  local found_llm
  found_llm="$(find "${tmp_dir}" -type f -name "${LM_LIB_NAME}" | head -n 1 || true)"
  if [[ -n "${found_llm}" ]]; then
    cp "${found_llm}" "${OUT_DIR}/${LM_LIB_NAME}"
    echo "✓ Saved ${OUT_DIR}/${LM_LIB_NAME}"
  else
    echo "Could not find ${LM_LIB_NAME} inside archive (LLM features may be disabled)."
  fi
}

if [[ -n "${RUNTIME_URL}" ]]; then
  echo "Downloading runtime from: ${RUNTIME_URL}"
  case "${RUNTIME_URL}" in
    *.tar.gz|*.tgz|*.zip)
      tmp_file="$(mktemp)"
      curl -L --fail --retry 3 --retry-delay 2 -o "${tmp_file}" "${RUNTIME_URL}"
      extract_and_copy "${tmp_file}"
      exit 0
      ;;
    *)
      curl -L --fail --retry 3 --retry-delay 2 -o "${OUT_DIR}/${LIB_NAME}" "${RUNTIME_URL}"
      echo "✓ Saved single library to ${OUT_DIR}/${LIB_NAME}"
      exit 0
      ;;
  esac
fi

DEFAULT_ASSET="${GOLLEK_LITERT_RUNTIME_ASSET:-litert-runtime-${PLATFORM}-${ARCH_TAG}.tar.gz}"
API_URL="https://api.github.com/repos/${GITHUB_REPO}/releases/${RELEASE_TAG}"
echo "Attempting GitHub release download:"
echo "  repo: ${GITHUB_REPO}"
echo "  release: ${RELEASE_TAG}"
echo "  asset: ${DEFAULT_ASSET}"

if command -v curl >/dev/null 2>&1; then
  json="$(curl -sSL "${API_URL}" || echo "")"
  url="$(echo "${json}" | grep -o "\"browser_download_url\": \"[^\"]*${DEFAULT_ASSET}[^\"]*\"" | head -n 1 | cut -d '"' -f 4 || true)"
  if [[ -n "${url}" ]]; then
    tmp_file="$(mktemp)"
    curl -L --fail --retry 3 --retry-delay 2 -o "${tmp_file}" "${url}"
    extract_and_copy "${tmp_file}"
    exit 0
  fi
fi

echo "This script does not fetch binaries automatically unless available in the release."
echo "Download the LiteRT and LiteRT-LM C libraries for your platform, then copy them to:"
echo "  ${OUT_DIR}/${LIB_NAME}"
echo "  ${OUT_DIR}/${LM_LIB_NAME}"
echo ""
echo "Tips:"
echo "- macOS: build from source or use a prebuilt package."
echo "- Linux: use a distro package or build from source."
echo ""
echo "If you already have the libraries, run:"
echo "  cp /path/to/${LIB_NAME} ${OUT_DIR}/"
echo "  cp /path/to/${LM_LIB_NAME} ${OUT_DIR}/"
