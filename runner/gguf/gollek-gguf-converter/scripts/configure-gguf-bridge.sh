#!/usr/bin/env bash
set -euo pipefail

BRIDGE_DIR="$(cd "$(dirname "$0")/../../gguf-bridge" && pwd)"
BUILD_DIR="${BRIDGE_DIR}/build"
CACHE_FILE="${BUILD_DIR}/CMakeCache.txt"

mkdir -p "${BUILD_DIR}"

if [ -f "${CACHE_FILE}" ]; then
  CACHED_HOME="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${CACHE_FILE}" | head -n1 || true)"
  if [ -n "${CACHED_HOME}" ] && [ "${CACHED_HOME}" != "${BRIDGE_DIR}" ]; then
    echo "Detected stale CMake cache source path:"
    echo "  cached: ${CACHED_HOME}"
    echo "  actual: ${BRIDGE_DIR}"
    echo "Cleaning ${BUILD_DIR} ..."
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
  fi
fi

cmake -S "${BRIDGE_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
