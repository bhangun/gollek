#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPI_SRC="$ROOT_DIR/core/gollek-spi/src/main/java"
OUTPUT_FILE="$ROOT_DIR/docs/error-codes.md"
BUILD_DIR="${TMPDIR:-/tmp}/gollek-error-codes"

mkdir -p "$BUILD_DIR"

JAVAC_BIN=""
JAVA_BIN=""

if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/javac" && -x "${JAVA_HOME}/bin/java" ]]; then
  JAVAC_BIN="${JAVA_HOME}/bin/javac"
  JAVA_BIN="${JAVA_HOME}/bin/java"
else
  JAVAC_BIN="$(command -v javac || true)"
  JAVA_BIN="$(command -v java || true)"
fi

if [[ -z "$JAVAC_BIN" || -z "$JAVA_BIN" ]]; then
  echo "Error: javac/java not found. Please set JAVA_HOME or add Java to PATH." >&2
  exit 1
fi

# Compile only what we need
"$JAVAC_BIN" \
  -d "$BUILD_DIR" \
  "$SPI_SRC/tech/kayys/gollek/spi/error/ErrorCode.java" \
  "$SPI_SRC/tech/kayys/gollek/spi/error/ErrorCodeDoc.java"

# Generate markdown from ErrorCodeDoc
"$JAVA_BIN" -cp "$BUILD_DIR" tech.kayys.gollek.spi.error.ErrorCodeDoc > "$OUTPUT_FILE"

echo "Wrote $OUTPUT_FILE"
