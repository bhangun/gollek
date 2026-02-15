#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/target"

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
PROMPT1="${PROMPT1:-who are you}"
PROMPT2="${PROMPT2:-answer in one short sentence: what is 2+2?}"

NATIVE_BIN="${GOLEK_CLI_NATIVE_BIN:-$TARGET_DIR/golek-cli-1.0.0-SNAPSHOT-runner}"
RUNNER_JAR="${GOLEK_CLI_RUNNER_JAR:-$TARGET_DIR/golek-cli-1.0.0-SNAPSHOT-runner.jar}"

TMP_OUT="$(mktemp /tmp/golek-gguf-smoke.XXXXXX)"
trap 'rm -f "$TMP_OUT"' EXIT

run_native() {
  local quiet_flag=""
  if "$NATIVE_BIN" chat --help 2>/dev/null | rg -q -- "--quiet"; then
    quiet_flag="--quiet"
  fi
  printf '%s\n%s\nexit\n' "$PROMPT1" "$PROMPT2" | \
    GOLEK_CLI_FILE_LOG=false QUARKUS_LOG_FILE_ENABLED=false \
    "$NATIVE_BIN" chat --provider gguf --model "$MODEL" --temperature 0.2 --max-tokens 64 $quiet_flag >"$TMP_OUT" 2>&1
}

run_jar() {
  local quiet_flag=""
  if java -jar "$RUNNER_JAR" chat --help 2>/dev/null | rg -q -- "--quiet"; then
    quiet_flag="--quiet"
  fi
  printf '%s\n%s\nexit\n' "$PROMPT1" "$PROMPT2" | \
    GOLEK_CLI_FILE_LOG=false QUARKUS_LOG_FILE_ENABLED=false \
    java -jar "$RUNNER_JAR" chat --provider gguf --model "$MODEL" --temperature 0.2 --max-tokens 64 $quiet_flag >"$TMP_OUT" 2>&1
}

set +e
if [[ -x "$NATIVE_BIN" ]]; then
  MODE="native"
  run_native
elif [[ -f "$RUNNER_JAR" ]]; then
  MODE="jar"
  run_jar
else
  echo "ERROR: no runnable CLI found."
  echo "Expected native: $NATIVE_BIN"
  echo "Expected jar:    $RUNNER_JAR"
  RC=2
  set -e
  exit "$RC"
fi
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
  if rg -n "Goodbye!" "$TMP_OUT" >/dev/null; then
    echo "FAIL: CLI exited with code $RC ($MODE) after completing chat (shutdown crash)."
  else
    echo "FAIL: CLI exited with code $RC ($MODE)."
  fi
  echo "---- output tail ----"
  tail -n 120 "$TMP_OUT"
  exit "$RC"
fi

if rg -n "GGML_ASSERT|abort|Stream error:|Failed to initialize GGUF Provider|FATAL:" "$TMP_OUT" >/dev/null; then
  echo "FAIL: runtime error detected ($MODE)."
  echo "---- output tail ----"
  tail -n 80 "$TMP_OUT"
  exit 1
fi

if ! rg -n "Assistant:" "$TMP_OUT" >/dev/null; then
  echo "FAIL: no assistant output detected ($MODE)."
  echo "---- output tail ----"
  tail -n 80 "$TMP_OUT"
  exit 1
fi

if ! rg -n "(?i)\\b4\\b|four" "$TMP_OUT" >/dev/null; then
  echo "WARN: relevance check weak (could not find '4' in second answer)."
  echo "---- output tail ----"
  tail -n 80 "$TMP_OUT"
  exit 1
fi

echo "PASS: GGUF smoke check ($MODE) passed."
echo "Output log: $TMP_OUT"
tail -n 30 "$TMP_OUT"
