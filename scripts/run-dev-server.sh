#!/bin/bash

set -euo pipefail

RESET="$(printf '\033[0m')"
BOLD="$(printf '\033[1m')"
GREEN="$(printf '\033[32m')"
CYAN="$(printf '\033[36m')"
YELLOW="$(printf '\033[33m')"
MAGENTA="$(printf '\033[35m')"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for arg in "$@"; do
  case "$arg" in
    -h|--help|--prewarm-plan|--prewarm-plan=*|--verify-fast-only|--verify-fast-only=*|--verify-fast-m4-smoke-only)
      exec bash "$ROOT_DIR/scripts/install-local-runtime.sh" "$@"
      ;;
  esac
done

source "$ROOT_DIR/scripts/module-selection-current.env"
ARCHITECTURE_VALUE="${ARCHITECTURE_TARGETS:-${ARCHITECTURE_TARGET:-native-java,binding}}"

if command -v /usr/libexec/java_home >/dev/null 2>&1; then
  JAVA_25_HOME="$(/usr/libexec/java_home -v 25 2>/dev/null || true)"
  if [[ -n "${JAVA_25_HOME}" ]]; then
    export JAVA_HOME="${JAVA_25_HOME}"
    export PATH="${JAVA_HOME}/bin:${PATH}"
  fi
fi

GRADLE_JAVA_HOME_ARG=()
if [[ -n "${JAVA_HOME:-}" ]]; then
  GRADLE_JAVA_HOME_ARG+=("-Dorg.gradle.java.home=${JAVA_HOME}")
fi
GRADLE_MAX_WORKERS_ARG=()
if [[ -n "${GOLLEK_GRADLE_MAX_WORKERS:-1}" ]]; then
  GRADLE_MAX_WORKERS_ARG+=("--max-workers=${GOLLEK_GRADLE_MAX_WORKERS:-1}")
fi

append_java_tool_option() {
  local option="$1"
  case " ${JAVA_TOOL_OPTIONS:-} " in
    *" ${option} "*) ;;
    *) JAVA_TOOL_OPTIONS="${option}${JAVA_TOOL_OPTIONS:+ ${JAVA_TOOL_OPTIONS}}" ;;
  esac
}

append_java_tool_option "--enable-native-access=ALL-UNNAMED"
append_java_tool_option "--add-modules=jdk.incubator.vector"
export JAVA_TOOL_OPTIONS

echo "${BOLD}${GREEN}:) Resolved backend targets:${RESET} ${BACKEND_TARGETS}"
echo "${BOLD}${GREEN}:) Resolved format targets:${RESET} ${FORMAT_TARGETS}"
echo "${BOLD}${GREEN}:) Resolved LLM targets:${RESET} ${LLM_TARGETS}"
echo "${BOLD}${MAGENTA}:) Architecture:${RESET} ${ARCHITECTURE_VALUE}"
echo "${BOLD}${MAGENTA}:) Profile:${RESET} ${BUILD_PROFILE}"
echo "${BOLD}${MAGENTA}:) Java home:${RESET} ${JAVA_HOME:-$(command -v java)}"
echo "${BOLD}${MAGENTA}:) Build mode:${RESET} Quarkus Dev"
echo "${BOLD}${GREEN}:) Selection manifest:${RESET} ${ROOT_DIR}/scripts/module-selection-current.env"
echo "$GREEN:) Module manifest resolved cleanly.$RESET"

cd "$ROOT_DIR"

QUARKUS_ARGS=""
for arg in "$@"; do
    if [[ "$arg" == *" "* ]]; then
        QUARKUS_ARGS="$QUARKUS_ARGS \"$arg\""
    else
        QUARKUS_ARGS="$QUARKUS_ARGS $arg"
    fi
done

if [[ -n "${QUARKUS_ARGS// /}" ]]; then
    exec ./gradlew :ui:gollek-cli:quarkusDev "--quarkus-args=$QUARKUS_ARGS" \
      -Pgollek.backend="${RESOLVED_BACKEND_PROPERTY}" \
      -Pgollek.profile="${BUILD_PROFILE}" \
      -Pgollek.model.formats="${FORMAT_TARGETS}" \
      -Pgollek.llm.types="${LLM_TARGETS}" \
      -Pgollek.architecture="${ARCHITECTURE_VALUE}"
else
    exec ./gradlew :ui:gollek-cli:quarkusDev \
      -Pgollek.backend="${RESOLVED_BACKEND_PROPERTY}" \
      -Pgollek.profile="${BUILD_PROFILE}" \
      -Pgollek.model.formats="${FORMAT_TARGETS}" \
      -Pgollek.llm.types="${LLM_TARGETS}" \
      -Pgollek.architecture="${ARCHITECTURE_VALUE}"
fi