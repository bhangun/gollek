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

if [[ -z "${JAVA_HOME:-}" ]] && command -v /usr/libexec/java_home >/dev/null 2>&1; then
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
append_java_tool_option "-XX:MaxDirectMemorySize=24g"
export JAVA_TOOL_OPTIONS

# ── Metal dylib discovery ─────────────────────────────────────────────────────
# Enable native Metal elementwise/FFN/attention kernels (opt-in gate).
export ALJABR_METAL_ENABLE_ELEMENTWISE_KERNELS="${ALJABR_METAL_ENABLE_ELEMENTWISE_KERNELS:-1}"

# If the metal dylib hasn't been pointed at explicitly, probe standard locations
# relative to the repo root (which may differ from CWD when invoked from gollek/).
if [[ -z "${ALJABR_METAL_DYLIB:-}" ]]; then
  WAYANG_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
  for DYLIB_CANDIDATE in \
      "$WAYANG_ROOT/aljabr/backend/metal/aljabr-backend-metal/target/native/darwin-aarch64/libaljabr_metal.dylib" \
      "$ROOT_DIR/aljabr/backend/metal/aljabr-backend-metal/target/native/darwin-aarch64/libaljabr_metal.dylib" \
      "$HOME/.aljabr/libs/libaljabr_metal.dylib"; do
    if [[ -f "$DYLIB_CANDIDATE" ]]; then
      export ALJABR_METAL_DYLIB="$DYLIB_CANDIDATE"
      break
    fi
  done
fi

if [[ -n "${ALJABR_METAL_DYLIB:-}" ]]; then
  append_java_tool_option "-Daljabr.metal.dylib=${ALJABR_METAL_DYLIB}"
  echo "${BOLD}${CYAN}:) Metal dylib:${RESET} ${ALJABR_METAL_DYLIB}"
else
  echo "${BOLD}${YELLOW}:) Metal dylib not found — Metal backend will fall back to CPU${RESET}"
fi


echo "${BOLD}${GREEN}:) Resolved backend targets:${RESET} ${BACKEND_TARGETS}"
echo "${BOLD}${GREEN}:) Resolved format targets:${RESET} ${FORMAT_TARGETS}"
echo "${BOLD}${GREEN}:) Resolved LLM targets:${RESET} ${LLM_TARGETS}"
echo "${BOLD}${MAGENTA}:) Architecture:${RESET} ${ARCHITECTURE_VALUE}"
echo "${BOLD}${MAGENTA}:) Profile:${RESET} ${BUILD_PROFILE}"
echo "${BOLD}${MAGENTA}:) Java home:${RESET} ${JAVA_HOME:-$(command -v java)}"
echo "${BOLD}${MAGENTA}:) Build mode:${RESET} Local CLI"
echo "${BOLD}${GREEN}:) Selection manifest:${RESET} ${ROOT_DIR}/scripts/module-selection-current.env"
echo "$GREEN:) Module manifest resolved cleanly.$RESET"

cd "$ROOT_DIR"

JAR_FILE="ui/gollek-cli/build/gollek.jar"

# If the CLI jar is missing, try to build it automatically to avoid "Unable to access jarfile" errors.
if [[ ! -f "$JAR_FILE" ]]; then
  echo "${BOLD}${YELLOW}:) CLI JAR not found at $JAR_FILE — building...${RESET}"
  if [[ -x "./gradlew" ]]; then
    ./gradlew build -p ui/gollek-cli -x test
  else
    gradle build -p ui/gollek-cli -x test
  fi
fi



    if [[ "${BUILD_ONLY:-false}" == "true" || "${1:-}" == "--build-only" ]]; then
      echo "${BOLD}${YELLOW}:) Building latest CLI...${RESET}"
      ./gradlew :ui:gollek-cli:quarkusBuild \
        -Pgollek.backend="${RESOLVED_BACKEND_PROPERTY}" \
        -Pgollek.profile="${BUILD_PROFILE}" \
        -Pgollek.model.formats="${FORMAT_TARGETS}" \
        -Pgollek.llm.types="${LLM_TARGETS}" \
        -Pgollek.architecture="${ARCHITECTURE_VALUE}"
      exit 0
    fi

# Verify JAR exists and execute the built application
if [[ ! -f "$JAR_FILE" ]]; then
  echo "${BOLD}${RED}Error:${RESET} Unable to access jarfile $JAR_FILE" >&2
  exit 1
fi

exec java ${JAVA_TOOL_OPTIONS:-} -jar "$JAR_FILE" "$@"