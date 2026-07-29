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

cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/module-selection-current.env"
ARCHITECTURE_VALUE="${ARCHITECTURE_TARGETS:-${ARCHITECTURE_TARGET:-native-java,binding}}"
PUBLISH_RUNTIME_LOCAL="${GOLLEK_PUBLISH_RUNTIME_LOCAL:-false}"
export GOLLEK_SKIP_LEGACY_SOURCE_HOTPATCHES="${GOLLEK_SKIP_LEGACY_SOURCE_HOTPATCHES:-true}"

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

bool_enabled() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off) return 1 ;;
    *) return 0 ;;
  esac
}

stop_build_daemons_after_install() {
  if ! bool_enabled "${GOLLEK_INSTALL_STOP_BUILD_DAEMONS:-true}"; then
    echo "${YELLOW}⚠ Leaving Gradle build daemons running. Set GOLLEK_INSTALL_STOP_BUILD_DAEMONS=true to restore benchmark hygiene.${RESET}"
    return 0
  fi

  if [[ ! -x "$ROOT_DIR/gradlew" ]]; then
    return 0
  fi

  echo "${BOLD}${CYAN}:) Benchmark hygiene:${RESET} stopping Gradle build daemons before local safetensor runs"
  "$ROOT_DIR/gradlew" "${GRADLE_JAVA_HOME_ARG[@]}" --stop >/dev/null 2>&1 || true

  local platform_root
  platform_root="$(cd "$ROOT_DIR/.." && pwd)"
  local -a gradle_homes=(
    "$ROOT_DIR/.gradle-sandbox"
    "$platform_root/.gradle-sandbox"
  )
  local seen=":"
  local gradle_home
  for gradle_home in "${gradle_homes[@]}"; do
    if [[ ! -d "$gradle_home" || "$seen" == *":$gradle_home:"* ]]; then
      continue
    fi
    seen="${seen}${gradle_home}:"
    GRADLE_USER_HOME="$gradle_home" "$ROOT_DIR/gradlew" "${GRADLE_JAVA_HOME_ARG[@]}" --stop >/dev/null 2>&1 || true
  done
}

is_legacy_gradle_module() {
  case "$1" in
    :ml:*|:trainer:*|:examples:gollek-ml-examples|:models:gollek-model-common|:runner:onnx:gollek-ml-export-onnx|:runner:onnx:gollek-ml-onnx|:runner:diffuser:gollek-diffuser) return 0 ;;
    *) return 1 ;;
  esac
}

read -r -a MODULE_ARRAY <<< "${MODULES:-}"
SUPPORTED_MODULES=()
SKIPPED_MODULES=()
for module in "${MODULE_ARRAY[@]}"; do
  if is_legacy_gradle_module "$module"; then
    SKIPPED_MODULES+=("$module")
  else
    SUPPORTED_MODULES+=("$module")
  fi
done

GRADLE_INSTALL_TASKS=()
for module in "${SUPPORTED_MODULES[@]}"; do
  GRADLE_INSTALL_TASKS+=("${module}:publishToMavenLocal")
done

echo "${BOLD}${CYAN}:) Resolved backend targets:${RESET} ${BACKEND_TARGETS}"
echo "${BOLD}${CYAN}:) Resolved format targets:${RESET} ${FORMAT_TARGETS}"
echo "${BOLD}${CYAN}:) Resolved LLM targets:${RESET} ${LLM_TARGETS}"
echo "${BOLD}${MAGENTA}:) Architecture:${RESET} ${ARCHITECTURE_VALUE}"
echo "${BOLD}${MAGENTA}:) Profile:${RESET} ${BUILD_PROFILE}"
echo "${BOLD}${MAGENTA}:) Java home:${RESET} ${JAVA_HOME:-$(command -v java)}"
echo "${BOLD}${MAGENTA}:) Build mode:${RESET} Gradle-only install"
echo "${BOLD}${GREEN}:) Selection manifest:${RESET} ${ROOT_DIR}/scripts/module-selection-current.env"
echo "$GREEN:) Module manifest resolved cleanly.$RESET"

if [[ ${#SKIPPED_MODULES[@]} -gt 0 ]]; then
  echo "${BOLD}${YELLOW}:| Warning:${RESET} skipping legacy Gradle modules in local install wrapper:"
  for module in "${SKIPPED_MODULES[@]}"; do
    echo "   - ${module}"
  done
fi

echo "${BOLD}${GREEN}:) Checking for local alkhawarizm project (publishing to mavenLocal)${RESET}"
ALKHAWARIZM_DIR="$(cd "$ROOT_DIR/.." && cd alkhawarizm 2>/dev/null && pwd || true)"
if [[ -n "${ALKHAWARIZM_DIR}" && -d "${ALKHAWARIZM_DIR}" ]]; then
  echo "${BOLD}${CYAN}:) Found alkhawarizm at ${ALKHAWARIZM_DIR}${RESET}"
  if [[ -f "${ALKHAWARIZM_DIR}/gradlew" || -x "${ALKHAWARIZM_DIR}/gradlew" ]]; then
    echo "${BOLD}${GREEN}:) Publishing alkhawarizm to mavenLocal for local development${RESET}"
    (cd "${ALKHAWARIZM_DIR}" && ./gradlew :core:alkhawarizm-core:publishToMavenLocal --no-parallel -x test "${GRADLE_JAVA_HOME_ARG[@]}" "${GRADLE_MAX_WORKERS_ARG[@]}" ) || echo "${YELLOW}⚠ alkhawarizm publish failed; continuing without it${RESET}"
  else
    echo "${YELLOW}⚠ No gradle wrapper in alkhawarizm; skipping publish.${RESET}"
  fi
fi

echo "${BOLD}${GREEN}:) Step 1/2:${RESET} packaging and installing gollek via the macOS local installer"
echo "${BOLD}${YELLOW}:| Install path:${RESET} ${GOLLEK_BIN_DIR:-$HOME/.local/bin}/gollek"
bash "$ROOT_DIR/scripts/install-local-runtime.sh" "$@"

echo "${BOLD}${GREEN}:) Step 2/2:${RESET} optional compatibility publish to local artifact cache"
if [[ "${PUBLISH_RUNTIME_LOCAL}" == "true" ]]; then
  if [[ ${#GRADLE_INSTALL_TASKS[@]} -gt 0 ]]; then
    if ./gradlew "${GRADLE_JAVA_HOME_ARG[@]}" "${GRADLE_MAX_WORKERS_ARG[@]}" "${GRADLE_INSTALL_TASKS[@]}" \
      -Pgollek.backend="${RESOLVED_BACKEND_PROPERTY}" \
      -Pgollek.profile="${BUILD_PROFILE}" \
      -Pgollek.model.formats="${FORMAT_TARGETS}" \
      -Pgollek.llm.types="${LLM_TARGETS}" \
      -Pgollek.architecture="${ARCHITECTURE_VALUE}"; then
      echo "${GREEN}✓ Published supported runtime artifacts to local artifact cache${RESET}"
    else
      echo "${YELLOW}⚠ Optional Gradle publish failed after local install; CLI install is still complete.${RESET}"
    fi
  else
    echo "${YELLOW}⚠ No supported Gradle runtime modules were selected for publish.${RESET}"
  fi
else
  echo "${YELLOW}⚠ Skipping local artifact publish by default. Set GOLLEK_PUBLISH_RUNTIME_LOCAL=true to enable it.${RESET}"
fi

stop_build_daemons_after_install

echo "${BOLD}${GREEN}:D Done.${RESET} Gollek is installed locally, and compatibility publish is optional."
