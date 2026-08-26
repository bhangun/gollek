#!/bin/bash
# Gollek - Local Development Installer
# Builds the CLI from source and installs a local shim for testing.
# Usage: ./scripts/install-local-runtime.sh [--native]

set -e

# --- Default Options ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRADLEW="${PROJECT_ROOT}/gradlew"
CLI_MODULE="ui/gollek-cli"
SDK_MODULE="sdk/gollek-sdk-local"
SAFETENSOR_ENGINE_MODULE="runner/safetensor/gollek-safetensor-engine"
METAL_MODULE="../alkhawarizm/backend/metal/alkhawarizm-backend-metal"
BIN_DIR="${GOLLEK_BIN_DIR:-${HOME}/.local/bin}"
LIB_DIR="${GOLLEK_LIB_DIR:-${HOME}/.gollek/libs}"
RUNTIME_DIR="${GOLLEK_RUNTIME_DIR:-${HOME}/.gollek/runtime}"
GOLLEK_RUNTIME_JAR="${RUNTIME_DIR}/gollek.jar"
GOLLEK_CLI_BIN="${BIN_DIR}/gollek"
GGUF_FAST_BRIDGE_SRC="${PROJECT_ROOT}/runner/gguf/gguf-fast-bridge/gollek_gguf_fast.cpp"
NATIVE_MODE=false
BUILD_ARGS="-c"
STOP_DAEMONS_ONLY=false
PRINT_PREWARM_PLAN=false
VERIFY_FAST_ONLY=false
VERIFY_FAST_PATHS="${GOLLEK_INSTALL_VERIFY_FAST_PATHS:-}"
SKIP_LEGACY_SOURCE_HOTPATCHES="${GOLLEK_SKIP_LEGACY_SOURCE_HOTPATCHES:-true}"
SKIP_BUILD="${GOLLEK_SKIP_BUILD:-false}"
INSTALL_PREWARM_MODELS="${GOLLEK_INSTALL_PREWARM_MODELS:-}"
INSTALL_PREWARM_MODELS_CONFIGURED=false
if [ -n "${GOLLEK_INSTALL_PREWARM_MODELS+x}" ]; then
    INSTALL_PREWARM_MODELS_CONFIGURED=true
fi

append_install_prewarm_models() {
    local models="$1"
    INSTALL_PREWARM_MODELS_CONFIGURED=true
    if [ -z "$models" ]; then
        return 0
    fi
    if [ -z "$INSTALL_PREWARM_MODELS" ]; then
        INSTALL_PREWARM_MODELS="$models"
    else
        INSTALL_PREWARM_MODELS="${INSTALL_PREWARM_MODELS},${models}"
    fi
}

apply_install_verify_m4_smoke_profile() {
    VERIFY_FAST_PATHS="safetensor"
    GOLLEK_INSTALL_VERIFY_FAST_AGGREGATE=true
    GOLLEK_INSTALL_VERIFY_FAST_REQUIRE_METAL=true
    GOLLEK_INSTALL_VERIFY_SAFETENSOR_PRESET=m4-smoke
    GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_PROFILE=true
    GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_METAL=true
    GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_METAL_PATHS=true
    GOLLEK_INSTALL_VERIFY_SAFETENSOR_REJECT_FALLBACK_PATHS=true
    GOLLEK_INSTALL_VERIFY_SAFETENSOR_MAX_TOKENS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_MAX_TOKENS:-3}"
}

if command -v /usr/libexec/java_home >/dev/null 2>&1; then
    JAVA_25_HOME="$(/usr/libexec/java_home -v 25 2>/dev/null || true)"
    if [ -n "$JAVA_25_HOME" ]; then
        export JAVA_HOME="$JAVA_25_HOME"
        export PATH="${JAVA_HOME}/bin:${PATH}"
    fi
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
# Force CPU backend for fast inference (disable Metal)
export GOLLEK_BACKEND=cpu
# Do not require Metal for fast-path verification/build
export GOLLEK_INSTALL_VERIFY_FAST_REQUIRE_METAL=false
export GOLLEK_ALLOW_METAL_BUILD_FAILURE=true

# --- Visuals ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Parse Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--native) NATIVE_MODE=true; BUILD_ARGS="$BUILD_ARGS -n" ;;
        -t|--tests) BUILD_ARGS="$BUILD_ARGS -t" ;;
        --skip-tests) BUILD_ARGS="$BUILD_ARGS" ;; # build.sh skips by default
        --stop-daemons-only) STOP_DAEMONS_ONLY=true ;;
        --prewarm)
            if [ "$#" -lt 2 ]; then
                echo -e "${RED}❌ --prewarm requires a model id, short id, or path${NC}"
                exit 1
            fi
            shift
            append_install_prewarm_models "$1"
            ;;
        --prewarm=*) append_install_prewarm_models "${1#*=}" ;;
        --prewarm-plan)
            PRINT_PREWARM_PLAN=true
            if [ "$#" -gt 1 ] && [[ "$2" != -* ]]; then
                shift
                append_install_prewarm_models "$1"
            fi
            ;;
        --prewarm-plan=*)
            PRINT_PREWARM_PLAN=true
            append_install_prewarm_models "${1#*=}"
            ;;
        --no-prewarm) INSTALL_PREWARM_MODELS=""; INSTALL_PREWARM_MODELS_CONFIGURED=true ;;
        --verify-fast) VERIFY_FAST_PATHS="${VERIFY_FAST_PATHS:-auto}" ;;
        --verify-fast=*) VERIFY_FAST_PATHS="${1#*=}" ;;
        --verify-fast-only)
            VERIFY_FAST_ONLY=true
            VERIFY_FAST_PATHS="${VERIFY_FAST_PATHS:-auto}"
            ;;
        --verify-fast-only=*)
            VERIFY_FAST_ONLY=true
            VERIFY_FAST_PATHS="${1#*=}"
            ;;
        --verify-fast-m4-smoke)
            apply_install_verify_m4_smoke_profile
            ;;
        --verify-fast-m4-smoke-only)
            VERIFY_FAST_ONLY=true
            apply_install_verify_m4_smoke_profile
            ;;
        --no-verify-fast) VERIFY_FAST_PATHS="" ;;
        -h|--help) 
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -n, --native     Build and install as native executable (requires GraalVM)"
            echo "  -t, --tests      Run tests during build"
            echo "  --skip-tests     Skip tests during build (default)"
            echo "  --stop-daemons-only"
            echo "                  Stop GGUF/LiteRT fast daemons and exit without building"
            echo "  --prewarm MODEL[,MODEL...]"
            echo "                  Prewarm installed GGUF/LiteRT fast daemon(s) after install"
            echo "                  Use --prewarm auto to select local runnable fast models"
            echo "                  Can also be set with GOLLEK_INSTALL_PREWARM_MODELS"
            echo "                  Auto selection defaults to one warm model per fast format"
            echo "                  and can be tuned with GOLLEK_INSTALL_PREWARM_AUTO_LIMIT"
            echo "                  and GOLLEK_INSTALL_PREWARM_AUTO_FORMATS"
            echo "                  Local JVM installs default to auto prewarm unless --no-prewarm"
            echo "                  or GOLLEK_INSTALL_AUTO_PREWARM=false is set"
            echo "  --prewarm-plan [MODEL[,MODEL...]]"
            echo "                  Print resolved prewarm model refs without installing"
            echo "  --no-prewarm     Ignore GOLLEK_INSTALL_PREWARM_MODELS for this install"
            echo "  --verify-fast[=auto|all|litert|gguf|gguf-compare|safetensor|gemma4-litert]"
            echo "                  Run installed LiteRT/GGUF/safetensor/Gemma4 LiteRT speed checks after install"
            echo "                  Can also be set with GOLLEK_INSTALL_VERIFY_FAST_PATHS"
            echo "                  Token counts can be tuned with"
            echo "                  GOLLEK_INSTALL_VERIFY_LITERT_MAX_TOKENS and"
            echo "                  GOLLEK_INSTALL_VERIFY_GGUF_MAX_TOKENS"
            echo "                  (defaults: LiteRT=10, GGUF=1)"
            echo "                  GGUF native warm timings can be bounded with"
            echo "                  GOLLEK_INSTALL_VERIFY_GGUF_WARM_{TOKENIZE,PREFILL,DECODE}_THRESHOLD_MS"
            echo "                  LiteRT warm profile timings can be bounded with"
            echo "                  GOLLEK_INSTALL_VERIFY_LITERT_WARM_{ENGINE_INIT,FIRST_CHUNK,TOTAL}_THRESHOLD_MS"
            echo "                  GGUF Java-vs-fallback compare can bound fallback timings with"
            echo "                  GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_FALLBACK_{PREFILL,DECODE}_THRESHOLD_MS"
            echo "                  Safetensor profiling can be bounded with"
            echo "                  GOLLEK_INSTALL_VERIFY_SAFETENSOR_{TOP_STAGE,PREFILL,DECODE,TPOT,ATTENTION,FFN,LOGITS}_THRESHOLD_MS"
            echo "                  Safetensor aggregate regression checks can use"
            echo "                  GOLLEK_INSTALL_VERIFY_SAFETENSOR_BASELINE_STAGES"
            echo "                  (use latest with GOLLEK_INSTALL_VERIFY_SAFETENSOR_BASELINE_ROOT/NAME)"
            echo "                  and GOLLEK_INSTALL_VERIFY_SAFETENSOR_{MAX_REGRESSION_PERCENT,MAX_REGRESSION_MS,MIN_BASELINE_MS}"
            echo "                  plus GOLLEK_INSTALL_VERIFY_SAFETENSOR_FAIL_PRIMARY_SHIFT=true"
            echo "                  and GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_PROFILE"
            echo "                  plus GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_METAL_PATHS"
            echo "                  plus GOLLEK_INSTALL_VERIFY_SAFETENSOR_REJECT_FALLBACK_PATHS"
            echo "                  (defaults true on macOS, false elsewhere)"
            echo "                  Safetensor aggregate presets can be selected with"
            echo "                  GOLLEK_INSTALL_VERIFY_SAFETENSOR_PRESET=m4-smoke"
            echo "                  Safetensor direct verification requires Metal on macOS;"
            echo "                  set GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_METAL=false"
            echo "                  or GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_METAL_PATHS=false"
            echo "                  only for diagnostics. Aggregate Metal policy can be"
            echo "                  tuned with GOLLEK_INSTALL_VERIFY_FAST_REQUIRE_METAL"
            echo "                  For fast repeated checks, set"
            echo "                  GOLLEK_INSTALL_VERIFY_LITERT_WARM_ONLY=true or"
            echo "                  GOLLEK_INSTALL_VERIFY_GGUF_WARM_ONLY=true"
            echo "                  Leaves verified fast daemons warm by default; set"
            echo "                  GOLLEK_INSTALL_VERIFY_FAST_KEEP_DAEMON=false to stop them"
            echo "                  Set GOLLEK_INSTALL_VERIFY_FAST_AGGREGATE=true to run"
            echo "                  scripts/verify-fast-speed-gates.sh and collect manifest/results"
            echo "                  TSV artifacts; optional directory:"
            echo "                  GOLLEK_INSTALL_VERIFY_FAST_SUMMARY_DIR"
            echo "                  Set GOLLEK_INSTALL_VERIFY_FAST_SLOWEST_LIMIT to tune"
            echo "                  the aggregate slowest-case terminal/TSV report"
            echo "                  Set GOLLEK_INSTALL_VERIFY_FAST_CONTINUE_ON_FAILURE=true"
            echo "                  to collect all selected aggregate gate results before failing"
            echo "                  GGUF verification requires first-repeat prompt-cache hits by"
            echo "                  default; set GOLLEK_INSTALL_VERIFY_GGUF_PROMPT_CACHE=false"
            echo "                  only for diagnostic prompts/models that cannot cache"
            echo "  --verify-fast-m4-smoke"
            echo "                  After install, run strict aggregate safetensor m4-smoke"
            echo "                  verification with Metal, Metal path proof, and fallback"
            echo "                  path rejection enabled"
            echo "  --verify-fast-m4-smoke-only"
            echo "                  Run the strict installed safetensor m4-smoke verification"
            echo "                  and exit without building"
            echo "  --verify-fast-only[=auto|all|litert|gguf|gguf-compare|safetensor|gemma4-litert]"
            echo "                  Verify the already-installed CLI and exit without building"
            echo "  --no-verify-fast Ignore GOLLEK_INSTALL_VERIFY_FAST_PATHS for this install"
            echo "  GOLLEK_ALLOW_METAL_BUILD_FAILURE=true"
            echo "                  Allow macOS install to continue if libgollek_metal.dylib fails"
            echo "  -h, --help       Show this help message"
            exit 0 
            ;;
    esac
    shift
done

install_auto_prewarm_enabled() {
    case "$(printf '%s' "${GOLLEK_INSTALL_AUTO_PREWARM:-true}" | tr '[:upper:]' '[:lower:]')" in
        0|false|no|off) return 1 ;;
        *) return 0 ;;
    esac
}

if [ "$INSTALL_PREWARM_MODELS_CONFIGURED" != "true" ] \
    && [ -z "$VERIFY_FAST_PATHS" ] \
    && install_auto_prewarm_enabled; then
    INSTALL_PREWARM_MODELS="auto"
fi

if [ "$PRINT_PREWARM_PLAN" != "true" ]; then
    echo -e "${BLUE}--------------------------------------------------${NC}"
    echo -e "${GREEN} Gollek Local Installer (Dev Mode) ${NC}"
    if [ "$NATIVE_MODE" = true ]; then
        echo -e "${YELLOW} Mode: Native (GraalVM) ${NC}"
    else
        echo -e "${YELLOW} Mode: JVM (Standard JAR) ${NC}"
    fi
    echo -e "${YELLOW} Build: Gradle-only ${NC}"
    echo -e "${BLUE}--------------------------------------------------${NC}"
fi

# Tokenization is pure Java on the Gradle install path. Historical native
# SentencePiece install/copy logic is intentionally disabled so local installs
# do not imply an external tokenizer runtime dependency.
install_tokenizer_native_runtime() {
    mkdir -p "${HOME}/.gollek/native" "${LIB_DIR}"
    echo -e "${GREEN}✓ Tokenizer runtime: pure Java (no SentencePiece native dependency)${NC}"
}

allow_metal_build_failure() {
    case "$(printf '%s' "${GOLLEK_ALLOW_METAL_BUILD_FAILURE:-false}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

stop_daemon_via_cli() {
    local command="$1"
    if [ ! -x "$GOLLEK_CLI_BIN" ]; then
        return 0
    fi
    GOLLEK_GGUF_FAST_RUN=true \
        GOLLEK_GGUF_FAST_DAEMON=true \
        GOLLEK_LITERT_FAST_RUN=true \
        GOLLEK_LITERT_FAST_DAEMON=true \
        "$GOLLEK_CLI_BIN" "$command" >/dev/null 2>&1 || true
}

is_daemon_command() {
    local command="$1"
    local class_name="$2"
    case " $command " in
        *" $class_name __daemon "*) return 0 ;;
        *" $class_name "*" __daemon "*) return 0 ;;
        *) return 1 ;;
    esac
}

terminate_daemon_pid() {
    local pid="$1"
    local label="$2"
    local attempt

    case "$pid" in
        ''|*[!0-9]*) return 0 ;;
    esac
    if [ "$pid" = "$$" ]; then
        return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
        return 0
    fi

    echo -e "${YELLOW}⚠ ${label} fast daemon did not stop via protocol; terminating pid ${pid}.${NC}"
    kill "$pid" >/dev/null 2>&1 || true
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    kill -9 "$pid" >/dev/null 2>&1 || true
}

stop_daemon_from_port_file() {
    local port_file="$1"
    local class_name="$2"
    local label="$3"
    local pid command

    if [ ! -f "$port_file" ]; then
        return 0
    fi
    pid="$(sed -n '2p' "$port_file" 2>/dev/null | tr -d '[:space:]' || true)"
    case "$pid" in
        ''|*[!0-9]*) return 0 ;;
    esac
    if ! kill -0 "$pid" >/dev/null 2>&1; then
        return 0
    fi
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if ! is_daemon_command "$command" "$class_name"; then
        return 0
    fi

    terminate_daemon_pid "$pid" "$label"
}

stop_daemons_by_process_scan() {
    local class_name="$1"
    local label="$2"
    local line pid command

    if ! command -v ps >/dev/null 2>&1; then
        return 0
    fi
    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"
        pid="${line%%[[:space:]]*}"
        command="${line#"$pid"}"
        command="${command#"${command%%[![:space:]]*}"}"
        case "$pid" in
            ''|*[!0-9]*) continue ;;
        esac
        if is_daemon_command "$command" "$class_name"; then
            terminate_daemon_pid "$pid" "$label"
        fi
    done < <(ps -axo pid=,command= 2>/dev/null || true)
}

bootout_fast_daemon_launchctl_jobs() {
    if [ "$(uname -s)" != "Darwin" ] || ! command -v launchctl >/dev/null 2>&1; then
        return 0
    fi
    local uid
    uid="$(id -u 2>/dev/null || true)"
    if [ -n "$uid" ]; then
        launchctl kill TERM "gui/${uid}/tech.kayys.gollek.gguf-fast-daemon" >/dev/null 2>&1 || true
        launchctl kill TERM "gui/${uid}/tech.kayys.gollek.litert-fast-daemon" >/dev/null 2>&1 || true
        launchctl bootout "gui/${uid}/tech.kayys.gollek.gguf-fast-daemon" >/dev/null 2>&1 || true
        launchctl bootout "gui/${uid}/tech.kayys.gollek.litert-fast-daemon" >/dev/null 2>&1 || true
    fi
    launchctl remove tech.kayys.gollek.gguf-fast-daemon >/dev/null 2>&1 || true
    launchctl remove tech.kayys.gollek.litert-fast-daemon >/dev/null 2>&1 || true
}

stop_existing_fast_daemons() {
    mkdir -p "${HOME}/.gollek/run" "${HOME}/.gollek/logs"
    echo -e "${BLUE}>>> Stopping existing fast inference daemons...${NC}"
    bootout_fast_daemon_launchctl_jobs
    stop_daemon_via_cli "__gguf-daemon-stop"
    stop_daemon_via_cli "__daemon-stop"
    stop_daemon_from_port_file \
        "${HOME}/.gollek/run/gguf-fast-daemon.port" \
        "tech.kayys.gollek.cli.commands.GgufFastRun" \
        "GGUF"
    stop_daemon_from_port_file \
        "${HOME}/.gollek/run/litert-fast-daemon.port" \
        "tech.kayys.gollek.cli.commands.LiteRtLmFastRun" \
        "LiteRT"
    stop_daemons_by_process_scan \
        "tech.kayys.gollek.cli.commands.GgufFastRun" \
        "GGUF"
    stop_daemons_by_process_scan \
        "tech.kayys.gollek.cli.commands.LiteRtLmFastRun" \
        "LiteRT"
    bootout_fast_daemon_launchctl_jobs
    rm -f "${HOME}/.gollek/run/gguf-fast-daemon.port" \
          "${HOME}/.gollek/run/litert-fast-daemon.port"
}

install_prewarm_backend() {
    if [ -n "${GOLLEK_INSTALL_PREWARM_BACKEND:-}" ]; then
        printf '%s\n' "$GOLLEK_INSTALL_PREWARM_BACKEND"
        return 0
    fi
    if [ "$(uname -s)" = "Darwin" ]; then
        printf '%s\n' "metal"
    fi
}

install_prewarm_auto_refs() {
    local mode="${1:-auto}"
    local index_path="${HOME}/.gollek/models/index.json"
    local formats="${GOLLEK_INSTALL_PREWARM_AUTO_FORMATS:-litert,gguf}"
    local limit="${GOLLEK_INSTALL_PREWARM_AUTO_LIMIT:-2}"

    case "$mode" in
        auto:gguf|gguf:auto) formats="gguf" ;;
        auto:litert|litert:auto|auto:litertlm|litertlm:auto) formats="litert" ;;
        auto:all|all:auto) formats="litert,gguf" ;;
    esac

    [ -f "$index_path" ] || return 0
    awk -v formats="$formats" -v limit="$limit" '
        function trim(value) {
            sub(/^[[:space:]]+/, "", value)
            sub(/[[:space:]]+$/, "", value)
            return value
        }
        function json_value(line, value) {
            value = line
            sub(/^[^:]*:[[:space:]]*"/, "", value)
            sub(/".*$/, "", value)
            return value
        }
        function json_bool(line, value) {
            value = line
            sub(/^[^:]*:[[:space:]]*/, "", value)
            sub(/[,[:space:]]*$/, "", value)
            return tolower(value)
        }
        function reset_entry() {
            id = ""; shortId = ""; name = ""; format = ""; runnable = ""; path = ""; source = ""
        }
        function shell_quote(value, out, i, ch) {
            out = "\047"
            for (i = 1; i <= length(value); i++) {
                ch = substr(value, i, 1)
                if (ch == "\047") {
                    out = out "\047\\\047\047"
                } else {
                    out = out ch
                }
            }
            return out "\047"
        }
        function path_exists(value) {
            return value == "" || system("test -e " shell_quote(value)) == 0
        }
        function format_key(value, lower) {
            lower = tolower(trim(value))
            if (lower == "gguf") {
                return "gguf"
            }
            if (lower == "litert" || lower == "litertlm" || lower == "lite-rt" || lower == "task" || lower == "tflite") {
                return "litert"
            }
            return ""
        }
        function candidate_ref() {
            if (shortId != "") {
                return shortId
            }
            if (id != "") {
                return id
            }
            if (name != "") {
                return name
            }
            return path
        }
        function remember_entry(   key, ref, preferred) {
            key = format_key(format)
            if (key == "" || !(key in wanted) || runnable != "true") {
                return
            }
            if (!path_exists(path)) {
                return
            }
            if (key == "litert" && tolower(path) ~ /\.(tflite|task)$/) {
                return
            }
            ref = candidate_ref()
            if (ref == "") {
                return
            }
            if (!(key in first)) {
                first[key] = ref
            }
            preferred = (tolower(source) == "local" && path ~ /\/models\/blobs\//)
            if (preferred && !(key in best)) {
                best[key] = ref
            }
        }
        BEGIN {
            split(formats, formatOrder, /[,;]/)
            for (i = 1; i <= length(formatOrder); i++) {
                key = format_key(formatOrder[i])
                if (key != "" && !(key in wanted)) {
                    wanted[key] = 1
                    order[++orderCount] = key
                }
            }
            if (limit !~ /^[0-9]+$/ || limit < 1) {
                limit = 2
            }
            reset_entry()
        }
        {
            lower = tolower($0)
            if (lower ~ /"shortid"[[:space:]]*:/) {
                shortId = json_value($0)
            } else if (lower ~ /"id"[[:space:]]*:/) {
                id = json_value($0)
            } else if (lower ~ /"name"[[:space:]]*:/) {
                name = json_value($0)
            } else if (lower ~ /"format"[[:space:]]*:/) {
                format = json_value($0)
            } else if (lower ~ /"runnable"[[:space:]]*:/) {
                runnable = json_bool($0)
            } else if (lower ~ /"path"[[:space:]]*:/) {
                path = json_value($0)
            } else if (lower ~ /"source"[[:space:]]*:/) {
                source = json_value($0)
            }

            if ($0 ~ /^[[:space:]]*}[,]?[[:space:]]*[{]?[[:space:]]*$/) {
                remember_entry()
                reset_entry()
            }
        }
        END {
            remember_entry()
            for (i = 1; i <= orderCount && printed < limit; i++) {
                key = order[i]
                ref = (key in best) ? best[key] : first[key]
                if (ref != "" && !(ref in emitted)) {
                    print ref
                    emitted[ref] = 1
                    printed++
                }
            }
        }
    ' "$index_path"
}

expand_install_prewarm_models() {
    printf '%s\n' "$INSTALL_PREWARM_MODELS" \
        | tr ',;' '\n' \
        | while IFS= read -r model_ref; do
            model_ref="${model_ref#"${model_ref%%[![:space:]]*}"}"
            model_ref="${model_ref%"${model_ref##*[![:space:]]}"}"
            [ -n "$model_ref" ] || continue
            case "$(printf '%s' "$model_ref" | tr '[:upper:]' '[:lower:]')" in
                auto|auto:*|*:auto) install_prewarm_auto_refs "$model_ref" ;;
                *) printf '%s\n' "$model_ref" ;;
            esac
        done \
        | awk 'NF && !seen[$0]++'
}

install_prewarm_java_opts() {
    local opts="${GOLLEK_JAVA_OPTS:-}"
    local idle_seconds="${GOLLEK_INSTALL_PREWARM_IDLE_SECONDS:-3600}"
    [ -n "$idle_seconds" ] || {
        printf '%s\n' "$opts"
        return 0
    }

    case " $opts " in
        *" -Dgollek.gguf.fast_run.daemon_idle_seconds="*) ;;
        *) opts="${opts:+$opts }-Dgollek.gguf.fast_run.daemon_idle_seconds=${idle_seconds}" ;;
    esac
    case " $opts " in
        *" -Dgollek.litert.fast_run.daemon_idle_seconds="*) ;;
        *) opts="${opts:+$opts }-Dgollek.litert.fast_run.daemon_idle_seconds=${idle_seconds}" ;;
    esac
    printf '%s\n' "$opts"
}

prewarm_installed_models() {
    if [ -z "$INSTALL_PREWARM_MODELS" ]; then
        return 0
    fi
    if [ "$NATIVE_MODE" = true ]; then
        echo -e "${YELLOW}⚠ Skipping daemon prewarm for native install mode.${NC}"
        return 0
    fi
    if [ ! -x "$GOLLEK_CLI_BIN" ]; then
        echo -e "${YELLOW}⚠ Skipping daemon prewarm because ${GOLLEK_CLI_BIN} is not executable.${NC}"
        return 0
    fi

    local prompt="${GOLLEK_INSTALL_PREWARM_PROMPT:-where is jakarta}"
    local max_tokens="${GOLLEK_INSTALL_PREWARM_MAX_TOKENS:-10}"
    local backend
    backend="$(install_prewarm_backend)"

    echo -e "${BLUE}>>> Prewarming installed fast inference daemon(s)...${NC}"
    expand_install_prewarm_models \
        | while IFS= read -r model_ref; do
            model_ref="${model_ref#"${model_ref%%[![:space:]]*}"}"
            model_ref="${model_ref%"${model_ref##*[![:space:]]}"}"
            [ -n "$model_ref" ] || continue

            echo -e "${BLUE}>>> Prewarming model ${model_ref}...${NC}"
            local args=("prewarm" "--model" "$model_ref" "--prompt" "$prompt" "--max-tokens" "$max_tokens")
            if [ -n "$backend" ]; then
                args+=("--backend" "$backend")
            fi
            if GOLLEK_JAVA_OPTS="$(install_prewarm_java_opts)" "$GOLLEK_CLI_BIN" "${args[@]}"; then
                echo -e "${GREEN}✓ Prewarmed ${model_ref}${NC}"
            else
                echo -e "${YELLOW}⚠ Prewarm failed for ${model_ref}; install remains usable and the first run will cold-start.${NC}"
            fi
        done
}

install_verify_model_present() {
    local ref="$1"
    local index_path="${HOME}/.gollek/models/index.json"
    [ -f "$index_path" ] || return 1
    awk -v ref="$ref" '
        function json_value(line, value) {
            value = line
            sub(/^[^:]*:[[:space:]]*"/, "", value)
            sub(/".*$/, "", value)
            return value
        }
        {
            lower = tolower($0)
            if (lower ~ /"(id|shortid|name|path)"[[:space:]]*:/) {
                value = json_value($0)
                if (value == ref || tolower(value) == tolower(ref)) {
                    found = 1
                }
            }
        }
        END { exit found ? 0 : 1 }
    ' "$index_path"
}

install_verify_should_run() {
    local target="$1"
    local mode
    mode="$(printf '%s' "${VERIFY_FAST_PATHS:-}" | tr '[:upper:]' '[:lower:]')"
    [ -n "$mode" ] || return 1

    case "$mode" in
        1|true|yes|on)
            case "$target" in
                litert|gguf) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        auto)
            case "$target" in
                litert) install_verify_model_present "${GOLLEK_INSTALL_VERIFY_LITERT_MODEL:-7c51c9}" ;;
                gguf) install_verify_model_present "${GOLLEK_INSTALL_VERIFY_GGUF_MODEL:-b71c9d}" ;;
                safetensor) install_verify_model_present "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_MODEL:-6f469a}" ;;
                *) return 1 ;;
            esac
            ;;
        both)
            case "$target" in
                litert|gguf) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        all)
            case "$target" in
                litert|gguf|gguf-compare|safetensor) return 0 ;;
                *) return 1 ;;
            esac
            ;;
    esac

    printf '%s\n' "$mode" \
        | tr ',; ' '\n' \
        | awk -v target="$target" 'tolower($0) == target { found = 1 } END { exit found ? 0 : 1 }'
}

install_verify_keep_daemon() {
    case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_FAST_KEEP_DAEMON:-true}" | tr '[:upper:]' '[:lower:]')" in
        0|false|no|off) return 1 ;;
        *) return 0 ;;
    esac
}

install_verify_aggregate_enabled() {
    case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_FAST_AGGREGATE:-false}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

install_verify_safetensor_require_metal_paths() {
    case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_METAL_PATHS:-auto}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        0|false|no|off) return 1 ;;
        *) [ "$(uname -s)" = "Darwin" ] ;;
    esac
}

install_verify_safetensor_require_metal_paths_value() {
    if install_verify_safetensor_require_metal_paths; then
        printf 'true'
    else
        printf 'false'
    fi
}

verify_installed_fast_paths() {
    if [ -z "$VERIFY_FAST_PATHS" ]; then
        return 0
    fi
    if [ "$NATIVE_MODE" = true ]; then
        echo -e "${YELLOW}⚠ Skipping fast-path verification for native install mode.${NC}"
        return 0
    fi
    if [ ! -x "$GOLLEK_CLI_BIN" ]; then
        echo -e "${YELLOW}⚠ Skipping fast-path verification because ${GOLLEK_CLI_BIN} is not executable.${NC}"
        return 0
    fi

    local ran_any=false
    local litert_bench="${GOLLEK_INSTALL_VERIFY_LITERT_BENCH:-${PROJECT_ROOT}/scripts/bench-litert-fast-run.sh}"
    local gguf_bench="${GOLLEK_INSTALL_VERIFY_GGUF_BENCH:-${PROJECT_ROOT}/scripts/bench-gguf-fast-run.sh}"
    local gguf_compare_bench="${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH:-${PROJECT_ROOT}/scripts/bench-gguf-engine-compare.sh}"
    local safetensor_bench="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH:-${PROJECT_ROOT}/scripts/bench-safetensor-inference.sh}"
    local safetensor_preset_script="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_PRESET_SCRIPT:-${PROJECT_ROOT}/scripts/safetensor-performance-presets.sh}"
    local gemma4_litert_smoke="${GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_SMOKE:-${PROJECT_ROOT}/scripts/smoke-gemma4-litert-warm.sh}"
    if install_verify_aggregate_enabled; then
        local aggregate_bench="${GOLLEK_INSTALL_VERIFY_FAST_AGGREGATE_BENCH:-${PROJECT_ROOT}/scripts/verify-fast-speed-gates.sh}"
        local aggregate_summary_args=()
        local aggregate_keep_daemon_args=()
        local aggregate_slowest_args=()
        local aggregate_metal_args=()
        if [ -n "${GOLLEK_INSTALL_VERIFY_FAST_SUMMARY_DIR:-}" ]; then
            aggregate_summary_args+=(--summary-dir "${GOLLEK_INSTALL_VERIFY_FAST_SUMMARY_DIR}")
        fi
        if [ -n "${GOLLEK_INSTALL_VERIFY_FAST_SLOWEST_LIMIT:-}" ]; then
            aggregate_slowest_args+=(--slowest-limit "${GOLLEK_INSTALL_VERIFY_FAST_SLOWEST_LIMIT}")
        fi
        if install_verify_keep_daemon; then
            aggregate_keep_daemon_args+=(--keep-daemon)
        fi
        case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_FAST_REQUIRE_METAL:-auto}" | tr '[:upper:]' '[:lower:]')" in
            1|true|yes|on) aggregate_metal_args+=(--require-metal) ;;
            0|false|no|off) aggregate_metal_args+=(--no-require-metal) ;;
        esac
        echo -e "${BLUE}>>> Verifying installed fast inference speed with aggregate gate...${NC}"
        GOLLEK_VERIFY_FAST_LITERT_BENCH="$litert_bench" \
        GOLLEK_VERIFY_FAST_GGUF_BENCH="$gguf_bench" \
        GOLLEK_VERIFY_FAST_GGUF_COMPARE_BENCH="$gguf_compare_bench" \
        GOLLEK_VERIFY_FAST_SAFETENSOR_BENCH="$safetensor_bench" \
        GOLLEK_VERIFY_PROMPT="${GOLLEK_INSTALL_VERIFY_PROMPT:-where is jakarta}" \
        GOLLEK_VERIFY_LITERT_EXPECTED="${GOLLEK_INSTALL_VERIFY_LITERT_EXPECTED:-Jakarta|Indonesia}" \
        GOLLEK_VERIFY_GGUF_EXPECTED="${GOLLEK_INSTALL_VERIFY_GGUF_EXPECTED:-Indonesia|Jakarta}" \
        GOLLEK_VERIFY_LITERT_MAX_TOKENS="${GOLLEK_INSTALL_VERIFY_LITERT_MAX_TOKENS:-10}" \
        GOLLEK_VERIFY_LITERT_WARM_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_LITERT_WARM_THRESHOLD_MS:-1500}" \
        GOLLEK_VERIFY_LITERT_COLD_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_LITERT_COLD_THRESHOLD_MS:-60000}" \
        GOLLEK_VERIFY_LITERT_WARMUP_RUNS="${GOLLEK_INSTALL_VERIFY_LITERT_WARMUP_RUNS:-1}" \
        GOLLEK_VERIFY_LITERT_WARM_ONLY="${GOLLEK_INSTALL_VERIFY_LITERT_WARM_ONLY:-false}" \
        GOLLEK_VERIFY_LITERT_WARM_ENGINE_INIT_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_LITERT_WARM_ENGINE_INIT_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_LITERT_WARM_TOTAL_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_LITERT_WARM_TOTAL_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_GGUF_MAX_TOKENS="${GOLLEK_INSTALL_VERIFY_GGUF_MAX_TOKENS:-1}" \
        GOLLEK_VERIFY_GGUF_WARM_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_GGUF_WARM_THRESHOLD_MS:-1500}" \
        GOLLEK_VERIFY_GGUF_COLD_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_GGUF_COLD_THRESHOLD_MS:-60000}" \
        GOLLEK_VERIFY_GGUF_WARMUP_RUNS="${GOLLEK_INSTALL_VERIFY_GGUF_WARMUP_RUNS:-1}" \
        GOLLEK_VERIFY_GGUF_PROMPT_CACHE="${GOLLEK_INSTALL_VERIFY_GGUF_PROMPT_CACHE:-true}" \
        GOLLEK_VERIFY_GGUF_WARM_ONLY="${GOLLEK_INSTALL_VERIFY_GGUF_WARM_ONLY:-false}" \
        GOLLEK_VERIFY_GGUF_WARM_TOKENIZE_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_GGUF_WARM_TOKENIZE_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_GGUF_WARM_PREFILL_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_GGUF_WARM_PREFILL_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_GGUF_WARM_DECODE_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_GGUF_WARM_DECODE_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_GGUF_COMPARE_MAX_TOKENS="${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_MAX_TOKENS:-10}" \
        GOLLEK_VERIFY_GGUF_COMPARE_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_THRESHOLD_MS:-10000}" \
        GOLLEK_VERIFY_GGUF_COMPARE_JAVA_MATVEC_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_MATVEC_THRESHOLD_MS:-50}" \
        GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_PREFILL_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_FALLBACK_PREFILL_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_DECODE_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_FALLBACK_DECODE_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_GGUF_COMPARE_JAVA_READY_REGEX="${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_READY_REGEX:-row-dot-primitives-ready}" \
        GOLLEK_VERIFY_GGUF_COMPARE_JAVA_CONFIG_REGEX="${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_CONFIG_REGEX:-type=[^,]+, layers=[1-9][0-9]*, hidden=[1-9][0-9]*, heads=[1-9][0-9]*/[1-9][0-9]*, headDim=[1-9][0-9]*, context=[1-9][0-9]*, vocab=[1-9][0-9]*}" \
        GOLLEK_VERIFY_GGUF_COMPARE_JAVA_PROBE_REGEX="${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_PROBE_REGEX:-}" \
        GOLLEK_VERIFY_GGUF_COMPARE_JAVA_REFUSAL="${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_REFUSAL:-true}" \
        GOLLEK_VERIFY_SAFETENSOR_MAX_TOKENS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_MAX_TOKENS:-2}" \
        GOLLEK_VERIFY_SAFETENSOR_PRESET="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_PRESET:-}" \
        GOLLEK_VERIFY_SAFETENSOR_PRESET_SCRIPT="$safetensor_preset_script" \
        GOLLEK_VERIFY_SAFETENSOR_REQUIRE_PROFILE="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_PROFILE:-true}" \
        GOLLEK_VERIFY_SAFETENSOR_REQUIRE_METAL_PATHS="$(install_verify_safetensor_require_metal_paths_value)" \
        GOLLEK_VERIFY_SAFETENSOR_REJECT_FALLBACK_PATHS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_REJECT_FALLBACK_PATHS:-false}" \
        GOLLEK_VERIFY_SAFETENSOR_TOP_STAGE_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_TOP_STAGE_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_SAFETENSOR_PREFILL_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_PREFILL_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_SAFETENSOR_DECODE_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_DECODE_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_SAFETENSOR_TPOT_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_TPOT_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_SAFETENSOR_ATTENTION_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_ATTENTION_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_SAFETENSOR_FFN_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_FFN_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_SAFETENSOR_LOGITS_THRESHOLD_MS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_LOGITS_THRESHOLD_MS:-}" \
        GOLLEK_VERIFY_SAFETENSOR_BASELINE_STAGES="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_BASELINE_STAGES:-}" \
        GOLLEK_VERIFY_SAFETENSOR_BASELINE_ROOT="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_BASELINE_ROOT:-}" \
        GOLLEK_VERIFY_SAFETENSOR_BASELINE_NAME="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_BASELINE_NAME:-}" \
        GOLLEK_VERIFY_SAFETENSOR_MAX_REGRESSION_PERCENT="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_MAX_REGRESSION_PERCENT:-}" \
        GOLLEK_VERIFY_SAFETENSOR_MAX_REGRESSION_MS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_MAX_REGRESSION_MS:-}" \
        GOLLEK_VERIFY_SAFETENSOR_MIN_BASELINE_MS="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_MIN_BASELINE_MS:-1}" \
        GOLLEK_VERIFY_SAFETENSOR_FAIL_PRIMARY_SHIFT="${GOLLEK_INSTALL_VERIFY_SAFETENSOR_FAIL_PRIMARY_SHIFT:-false}" \
        GOLLEK_VERIFY_CONTINUE_ON_FAILURE="${GOLLEK_INSTALL_VERIFY_FAST_CONTINUE_ON_FAILURE:-false}" \
            bash "$aggregate_bench" \
                --only "$VERIFY_FAST_PATHS" \
                --gollek-bin "$GOLLEK_CLI_BIN" \
                --litert-model "${GOLLEK_INSTALL_VERIFY_LITERT_MODEL:-7c51c9}" \
                --gguf-model "${GOLLEK_INSTALL_VERIFY_GGUF_MODEL:-b71c9d}" \
                --safetensor-model "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_MODEL:-6f469a}" \
                "${aggregate_metal_args[@]}" \
                "${aggregate_summary_args[@]}" \
                "${aggregate_slowest_args[@]}" \
                "${aggregate_keep_daemon_args[@]}"
        return 0
    fi
    local keep_daemon_args=()
    local litert_profile_threshold_args=()
    local litert_warm_only_args=()
    local gguf_prompt_cache_args=(--require-prompt-cache)
    local gguf_native_threshold_args=()
    local gguf_warm_only_args=()
    local gguf_compare_fallback_threshold_args=()
    local gguf_compare_java_refusal_args=(--verify-java-refusal)
    local gguf_compare_summary_args=()
    local safetensor_profile_args=(--require-profile)
    local safetensor_metal_args=()
    local safetensor_metal_path_args=()
    local safetensor_fallback_path_args=()
    local safetensor_threshold_args=()
    local safetensor_summary_args=()
    local safetensor_preset_args=()
    if install_verify_keep_daemon; then
        keep_daemon_args=(--keep-daemon)
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_LITERT_WARM_ENGINE_INIT_THRESHOLD_MS:-}" ]; then
        litert_profile_threshold_args+=(--warm-engine-init-threshold-ms "${GOLLEK_INSTALL_VERIFY_LITERT_WARM_ENGINE_INIT_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS:-}" ]; then
        litert_profile_threshold_args+=(--warm-first-chunk-threshold-ms "${GOLLEK_INSTALL_VERIFY_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_LITERT_WARM_TOTAL_THRESHOLD_MS:-}" ]; then
        litert_profile_threshold_args+=(--warm-profile-total-threshold-ms "${GOLLEK_INSTALL_VERIFY_LITERT_WARM_TOTAL_THRESHOLD_MS}")
    fi
    case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_LITERT_WARM_ONLY:-false}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) litert_warm_only_args=(--warm-only) ;;
    esac
    case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_GGUF_PROMPT_CACHE:-true}" | tr '[:upper:]' '[:lower:]')" in
        0|false|no|off) gguf_prompt_cache_args=() ;;
    esac
    if [ -n "${GOLLEK_INSTALL_VERIFY_GGUF_WARM_TOKENIZE_THRESHOLD_MS:-}" ]; then
        gguf_native_threshold_args+=(--warm-tokenize-threshold-ms "${GOLLEK_INSTALL_VERIFY_GGUF_WARM_TOKENIZE_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_GGUF_WARM_PREFILL_THRESHOLD_MS:-}" ]; then
        gguf_native_threshold_args+=(--warm-prefill-threshold-ms "${GOLLEK_INSTALL_VERIFY_GGUF_WARM_PREFILL_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_GGUF_WARM_DECODE_THRESHOLD_MS:-}" ]; then
        gguf_native_threshold_args+=(--warm-decode-threshold-ms "${GOLLEK_INSTALL_VERIFY_GGUF_WARM_DECODE_THRESHOLD_MS}")
    fi
    case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_GGUF_WARM_ONLY:-false}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) gguf_warm_only_args=(--warm-only) ;;
    esac
    if [ -n "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_FALLBACK_PREFILL_THRESHOLD_MS:-}" ]; then
        gguf_compare_fallback_threshold_args+=(--fallback-prefill-threshold-ms "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_FALLBACK_PREFILL_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_FALLBACK_DECODE_THRESHOLD_MS:-}" ]; then
        gguf_compare_fallback_threshold_args+=(--fallback-decode-threshold-ms "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_FALLBACK_DECODE_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_SUMMARY_FILE:-}" ]; then
        gguf_compare_summary_args+=(--summary-file "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_SUMMARY_FILE}")
    fi
    case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_REFUSAL:-true}" | tr '[:upper:]' '[:lower:]')" in
        0|false|no|off) gguf_compare_java_refusal_args=(--no-verify-java-refusal) ;;
    esac
    case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_PROFILE:-true}" | tr '[:upper:]' '[:lower:]')" in
        0|false|no|off) safetensor_profile_args=() ;;
    esac
    case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_METAL:-auto}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) safetensor_metal_args+=(--require-metal) ;;
        0|false|no|off) ;;
        *) [ "$(uname -s)" = "Darwin" ] && safetensor_metal_args+=(--require-metal) ;;
    esac
    if install_verify_safetensor_require_metal_paths; then
        safetensor_metal_path_args+=(--require-metal-paths)
    fi
    case "$(printf '%s' "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_REJECT_FALLBACK_PATHS:-false}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) safetensor_fallback_path_args+=(--reject-fallback-paths) ;;
    esac
    if [ -n "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_TOP_STAGE_THRESHOLD_MS:-}" ]; then
        safetensor_threshold_args+=(--max-top-stage-ms "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_TOP_STAGE_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_PREFILL_THRESHOLD_MS:-}" ]; then
        safetensor_threshold_args+=(--max-prefill-ms "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_PREFILL_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_DECODE_THRESHOLD_MS:-}" ]; then
        safetensor_threshold_args+=(--max-decode-ms "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_DECODE_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_TPOT_THRESHOLD_MS:-}" ]; then
        safetensor_threshold_args+=(--max-tpot-ms "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_TPOT_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_ATTENTION_THRESHOLD_MS:-}" ]; then
        safetensor_threshold_args+=(--max-attention-ms "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_ATTENTION_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_FFN_THRESHOLD_MS:-}" ]; then
        safetensor_threshold_args+=(--max-ffn-ms "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_FFN_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_LOGITS_THRESHOLD_MS:-}" ]; then
        safetensor_threshold_args+=(--max-logits-ms "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_LOGITS_THRESHOLD_MS}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_SUMMARY_FILE:-}" ]; then
        safetensor_summary_args+=(--summary-file "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_SUMMARY_FILE}")
    fi
    if [ -n "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_PRESET:-}" ]; then
        safetensor_preset_args+=(--preset "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_PRESET}")
    fi
    echo -e "${BLUE}>>> Verifying installed fast inference speed...${NC}"
    if install_verify_should_run "litert"; then
        ran_any=true
        bash "$litert_bench" \
            --gollek-bin "$GOLLEK_CLI_BIN" \
            --model "${GOLLEK_INSTALL_VERIFY_LITERT_MODEL:-7c51c9}" \
            --max-tokens "${GOLLEK_INSTALL_VERIFY_LITERT_MAX_TOKENS:-10}" \
            --warm-threshold-ms "${GOLLEK_INSTALL_VERIFY_LITERT_WARM_THRESHOLD_MS:-1500}" \
            --cold-threshold-ms "${GOLLEK_INSTALL_VERIFY_LITERT_COLD_THRESHOLD_MS:-60000}" \
            --warmup-runs "${GOLLEK_INSTALL_VERIFY_LITERT_WARMUP_RUNS:-1}" \
            "${litert_profile_threshold_args[@]}" \
            "${litert_warm_only_args[@]}" \
            "${keep_daemon_args[@]}"
    fi
    if install_verify_should_run "gemma4-litert"; then
        ran_any=true
        local gemma4_litert_keep_daemon_args=()
        if install_verify_keep_daemon; then
            gemma4_litert_keep_daemon_args=(--keep-daemon)
        fi
        bash "$gemma4_litert_smoke" \
            --gollek-bin "$GOLLEK_CLI_BIN" \
            --model "${GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_MODEL:-0576e9}" \
            --prompt "${GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_PROMPT:-${GOLLEK_INSTALL_VERIFY_PROMPT:-what is jakarta}}" \
            --expected "${GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_EXPECTED:-Jakarta|Indonesia}" \
            --max-tokens "${GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_MAX_TOKENS:-30}" \
            --warm-threshold-ms "${GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_WARM_THRESHOLD_MS:-1000}" \
            --warm-first-chunk-threshold-ms "${GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS:-250}" \
            --warm-profile-total-threshold-ms "${GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_WARM_TOTAL_THRESHOLD_MS:-1000}" \
            "${gemma4_litert_keep_daemon_args[@]}"
    fi
    if install_verify_should_run "gguf"; then
        ran_any=true
        bash "$gguf_bench" \
            --gollek-bin "$GOLLEK_CLI_BIN" \
            --model "${GOLLEK_INSTALL_VERIFY_GGUF_MODEL:-b71c9d}" \
            --max-tokens "${GOLLEK_INSTALL_VERIFY_GGUF_MAX_TOKENS:-1}" \
            --warm-threshold-ms "${GOLLEK_INSTALL_VERIFY_GGUF_WARM_THRESHOLD_MS:-1500}" \
            --cold-threshold-ms "${GOLLEK_INSTALL_VERIFY_GGUF_COLD_THRESHOLD_MS:-60000}" \
            --warmup-runs "${GOLLEK_INSTALL_VERIFY_GGUF_WARMUP_RUNS:-1}" \
            "${gguf_prompt_cache_args[@]}" \
            "${gguf_native_threshold_args[@]}" \
            "${gguf_warm_only_args[@]}" \
            "${keep_daemon_args[@]}"
    fi
    if install_verify_should_run "gguf-compare"; then
        ran_any=true
            bash "$gguf_compare_bench" \
                --gollek-bin "$GOLLEK_CLI_BIN" \
                --model "${GOLLEK_INSTALL_VERIFY_GGUF_MODEL:-b71c9d}" \
                --threshold-ms "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_THRESHOLD_MS:-10000}" \
                --java-ready-regex "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_READY_REGEX:-row-dot-primitives-ready}" \
                --java-config-regex "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_CONFIG_REGEX:-type=[^,]+, layers=[1-9][0-9]*, hidden=[1-9][0-9]*, heads=[1-9][0-9]*/[1-9][0-9]*, headDim=[1-9][0-9]*, context=[1-9][0-9]*, vocab=[1-9][0-9]*}" \
                --java-matvec-threshold-ms "${GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_MATVEC_THRESHOLD_MS:-50}" \
                "${gguf_compare_fallback_threshold_args[@]}" \
                "${gguf_compare_summary_args[@]}" \
                "${gguf_compare_java_refusal_args[@]}"
    fi
    if install_verify_should_run "safetensor"; then
        ran_any=true
        GOLLEK_BENCH_SAFETENSOR_PRESET_SCRIPT="$safetensor_preset_script" \
        bash "$safetensor_bench" \
            --gollek-bin "$GOLLEK_CLI_BIN" \
            --model "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_MODEL:-6f469a}" \
            --out-dir "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_OUT_DIR:-${HOME}/.gollek/benchmarks/safetensor/install-verify}" \
            --label "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_LABEL:-install-verify-safetensor}" \
            --det-prompt "${GOLLEK_INSTALL_VERIFY_PROMPT:-where is jakarta}" \
            --max-tokens "${GOLLEK_INSTALL_VERIFY_SAFETENSOR_MAX_TOKENS:-2}" \
            "${safetensor_preset_args[@]}" \
            --quick \
            --profile \
            "${safetensor_summary_args[@]}" \
            "${safetensor_profile_args[@]}" \
            "${safetensor_metal_args[@]}" \
            "${safetensor_metal_path_args[@]}" \
            "${safetensor_fallback_path_args[@]}" \
            "${safetensor_threshold_args[@]}"
    fi
    if [ "$ran_any" != "true" ]; then
        echo -e "${YELLOW}⚠ No installed LiteRT/GGUF/safetensor/Gemma4 LiteRT fast models matched --verify-fast=${VERIFY_FAST_PATHS}; skipping verification.${NC}"
    fi
}

if [ "$STOP_DAEMONS_ONLY" = true ]; then
    stop_existing_fast_daemons
    echo -e "${GREEN}✓ Fast inference daemons stopped${NC}"
    exit 0
fi

if [ "$VERIFY_FAST_ONLY" = true ]; then
    verify_installed_fast_paths
    exit 0
fi

if [ "$PRINT_PREWARM_PLAN" = "true" ]; then
    if [ -z "$INSTALL_PREWARM_MODELS" ]; then
        INSTALL_PREWARM_MODELS="auto"
    fi
    expand_install_prewarm_models
    exit 0
fi

build_gguf_fast_bridge() {
    if [ "$(uname -s)" != "Darwin" ]; then
        return 0
    fi
    if [ ! -f "$GGUF_FAST_BRIDGE_SRC" ]; then
        echo -e "${YELLOW}⚠ Skipping GGUF fast bridge build (source file not found).${NC}"
        return 0
    fi
    if ! command -v c++ >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠ Skipping GGUF fast bridge build (c++ compiler not found).${NC}"
        return 0
    fi

    local gemma4_llama_src="${HOME}/.gollek/source/vendor/llama.cpp-gemma4/llama.cpp"
    local gemma4_llama_lib_dir="${gemma4_llama_src}/build-gollek/bin"
    local default_llama_src="${HOME}/.gollek/source/vendor/llama.cpp/llama.cpp"
    local llama_src="${GOLLEK_LLAMA_SOURCE_DIR:-}"
    local llama_lib_dir="${GOLLEK_LLAMA_LIB_DIR:-}"
    if [ -z "$llama_src" ] && [ -f "${gemma4_llama_src}/include/llama.h" ]; then
        llama_src="$gemma4_llama_src"
    fi
    if [ -z "$llama_src" ]; then
        llama_src="$default_llama_src"
    fi
    if [ -z "$llama_lib_dir" ] && [ -f "${gemma4_llama_lib_dir}/libllama.dylib" ]; then
        llama_lib_dir="$gemma4_llama_lib_dir"
    fi
    if [ -z "$llama_lib_dir" ]; then
        llama_lib_dir="${LIB_DIR}/llama"
    fi
    local llama_header="${llama_src}/include/llama.h"
    local ggml_header="${llama_src}/ggml/include/ggml-backend.h"
    if [ ! -f "$llama_header" ] || [ ! -f "$ggml_header" ] || [ ! -f "${llama_lib_dir}/libllama.dylib" ]; then
        echo -e "${YELLOW}⚠ Skipping GGUF fast bridge build (llama.cpp headers/libs not found).${NC}"
        echo -e "${YELLOW}  Expected headers under ${llama_src} and libs under ${llama_lib_dir}.${NC}"
        return 0
    fi

    local runtime_llama_lib_dir="${LIB_DIR}/llama"
    if [ "$llama_lib_dir" != "$runtime_llama_lib_dir" ]; then
        echo -e "${BLUE}>>> Installing llama.cpp runtime libraries for GGUF fast path...${NC}"
        mkdir -p "$runtime_llama_lib_dir"
        local copied_any=false
        for lib in "$llama_lib_dir"/libllama*.dylib "$llama_lib_dir"/libggml*.dylib "$llama_lib_dir"/libmtmd*.dylib; do
            if [ -e "$lib" ]; then
                cp -PR "$lib" "$runtime_llama_lib_dir/"
                copied_any=true
            fi
        done
        if [ "$copied_any" = true ]; then
            echo -e "${GREEN}✓ Installed llama.cpp runtime libraries to ${runtime_llama_lib_dir}${NC}"
            llama_lib_dir="$runtime_llama_lib_dir"
        else
            echo -e "${YELLOW}⚠ No llama.cpp runtime libraries were copied from ${llama_lib_dir}.${NC}"
        fi
    fi

    echo -e "${BLUE}>>> Building and installing GGUF llama.cpp fast bridge...${NC}"
    mkdir -p "$LIB_DIR"
    if c++ -std=c++17 -O3 -DNDEBUG -fPIC -dynamiclib \
        -I"${llama_src}/include" \
        -I"${llama_src}/ggml/include" \
        "$GGUF_FAST_BRIDGE_SRC" \
        -L"${llama_lib_dir}" \
        -lllama -lggml -lggml-cpu -lggml-blas -lggml-metal -lggml-base \
        -Wl,-rpath,"${llama_lib_dir}" \
        -o "${LIB_DIR}/libgollek_gguf_fast.dylib"; then
        echo -e "${GREEN}✓ Installed libgollek_gguf_fast.dylib to ${LIB_DIR}/${NC}"
    else
        echo -e "${YELLOW}⚠ GGUF fast bridge build failed; full CLI fallback remains available.${NC}"
    fi
}

# 1. Build the project
stop_existing_fast_daemons
echo -e "${BLUE}>>> Building Gollek...${NC}"

METAL_MAKE_DIR="${PROJECT_ROOT}/${METAL_MODULE}/src/main/cpp/metal"
CLI_GRADLE_JAR="${PROJECT_ROOT}/${CLI_MODULE}/build/gollek.jar"
CLI_GRADLE_QUARKUS_GEN_DIR="${PROJECT_ROOT}/${CLI_MODULE}/build/quarkus-build/gen"
CLI_GRADLE_QUARKUS_GEN_JAR="${CLI_GRADLE_QUARKUS_GEN_DIR}/gollek.jar"
SAFETENSOR_ENGINE_CLASSES_GRADLE="${PROJECT_ROOT}/${SAFETENSOR_ENGINE_MODULE}/build/classes/java/main"
SAFETENSOR_ENGINE_CLASSES="$SAFETENSOR_ENGINE_CLASSES_GRADLE"
METAL_CLASSES_GRADLE="${PROJECT_ROOT}/${METAL_MODULE}/build/classes/java/main"

if [ ! -x "$GRADLEW" ]; then
    echo -e "${RED}❌ Gradle wrapper not found at ${GRADLEW}${NC}"
    exit 1
fi

GRADLE_ARGS=("--no-daemon" "--no-build-cache")
# Allow skipping strict model-family validations during local installs
GRADLE_ARGS+=("-Pgollek.skipModelFamilyBundlePresetValidation=true")
GRADLE_ARGS+=("-Pgollek.skipModelFamilyModuleCatalogValidation=true")
# Task-level switch recognized by ui/gollek-cli: skipModelFamilyModuleCatalogValidation (no gollek. prefix)
GRADLE_ARGS+=("-PskipModelFamilyModuleCatalogValidation=true")
if [ -n "${JAVA_HOME:-}" ]; then
    GRADLE_ARGS=("-Dorg.gradle.java.home=${JAVA_HOME}" "${GRADLE_ARGS[@]}")
fi
if [ -n "${GOLLEK_GRADLE_MAX_WORKERS:-1}" ]; then
    GRADLE_ARGS+=("--max-workers=${GOLLEK_GRADLE_MAX_WORKERS:-1}")
fi
if [[ "$BUILD_ARGS" != *"-t"* ]]; then
    GRADLE_ARGS+=("-x" "test")
fi

GRADLE_TASKS=(":ui:gollek-cli:quarkusBuild" ":ui:gollek-api:quarkusBuild")
if [[ "$BUILD_ARGS" == *"-c"* ]]; then
    GRADLE_TASKS=("clean" "${GRADLE_TASKS[@]}")
fi
if [ "$NATIVE_MODE" = true ]; then
    echo -e "${YELLOW}ℹ Building native image (this may take a while)...${NC}"
    GRADLE_ARGS+=("-Dquarkus.native.enabled=true" "-Dquarkus.native.container-build=false" "-Dgraalvm.metadataRepository.enabled=false")
fi

if [ "$SKIP_BUILD" = "true" ]; then
    echo -e "${YELLOW}⚠ Skipping Gradle build because GOLLEK_SKIP_BUILD=true${NC}"
else
    echo -e "${BLUE}>>> Running: ${GRADLEW} ${GRADLE_ARGS[*]} ${GRADLE_TASKS[*]}${NC}"
    if "$GRADLEW" "${GRADLE_ARGS[@]}" "${GRADLE_TASKS[@]}"; then
        echo -e "${GREEN}✓ Built Gollek CLI via Gradle${NC}"
    else
        echo -e "${RED}❌ Gradle build failed. Maven fallback has been removed from this installer.${NC}"
        exit 1
    fi
fi

if [ "$NATIVE_MODE" = false ] && [ "$(uname -s)" = "Darwin" ]; then
    if [ ! -f "${METAL_MAKE_DIR}/Makefile" ]; then
        if allow_metal_build_failure; then
            echo -e "${YELLOW}⚠ Skipping Metal native bridge build (Makefile not found at ${METAL_MAKE_DIR}).${NC}"
        else
            echo -e "${RED}❌ Metal native bridge Makefile not found at ${METAL_MAKE_DIR}.${NC}"
            echo -e "${RED}   Set GOLLEK_ALLOW_METAL_BUILD_FAILURE=true only if you intentionally want a non-Metal local install.${NC}"
            exit 1
        fi
    else
        echo -e "${BLUE}>>> Building and installing Metal native bridge...${NC}"
        if make -C "${METAL_MAKE_DIR}" install INSTALL_DIR="${LIB_DIR}"; then
            echo -e "${GREEN}✓ Installed libgollek_metal.dylib to ${LIB_DIR}/${NC}"
        elif allow_metal_build_failure; then
            echo -e "${YELLOW}⚠ Metal native bridge build/install failed; continuing because GOLLEK_ALLOW_METAL_BUILD_FAILURE=true.${NC}"
        else
            echo -e "${RED}❌ Metal native bridge build/install failed.${NC}"
            echo -e "${RED}   Fix the native Metal build or set GOLLEK_ALLOW_METAL_BUILD_FAILURE=true only for diagnostics.${NC}"
            exit 1
        fi
    fi
fi

if [ "$NATIVE_MODE" = false ]; then
    build_gguf_fast_bridge
fi

# 2. Locate Artifacts
echo -e "${BLUE}>>> Locating artifacts...${NC}"

if [ "$NATIVE_MODE" = true ]; then
    # 2a. Locate Native Executable
    NATIVE_PATH="${PROJECT_ROOT}/${CLI_MODULE}/build/gollek"
    if [ ! -f "$NATIVE_PATH" ]; then
        NATIVE_PATH=$(find "${PROJECT_ROOT}/${CLI_MODULE}/build" -type f -perm -111 -name "*-runner" -maxdepth 3 | head -n 1)
    fi
    if [ -z "$NATIVE_PATH" ] || [ ! -f "$NATIVE_PATH" ]; then
        # Compatibility fallback for legacy Maven-native layouts.
        NATIVE_PATH="${PROJECT_ROOT}/${CLI_MODULE}/target/gollek"
    fi
    if [ -z "$NATIVE_PATH" ] || [ ! -f "$NATIVE_PATH" ]; then
        NATIVE_PATH=$(find "${PROJECT_ROOT}/${CLI_MODULE}/target" -type f -perm -111 -not -name "*.jar" -not -name "*.sh" -maxdepth 2 | head -n 1)
    fi

    if [ -z "$NATIVE_PATH" ] || [ ! -f "$NATIVE_PATH" ]; then
        echo -e "${RED}❌ Could not find native executable in ${CLI_MODULE}/build or ${CLI_MODULE}/target${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Found Native Binary: $(basename "$NATIVE_PATH")${NC}"

    # 2b. Locate SDK Shared Library
    echo -e "${BLUE}>>> Locating SDK Shared Library...${NC}"
    LIB_NAME="libgollek_sdk_local.dylib"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then LIB_NAME="libgollek_sdk_local.so"; fi
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then LIB_NAME="gollek_sdk_local.dll"; fi
    
    SDK_LIB_PATH="${PROJECT_ROOT}/${SDK_MODULE}/target/${LIB_NAME}"
    if [ ! -f "$SDK_LIB_PATH" ]; then
        # Fallback search
        SDK_LIB_PATH=$(find "${PROJECT_ROOT}/${SDK_MODULE}/target" -name "*gollek_sdk_local*" -not -name "*.jar" -maxdepth 1 | head -n 1)
    fi
    
    if [ -n "$SDK_LIB_PATH" ] && [ -f "$SDK_LIB_PATH" ]; then
        echo -e "${GREEN}✓ Found SDK Library: $(basename "$SDK_LIB_PATH")${NC}"
        mkdir -p "$LIB_DIR"
        cp "$SDK_LIB_PATH" "$LIB_DIR/"
        echo -e "${GREEN}✓ Installed SDK Library to ${LIB_DIR}/${NC}"
    else
        echo -e "${YELLOW}⚠ Could not find SDK shared library. Native SDK features might be unavailable.${NC}"
    fi
else
    jar_contains_entry() {
        local listing_file="$1"
        local entry="$2"
        grep -Fxq "$entry" "$listing_file"
    }

    is_runnable_cli_jar() {
        local candidate="$1" listing_file ok
        [ -n "$candidate" ] && [ -f "$candidate" ] || return 1
        listing_file="$(mktemp "${TMPDIR:-/tmp}/gollek-cli-jar-list.XXXXXX")" || return 1
        if ! env -u JAVA_TOOL_OPTIONS jar tf "$candidate" >"$listing_file" 2>/dev/null; then
            rm -f "$listing_file"
            return 1
        fi
        ok=0
        jar_contains_entry "$listing_file" "tech/kayys/gollek/cli/commands/LiteRtLmFastRun.class" || ok=1
        jar_contains_entry "$listing_file" "com/google/ai/edge/litertlm/Engine.class" || ok=1
        jar_contains_entry "$listing_file" "com/google/ai/edge/litertlm/Backend.class" || ok=1
        rm -f "$listing_file"
        return "$ok"
    }

    locate_quarkus_gradle_artifact() {
        local properties_file artifact_path resolved_path
        for properties_file in \
            "${CLI_GRADLE_QUARKUS_GEN_DIR}/quarkus-artifact.properties" \
            "${PROJECT_ROOT}/${CLI_MODULE}/build/quarkus-artifact.properties"; do
            [ -f "$properties_file" ] || continue
            artifact_path="$(sed -n 's/^path=//p' "$properties_file" | tail -n 1)"
            [ -n "$artifact_path" ] || continue
            case "$artifact_path" in
                /*) resolved_path="$artifact_path" ;;
                *) resolved_path="$(dirname "$properties_file")/${artifact_path}" ;;
            esac
            if is_runnable_cli_jar "$resolved_path"; then
                printf '%s\n' "$resolved_path"
                return 0
            fi
        done
        return 1
    }

    # Locate JAR
    JAR_PATH=""
    if is_runnable_cli_jar "${CLI_GRADLE_JAR}"; then
        JAR_PATH="${CLI_GRADLE_JAR}"
    fi
    if [ -z "$JAR_PATH" ] && is_runnable_cli_jar "${CLI_GRADLE_QUARKUS_GEN_JAR}"; then
        JAR_PATH="${CLI_GRADLE_QUARKUS_GEN_JAR}"
    fi
    if [ -z "$JAR_PATH" ]; then
        JAR_PATH="$(locate_quarkus_gradle_artifact || true)"
    fi
    if [ -z "$JAR_PATH" ]; then
        while IFS= read -r candidate; do
            if is_runnable_cli_jar "$candidate"; then
                JAR_PATH="$candidate"
                break
            fi
        done < <(find "${PROJECT_ROOT}/${CLI_MODULE}/build" -maxdepth 4 -name "gollek*.jar" -not -path "*/build/libs/*" 2>/dev/null | sort)
    fi
    if [ -z "$JAR_PATH" ]; then
        # Compatibility fallback for legacy Maven packagers.
        if is_runnable_cli_jar "${PROJECT_ROOT}/${CLI_MODULE}/target/gollek.jar"; then
            JAR_PATH="${PROJECT_ROOT}/${CLI_MODULE}/target/gollek.jar"
        fi
    fi
    if [ -z "$JAR_PATH" ]; then
        while IFS= read -r candidate; do
            if is_runnable_cli_jar "$candidate"; then
                JAR_PATH="$candidate"
                break
            fi
        done < <(find "${PROJECT_ROOT}/${CLI_MODULE}/target" -maxdepth 2 -name "gollek*.jar" 2>/dev/null | sort)
    fi
    if is_runnable_cli_jar "${CLI_GRADLE_JAR}"; then
        JAR_PATH="${CLI_GRADLE_JAR}"
    fi

    if [ -z "$JAR_PATH" ] || [ ! -f "$JAR_PATH" ]; then
        echo -e "${RED}❌ Could not find a runnable Gollek CLI fat JAR in ${CLI_MODULE}/build/quarkus-build/gen, ${CLI_MODULE}/build, or ${CLI_MODULE}/target${NC}"
        echo -e "${YELLOW}  The slim Gradle jar under build/libs is intentionally rejected because it lacks runtime dependencies such as LiteRT-LM.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Found JAR: $(basename "$JAR_PATH")${NC}"

    if [ "$SKIP_LEGACY_SOURCE_HOTPATCHES" = "true" ]; then
        echo -e "${YELLOW}⚠ Skipping legacy post-build jar hot-patches on Gradle install path.${NC}"
        echo -e "${YELLOW}  Gradle/Quarkus output is treated as the source of truth. Set GOLLEK_SKIP_LEGACY_SOURCE_HOTPATCHES=false to re-enable the old patch path.${NC}"
    else
    # Patch GGUFTokenizer directly into the packaged jar so local runs use the
    # updated runtime-facing tokenizer even when the standalone GGUF reactor
    # is not buildable in this checkout.
    TOKENIZER_PATCH_SRC="${PROJECT_ROOT}/runner/gguf/gollek-gguf-core/src/main/java/tech/kayys/gollek/gguf/tokenizer/GGUFTokenizer.java"
    TOKENIZER_PATCH_OUT="${PROJECT_ROOT}/tools/.build-tokenizer"
    BPE_TOKENIZER_PATCH_SRC="${PROJECT_ROOT}/core/gollek-tokenizer-core/src/main/java/tech/kayys/gollek/tokenizer/impl/BpeTokenizer.java"
    BPE_TOKENIZER_PATCH_OUT="${PROJECT_ROOT}/tools/.build-bpe-tokenizer"
    LITERT_PATCH_OUT="${PROJECT_ROOT}/tools/.build-litert-runner"
    LITERT_TOKENIZER_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTTokenizer.java"
    LITERT_BINDINGS_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTNativeBindings.java"
    LITERT_INFERENCE_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTInferenceRunner.java"
    LITERT_NATIVE_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTGemmaNativeRunner.java"
    LITERT_GEMMA_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTGemmaMetalRunner.java"
    LITERT_CPU_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTCpuRunner.java"
    LITERT_PROVIDER_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTProvider.java"
    CHAT_TEMPLATE_PATCH_SRC="${PROJECT_ROOT}/core/gollek-tokenizer-core/src/main/java/tech/kayys/gollek/tokenizer/template/ChatTemplateFormatter.java"
    CHAT_TEMPLATE_PATCH_OUT="${PROJECT_ROOT}/tools/.build-chat-template"
    GEMMA_FAMILY_PATCH_SRC="${PROJECT_ROOT}/models/gollek-model-gemma/src/main/java/tech/kayys/gollek/models/GemmaFamily.java"
    GEMMA_FAMILY_PATCH_OUT="${PROJECT_ROOT}/tools/.build-gemma-family"
    MODEL_CONFIG_PATCH_SRC="${PROJECT_ROOT}/spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/ModelConfig.java"
    MODEL_CONFIG_PATCH_OUT="${PROJECT_ROOT}/tools/.build-model-config"
    ACCEL_TENSOR_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-core/src/main/java/tech/kayys/gollek/safetensor/core/tensor/AccelTensor.java"
    ACCEL_TENSOR_PATCH_OUT="${PROJECT_ROOT}/tools/.build-accel-tensor"
    ACCEL_OPS_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-core/src/main/java/tech/kayys/gollek/safetensor/core/tensor/AccelOps.java"
    ACCEL_OPS_PATCH_OUT="${PROJECT_ROOT}/tools/.build-accel-ops"
    DIRECT_FORWARD_PASS_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/forward/DirectForwardPass.java"
    DIRECT_INFERENCE_ENGINE_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/generation/DirectInferenceEngine.java"
    FLASH_ATTENTION_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel.java"
    ROPE_FREQUENCY_CACHE_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/generation/attention/RopeFrequencyCache.java"
    KV_CACHE_MANAGER_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/generation/kv/KVCacheManager.java"
    SAFETENSOR_ENGINE_PATCH_OUT="${PROJECT_ROOT}/tools/.build-safetensor-engine"
    METAL_FA4_PATCH_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/binding/MetalFlashAttentionBinding.java"
    METAL_FA4_PATCH_OUT="${PROJECT_ROOT}/tools/.build-metal-fa4"
    METAL_RUNNER_MODE_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/config/MetalRunnerMode.java"
    METAL_CAPABILITIES_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/detection/MetalCapabilities.java"
    METAL_APPLE_DETECTOR_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/detection/AppleSiliconDetector.java"
    METAL_RUNNER_PATCH_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/runner/MetalRunner.java"
    METAL_RUNNER_PATCH_OUT="${PROJECT_ROOT}/tools/.build-metal-runner"
    METAL_OFFLOAD_PATCH_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/runner/MetalWeightOffloadingRunner.java"
    METAL_OFFLOAD_PATCH_OUT="${PROJECT_ROOT}/tools/.build-metal-offload"
    RUNNER_METADATA_PATCH_SRC="${PROJECT_ROOT}/spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/RunnerMetadata.java"
    METAL_FA4_PREBUILT_CLASS="${METAL_CLASSES_GRADLE}/tech/kayys/gollek/metal/binding/MetalFlashAttentionBinding.class"
    METAL_RUNNER_PREBUILT_CLASS="${METAL_CLASSES_GRADLE}/tech/kayys/gollek/metal/runner/MetalRunner.class"
    METAL_OFFLOAD_PREBUILT_CLASS="${METAL_CLASSES_GRADLE}/tech/kayys/gollek/metal/runner/MetalWeightOffloadingRunner.class"
    if [ -f "$TOKENIZER_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling GGUF tokenizer patch...${NC}"
        mkdir -p "$TOKENIZER_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$TOKENIZER_PATCH_OUT" "$TOKENIZER_PATCH_SRC"
        (
            cd "$TOKENIZER_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/gguf/tokenizer/GGUFTokenizer.class
        )
        echo -e "${GREEN}✓ Patched GGUFTokenizer.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping GGUF tokenizer patch (source file not found).${NC}"
    fi

    if [ -f "$BPE_TOKENIZER_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling BPE tokenizer patch...${NC}"
        mkdir -p "$BPE_TOKENIZER_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$BPE_TOKENIZER_PATCH_OUT" "$BPE_TOKENIZER_PATCH_SRC"
        (
            cd "$BPE_TOKENIZER_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/tokenizer/impl/BpeTokenizer.class
        )
        echo -e "${GREEN}✓ Patched BpeTokenizer.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping BPE tokenizer patch (source file not found).${NC}"
    fi

    if [ -f "$LITERT_TOKENIZER_PATCH_SRC" ] && [ -f "$LITERT_INFERENCE_PATCH_SRC" ] \
        && [ -f "$LITERT_GEMMA_PATCH_SRC" ] && [ -f "$LITERT_CPU_PATCH_SRC" ] \
        && [ -f "$LITERT_PROVIDER_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling LiteRT runtime patches...${NC}"
        mkdir -p "$LITERT_PATCH_OUT"
        javac --release 25 -cp "$JAR_PATH" -d "$LITERT_PATCH_OUT" \
            "$LITERT_TOKENIZER_PATCH_SRC" \
            "$LITERT_BINDINGS_PATCH_SRC" \
            "$LITERT_INFERENCE_PATCH_SRC" \
            "$LITERT_NATIVE_PATCH_SRC" \
            "$LITERT_GEMMA_PATCH_SRC" \
            "$LITERT_CPU_PATCH_SRC" \
            "$LITERT_PROVIDER_PATCH_SRC"
        (
            cd "$LITERT_PATCH_OUT"
            zip -q "$JAR_PATH" \
                tech/kayys/gollek/provider/litert/LiteRTTokenizer*.class \
                tech/kayys/gollek/provider/litert/LiteRTNativeBindings*.class \
                tech/kayys/gollek/provider/litert/LiteRTInferenceRunner*.class \
                tech/kayys/gollek/provider/litert/LiteRTGemmaNativeRunner*.class \
                tech/kayys/gollek/provider/litert/LiteRTGemmaMetalRunner*.class \
                tech/kayys/gollek/provider/litert/LiteRTCpuRunner*.class \
                tech/kayys/gollek/provider/litert/LiteRTProvider*.class
        )
        echo -e "${GREEN}✓ Patched LiteRT runtime classes into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping LiteRT runtime patches (source file not found).${NC}"
    fi

    if [ -f "$CHAT_TEMPLATE_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling chat template patch...${NC}"
        mkdir -p "$CHAT_TEMPLATE_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$CHAT_TEMPLATE_PATCH_OUT" "$CHAT_TEMPLATE_PATCH_SRC"
        (
            cd "$CHAT_TEMPLATE_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/models/core/ChatTemplateFormatter.class
        )
        echo -e "${GREEN}✓ Patched ChatTemplateFormatter.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping chat template patch (source file not found).${NC}"
    fi

    if [ "$SKIP_LEGACY_SOURCE_HOTPATCHES" = "true" ]; then
        echo -e "${YELLOW}⚠ Skipping legacy Gemma family jar patch on Gradle install path (keeps CDI bean discovery owned by model modules).${NC}"
    elif [ -f "$GEMMA_FAMILY_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling Gemma family patch...${NC}"
        mkdir -p "$GEMMA_FAMILY_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$GEMMA_FAMILY_PATCH_OUT" "$GEMMA_FAMILY_PATCH_SRC"
        (
            cd "$GEMMA_FAMILY_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/models/GemmaFamily.class
        )
        echo -e "${GREEN}✓ Patched GemmaFamily.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping Gemma family patch (source file not found).${NC}"
    fi

    if [ -f "$MODEL_CONFIG_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling ModelConfig patch...${NC}"
        mkdir -p "$MODEL_CONFIG_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$MODEL_CONFIG_PATCH_OUT" "$MODEL_CONFIG_PATCH_SRC"
        (
            cd "$MODEL_CONFIG_PATCH_OUT"
            zip -q "$JAR_PATH" \
                tech/kayys/gollek/spi/model/ModelConfig.class \
                tech/kayys/gollek/spi/model/ModelConfig\$RopeScaling.class
        )
        echo -e "${GREEN}✓ Patched ModelConfig.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping ModelConfig patch (source file not found).${NC}"
    fi

    if [ -f "$ACCEL_TENSOR_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling AccelTensor patch...${NC}"
        mkdir -p "$ACCEL_TENSOR_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$ACCEL_TENSOR_PATCH_OUT" "$ACCEL_TENSOR_PATCH_SRC"
        (
            cd "$ACCEL_TENSOR_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/safetensor/core/tensor/AccelTensor*.class
        )
        echo -e "${GREEN}✓ Patched AccelTensor*.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping AccelTensor patch (source file not found).${NC}"
    fi

    if [ -f "$ACCEL_OPS_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling AccelOps patch...${NC}"
        mkdir -p "$ACCEL_OPS_PATCH_OUT"
        javac --release 25 --enable-preview --add-modules jdk.incubator.vector \
            -cp "$JAR_PATH" -d "$ACCEL_OPS_PATCH_OUT" "$ACCEL_OPS_PATCH_SRC"
        (
            cd "$ACCEL_OPS_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/safetensor/core/tensor/AccelOps*.class
        )
        echo -e "${GREEN}✓ Patched AccelOps*.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping AccelOps patch (source file not found).${NC}"
    fi

    if [ -d "$SAFETENSOR_ENGINE_CLASSES" ]; then
            echo -e "${BLUE}>>> Patching safetensor engine class set from Gradle build output...${NC}"
            (
                cd "$SAFETENSOR_ENGINE_CLASSES"
                zip -q "$JAR_PATH" \
                    tech/kayys/gollek/safetensor/engine/forward/DirectForwardPass*.class \
                    tech/kayys/gollek/safetensor/engine/generation/DirectInferenceEngine*.class \
                    tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel*.class \
                    tech/kayys/gollek/safetensor/engine/generation/attention/RopeFrequencyCache*.class \
                    tech/kayys/gollek/safetensor/engine/generation/kv/KVCacheManager*.class
            )
            echo -e "${GREEN}✓ Patched safetensor engine class set into jar via Gradle module classes${NC}"
    else
        echo -e "${BLUE}>>> Compiling safetensor engine patch set directly against the packaged jar...${NC}"
        mkdir -p "$SAFETENSOR_ENGINE_PATCH_OUT"
        if javac --release 25 --enable-preview --add-modules jdk.incubator.vector \
            -cp "$JAR_PATH" -d "$SAFETENSOR_ENGINE_PATCH_OUT" \
            "$DIRECT_FORWARD_PASS_PATCH_SRC" \
            "$DIRECT_INFERENCE_ENGINE_PATCH_SRC" \
            "$FLASH_ATTENTION_PATCH_SRC" \
            "$ROPE_FREQUENCY_CACHE_PATCH_SRC" \
            "$KV_CACHE_MANAGER_PATCH_SRC"; then
            (
                cd "$SAFETENSOR_ENGINE_PATCH_OUT"
                zip -q "$JAR_PATH" \
                    tech/kayys/gollek/safetensor/engine/forward/DirectForwardPass*.class \
                    tech/kayys/gollek/safetensor/engine/generation/DirectInferenceEngine*.class \
                    tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel*.class \
                    tech/kayys/gollek/safetensor/engine/generation/attention/RopeFrequencyCache*.class \
                    tech/kayys/gollek/safetensor/engine/generation/kv/KVCacheManager*.class
            )
            echo -e "${GREEN}✓ Patched safetensor engine class set into jar via direct source compile${NC}"
        else
            echo -e "${YELLOW}⚠ Direct safetensor engine source patch compile failed; trying prebuilt module classes fallback.${NC}"
            if [ -d "$SAFETENSOR_ENGINE_CLASSES" ]; then
                (
                    cd "$SAFETENSOR_ENGINE_CLASSES"
                    zip -q "$JAR_PATH" \
                        tech/kayys/gollek/safetensor/engine/forward/DirectForwardPass*.class \
                        tech/kayys/gollek/safetensor/engine/generation/DirectInferenceEngine*.class \
                        tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel*.class \
                        tech/kayys/gollek/safetensor/engine/generation/attention/RopeFrequencyCache*.class \
                        tech/kayys/gollek/safetensor/engine/generation/kv/KVCacheManager*.class
                )
                echo -e "${GREEN}✓ Patched safetensor engine class set into jar via prebuilt module classes${NC}"
            else
                echo -e "${YELLOW}⚠ Skipping safetensor engine patch set (direct source compile and prebuilt class fallback both failed).${NC}"
            fi
        fi
    fi

    if [ -f "$METAL_FA4_PATCH_SRC" ]; then
        if [ -f "$METAL_FA4_PREBUILT_CLASS" ]; then
            echo -e "${BLUE}>>> Patching Metal FA4 binding from Gradle build output...${NC}"
            (
                cd "$METAL_CLASSES_GRADLE"
                zip -q "$JAR_PATH" tech/kayys/gollek/metal/binding/MetalFlashAttentionBinding.class
            )
            echo -e "${GREEN}✓ Patched MetalFlashAttentionBinding.class into jar via Gradle module classes${NC}"
        else
            echo -e "${BLUE}>>> Compiling Metal FA4 patch...${NC}"
            mkdir -p "$METAL_FA4_PATCH_OUT"
            javac -cp "$JAR_PATH" -d "$METAL_FA4_PATCH_OUT" "$METAL_FA4_PATCH_SRC"
            (
                cd "$METAL_FA4_PATCH_OUT"
                zip -q "$JAR_PATH" tech/kayys/gollek/metal/binding/MetalFlashAttentionBinding.class
            )
            echo -e "${GREEN}✓ Patched MetalFlashAttentionBinding.class into jar${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Skipping Metal FA4 patch (source file not found).${NC}"
    fi

    if [ -f "$METAL_RUNNER_PATCH_SRC" ]; then
        if [ -f "$METAL_RUNNER_PREBUILT_CLASS" ]; then
            echo -e "${BLUE}>>> Patching Metal runner from Gradle build output...${NC}"
            (
                cd "$METAL_CLASSES_GRADLE"
                zip -q "$JAR_PATH" \
                    tech/kayys/gollek/metal/runner/MetalRunner.class
            )
            echo -e "${GREEN}✓ Patched MetalRunner.class into jar via Gradle module classes${NC}"
        elif [ "$SKIP_LEGACY_SOURCE_HOTPATCHES" = "true" ]; then
            echo -e "${YELLOW}⚠ Skipping legacy Metal runner hot-patch on Gradle install path (runner slice is not part of the maintained Gradle build output).${NC}"
        else
            echo -e "${BLUE}>>> Compiling Metal runner patch...${NC}"
            mkdir -p "$METAL_RUNNER_PATCH_OUT"
            if javac -cp "$JAR_PATH" -d "$METAL_RUNNER_PATCH_OUT" \
                "$RUNNER_METADATA_PATCH_SRC" \
                "$METAL_RUNNER_MODE_SRC" \
                "$METAL_CAPABILITIES_SRC" \
                "$METAL_APPLE_DETECTOR_SRC" \
                "$METAL_RUNNER_PATCH_SRC"; then
                (
                    cd "$METAL_RUNNER_PATCH_OUT"
                    zip -q "$JAR_PATH" \
                        tech/kayys/gollek/spi/model/RunnerMetadata.class \
                        tech/kayys/gollek/metal/config/MetalRunnerMode.class \
                        tech/kayys/gollek/metal/detection/MetalCapabilities.class \
                        tech/kayys/gollek/metal/detection/AppleSiliconDetector.class \
                        tech/kayys/gollek/metal/runner/MetalRunner.class
                )
                echo -e "${GREEN}✓ Patched MetalRunner.class into jar${NC}"
            else
                echo -e "${YELLOW}⚠ Skipping Metal runner patch (compile failed in current packaged classpath).${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}⚠ Skipping Metal runner patch (source file not found).${NC}"
    fi

    if [ -f "$METAL_OFFLOAD_PATCH_SRC" ]; then
        if [ -f "$METAL_OFFLOAD_PREBUILT_CLASS" ]; then
            echo -e "${BLUE}>>> Patching Metal offload runner from Gradle build output...${NC}"
            (
                cd "$METAL_CLASSES_GRADLE"
                zip -q "$JAR_PATH" \
                    tech/kayys/gollek/metal/runner/MetalWeightOffloadingRunner.class
            )
            echo -e "${GREEN}✓ Patched MetalWeightOffloadingRunner.class into jar via Gradle module classes${NC}"
        elif [ "$SKIP_LEGACY_SOURCE_HOTPATCHES" = "true" ]; then
            echo -e "${YELLOW}⚠ Skipping legacy Metal offload runner hot-patch on Gradle install path (runner slice is not part of the maintained Gradle build output).${NC}"
        else
            echo -e "${BLUE}>>> Compiling Metal offload runner patch...${NC}"
            mkdir -p "$METAL_OFFLOAD_PATCH_OUT"
            if javac -cp "$JAR_PATH" -d "$METAL_OFFLOAD_PATCH_OUT" \
                "$RUNNER_METADATA_PATCH_SRC" \
                "$METAL_RUNNER_MODE_SRC" \
                "$METAL_CAPABILITIES_SRC" \
                "$METAL_APPLE_DETECTOR_SRC" \
                "$METAL_OFFLOAD_PATCH_SRC"; then
                (
                    cd "$METAL_OFFLOAD_PATCH_OUT"
                    zip -q "$JAR_PATH" \
                        tech/kayys/gollek/spi/model/RunnerMetadata.class \
                        tech/kayys/gollek/metal/config/MetalRunnerMode.class \
                        tech/kayys/gollek/metal/detection/MetalCapabilities.class \
                        tech/kayys/gollek/metal/detection/AppleSiliconDetector.class \
                        tech/kayys/gollek/metal/runner/MetalWeightOffloadingRunner.class
                )
                echo -e "${GREEN}✓ Patched MetalWeightOffloadingRunner.class into jar${NC}"
            else
                echo -e "${YELLOW}⚠ Skipping Metal offload runner patch (compile failed in current packaged classpath).${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}⚠ Skipping Metal offload runner patch (source file not found).${NC}"
    fi

    # Ensure packaged jar includes Metal backend classes whenever safetensor
    # engine classes that reference tech.kayys.gollek.metal.* are patched in.
    if [ -d "${METAL_CLASSES_GRADLE}/tech/kayys/gollek/metal" ]; then
        echo -e "${BLUE}>>> Injecting full Metal backend class tree into packaged jar...${NC}"
        (
            cd "$METAL_CLASSES_GRADLE"
            zip -q -r "$JAR_PATH" tech/kayys/gollek/metal
        )
        echo -e "${GREEN}✓ Injected tech/kayys/gollek/metal/** into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping full Metal class injection (Gradle metal classes not found).${NC}"
    fi

    # Patch bundled classes for known dev-runtime issues in the current uber-jar.
    ASM_JAR="${HOME}/.m2/repository/org/ow2/asm/asm/9.9.1/asm-9.9.1.jar"
    ASM_TREE_JAR="${HOME}/.m2/repository/org/ow2/asm/asm-tree/9.9.1/asm-tree-9.9.1.jar"
    PATCH_TOOL_SRC="${PROJECT_ROOT}/tools/DisableDirectForwardMetal.java"
    PATCH_TOOL_OUT="${PROJECT_ROOT}/tools/.build"
    if [ -f "$ASM_JAR" ] && [ -f "$ASM_TREE_JAR" ] && [ -f "$PATCH_TOOL_SRC" ]; then
        echo -e "${BLUE}>>> Applying dev jar patches...${NC}"
        mkdir -p "$PATCH_TOOL_OUT"
        javac -cp "${ASM_JAR}:${ASM_TREE_JAR}" -d "$PATCH_TOOL_OUT" "$PATCH_TOOL_SRC"
        if java -cp "${PATCH_TOOL_OUT}:${ASM_JAR}:${ASM_TREE_JAR}" DisableDirectForwardMetal "$JAR_PATH"; then
            echo -e "${GREEN}✓ Applied jar patches${NC}"
        else
            echo -e "${YELLOW}⚠ Dev jar patch tool failed on the current packaged layout; continuing with the built jar.${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Skipping jar patches (ASM tooling not available).${NC}"
    fi

    echo -e "${BLUE}>>> Verifying patched runtime classes...${NC}"
    if ! javap -p -classpath "$JAR_PATH" tech.kayys.gollek.safetensor.core.tensor.AccelTensor \
        | grep -q "dequantizeCachedUpTo(long)"; then
        echo -e "${RED}❌ Patched jar verification failed: AccelTensor.dequantizeCachedUpTo(long) missing${NC}"
        exit 1
    fi
    if ! javap -p -classpath "$JAR_PATH" tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass \
        | grep -q "tryCachedFfnDownHalfLinear"; then
        echo -e "${RED}❌ Patched jar verification failed: DirectForwardPass.tryCachedFfnDownHalfLinear missing${NC}"
        exit 1
    fi
    if ! javap -classpath "$JAR_PATH" tech.kayys.gollek.models.core.ChatTemplateFormatter \
        | grep -q "format"; then
        echo -e "${RED}❌ Patched jar verification failed: ChatTemplateFormatter not loadable${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Verified patched runtime symbols in jar${NC}"
    fi
fi

# Install native tokenizer dependencies for local runtime usage.
install_tokenizer_native_runtime

# 3. Create Local Bin Directory
mkdir -p "$BIN_DIR"

if [ "$NATIVE_MODE" = false ]; then
    mkdir -p "$RUNTIME_DIR"
    cp "$JAR_PATH" "${GOLLEK_RUNTIME_JAR}.tmp"
    mv "${GOLLEK_RUNTIME_JAR}.tmp" "$GOLLEK_RUNTIME_JAR"
    JAR_PATH="$GOLLEK_RUNTIME_JAR"
    echo -e "${GREEN}✓ Installed CLI runtime jar to ${GOLLEK_RUNTIME_JAR}${NC}"

    # Also install API server runtime
    API_BUILD_DIR="${PROJECT_ROOT}/ui/gollek-api/build/quarkus-app"
    if [ -d "$API_BUILD_DIR" ]; then
        mkdir -p "${RUNTIME_DIR}/api"
        rm -rf "${RUNTIME_DIR}/api"/*
        cp -R "${API_BUILD_DIR}"/* "${RUNTIME_DIR}/api/"
        echo -e "${GREEN}✓ Installed API server runtime to ${RUNTIME_DIR}/api/quarkus-run.jar${NC}"
    fi
fi

# 4. Create the Shim/Binary link
echo -e "${BLUE}>>> Creating executable at ${GOLLEK_CLI_BIN}...${NC}"

if [ "$NATIVE_MODE" = true ]; then
    # For native, we create a shim that sets library paths but calls the binary directly
    cat <<EOF > "$GOLLEK_CLI_BIN"
#!/bin/bash
# Local Gollek CLI Native Shim (Generated by install-local-macos.sh)
# Points to: $NATIVE_PATH

# Ensure log directory exists
mkdir -p "\$HOME/.gollek/logs"

# Launch native binary
exec "$NATIVE_PATH" "\$@"
EOF
else
    # Standard JAR Shim - Optimized for Java 25 with FFM/Vector support
    cat <<EOF > "$GOLLEK_CLI_BIN"
#!/bin/bash
# Local Gollek CLI Shim (Generated by install-local-macos.sh)
# Points to: $JAR_PATH

# Ensure Java 25 is used if possible without paying SDKMAN startup cost on every run.
if [ -n "\${GOLLEK_JAVA_HOME:-}" ]; then
    export JAVA_HOME="\$GOLLEK_JAVA_HOME"
    export PATH="\$JAVA_HOME/bin:\$PATH"
elif command -v /usr/libexec/java_home >/dev/null 2>&1; then
    JAVA_25_HOME="\$(/usr/libexec/java_home -v 25 2>/dev/null || true)"
    if [ -n "\$JAVA_25_HOME" ]; then
        export JAVA_HOME="\$JAVA_25_HOME"
        export PATH="\$JAVA_HOME/bin:\$PATH"
    fi
elif [ -s "\$HOME/.sdkman/bin/sdkman-init.sh" ]; then
    source "\$HOME/.sdkman/bin/sdkman-init.sh"
    sdk use java 25-open >/dev/null 2>&1 || true
fi

# Ensure log directory exists
mkdir -p "\$HOME/.gollek/logs"

# Give native GGUF runs enough direct-memory headroom by default.
: "\${GOLLEK_MAX_DIRECT_MEMORY:=4g}"
# Gemma LiteRT-LM CPU fallback moves large KV-cache tensors through Java.
# A 4 GiB default heap can trigger GC pressure on Apple Silicon.
: "\${GOLLEK_JAVA_HEAP:=2g}"
# Keep the official LiteRT-LM engine warm between repeated local run calls.
# Set GOLLEK_LITERT_FAST_DAEMON=false to force one-shot process behavior.
: "\${GOLLEK_LITERT_FAST_DAEMON:=true}"
# LiteRT-LM uses auto by default: launchctl on macOS for a truly detached
# warmed daemon, with nohup fallback when launchctl is unavailable.
: "\${GOLLEK_LITERT_FAST_DAEMON_LAUNCHER:=auto}"
# Short prompts should not force a 2048-token LiteRT-LM engine. Dynamic sizing
# keeps Metal warm latency low while preserving an env override for diagnostics.
: "\${GOLLEK_LITERT_FAST_DYNAMIC_ENGINE_TOKENS:=true}"
# Enable the native llama.cpp GGUF shortcut for simple local run calls.
: "\${GOLLEK_GGUF_FAST_RUN:=true}"
# Keep the llama.cpp GGUF fast path warm between repeated local run calls.
# Set GOLLEK_GGUF_FAST_DAEMON=false to force one-shot process behavior.
: "\${GOLLEK_GGUF_FAST_DAEMON:=true}"
# Keep the Metal-backed GGUF daemon in a separate macOS service so the parent
# hard-exit path cannot terminate its warm llama.cpp session.
if [ "\$(uname -s)" = "Darwin" ]; then
    : "\${GOLLEK_GGUF_FAST_DAEMON_LAUNCHER:=launchctl}"
else
    : "\${GOLLEK_GGUF_FAST_DAEMON_LAUNCHER:=nohup}"
fi
# Successful one-shot llama.cpp runs hard-exit before Metal teardown to avoid
# upstream ggml-metal finalizer assertions. Daemon runs are not affected.
: "\${GOLLEK_GGUF_FAST_HARD_EXIT_AFTER_RUN:=true}"
# Safetensor Metal runs are bandwidth-bound; stale GGUF/LiteRT daemons can hold
# gigabytes of memory and make short benchmarks look pathologically slow.
: "\${GOLLEK_SAFETENSOR_STOP_FAST_DAEMONS:=true}"

# LiteRT accelerator plugins are loaded by native code via dlopen, so they need
# the macOS dynamic loader path in addition to Java's library path.
export DYLD_LIBRARY_PATH="\$HOME/.gollek/libs:\$HOME/.gollek/libs/llama\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}"
export DYLD_FALLBACK_LIBRARY_PATH="\$HOME/.gollek/libs:\$HOME/.gollek/libs/llama\${DYLD_FALLBACK_LIBRARY_PATH:+:\$DYLD_FALLBACK_LIBRARY_PATH}"

gollek_gguf_fast_command() {
    case "\${1:-}" in
        run|prewarm|warmup|__gguf-daemon-stop|__gguf-daemon-prewarm) return 0 ;;
        *) return 1 ;;
    esac
}

gollek_litert_fast_command() {
    case "\${1:-}" in
        run|prewarm|warmup|__daemon-stop|__litert-daemon-prewarm) return 0 ;;
        *) return 1 ;;
    esac
}

gollek_lower() {
    printf '%s' "\${1:-}" | tr '[:upper:]' '[:lower:]'
}

gollek_provider_arg() {
    local arg
    while [ "\$#" -gt 0 ]; do
        arg="\$1"
        case "\$arg" in
            --provider=*)
                printf '%s\n' "\${arg#--provider=}"
                return 0
                ;;
            --provider)
                shift
                [ "\$#" -gt 0 ] && printf '%s\n' "\$1"
                return 0
                ;;
        esac
        shift
    done
    return 1
}

gollek_model_arg() {
    local arg
    while [ "\$#" -gt 0 ]; do
        arg="\$1"
        case "\$arg" in
            --model=*|--modelFile=*|--model-file=*|--model-path=*|-m=*)
                printf '%s\n' "\${arg#*=}"
                return 0
                ;;
            --model|--modelFile|--model-file|--model-path|-m)
                shift
                [ "\$#" -gt 0 ] && printf '%s\n' "\$1"
                return 0
                ;;
        esac
        shift
    done
    return 1
}

gollek_litert_model_path() {
    local normalized
    normalized="\$(gollek_lower "\${1:-}")"
    case "\$normalized" in
        *.litertlm|*.task|*.tflite) return 0 ;;
        *) return 1 ;;
    esac
}

gollek_gguf_model_path() {
    local normalized
    normalized="\$(gollek_lower "\${1:-}")"
    case "\$normalized" in
        *.gguf) return 0 ;;
        *) return 1 ;;
    esac
}

gollek_sha256() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 | awk '{ print \$1 }'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{ print \$1 }'
    else
        return 1
    fi
}

gollek_normalized_path_for_cache() {
    local path dir base
    path="\${1:-}"
    [ -n "\$path" ] || return 1
    dir="\$(dirname "\$path" 2>/dev/null || true)"
    base="\$(basename "\$path" 2>/dev/null || true)"
    if [ -n "\$dir" ] && [ -n "\$base" ] && [ -d "\$dir" ]; then
        printf '%s/%s' "\$(cd "\$dir" 2>/dev/null && pwd -P)" "\$base"
    else
        printf '%s' "\$path"
    fi
}

gollek_gemma4_equivalent_key() {
    awk '
        {
            value = tolower(\$0)
            gsub(/__/, "/", value)
            gsub(/\\\\/, "/", value)
            gsub(/[.](litertlm|litert|tflite|tfl|task|gguf|safetensors?|bin)$/, " ", value)
            gsub(/qat/, " ", value)
            gsub(/mobile/, " ", value)
            gsub(/transformers/, " ", value)
            gsub(/safetensor/, " ", value)
            gsub(/safetensors/, " ", value)
            gsub(/google/, " ", value)
            gsub(/litert/, " ", value)
            gsub(/community/, " ", value)
            gsub(/qualcomm/, " ", value)
            count = split(value, tokens, /[^a-z0-9]+/)
            for (i = 1; i <= count; i++) {
                if (tokens[i] != "") {
                    printf "%s", tokens[i]
                }
            }
        }
    '
}

gollek_gemma4_litert_cache_key() {
    local requested model_path equivalent
    requested="\${1:-}"
    model_path="\${2:-}"
    [ -n "\$model_path" ] || return 1
    equivalent="\$(printf '%s %s' "\$requested" "\$model_path" | gollek_gemma4_equivalent_key)"
    printf '%s\n%s\n%s' "\$requested" "\$model_path" "\$equivalent" | gollek_sha256
}

gollek_index_field_for_ref() {
    local ref field index_path
    ref="\${1:-}"
    field="\${2:-}"
    index_path="\$HOME/.gollek/models/index.json"
    if [ -z "\$ref" ] || [ -z "\$field" ] || [ ! -f "\$index_path" ]; then
        return 1
    fi
    awk -v ref="\$ref" -v field="\$field" '
        function json_value(line, value) {
            value = line
            sub(/^[^:]*:[[:space:]]*"/, "", value)
            sub(/".*$/, "", value)
            return value
        }
        function object_boundary(line) {
            return line ~ /^[[:space:]]*}[,]?[[:space:]]*[{]?[[:space:]]*$/
        }
        {
            lower = tolower(\$0)
            if (lower ~ /"(id|shortid|name|path)"[[:space:]]*:/) {
                value = json_value(\$0)
                normalized_value = tolower(value)
                normalized_ref = tolower(ref)
                if (value == ref || normalized_value == normalized_ref) {
                    found = 1
                }
            }
            if (found && lower ~ "\"" tolower(field) "\"[[:space:]]*:") {
                print json_value(\$0)
                exit 0
            }
            if (object_boundary(\$0)) {
                found = 0
            }
        }
    ' "\$index_path"
}

gollek_index_format_for_ref() {
    gollek_index_field_for_ref "\${1:-}" "format"
}

gollek_index_path_for_ref() {
    gollek_index_field_for_ref "\${1:-}" "path"
}

gollek_cached_gemma4_litert_route_requested() {
    local model model_path normalized_model_path route_cache cache_key alt_cache_key
    model="\$(gollek_model_arg "\$@" 2>/dev/null || true)"
    [ -n "\$model" ] || return 1
    model_path="\$(gollek_index_path_for_ref "\$model" 2>/dev/null || true)"
    [ -n "\$model_path" ] || model_path="\$model"
    normalized_model_path="\$(gollek_normalized_path_for_cache "\$model_path" 2>/dev/null || true)"
    route_cache="\$HOME/.gollek/cache/gemma4-mobile-qat-litert-routes.tsv"
    [ -n "\$normalized_model_path" ] && [ -f "\$route_cache" ] || return 1
    cache_key="\$(gollek_gemma4_litert_cache_key "\$normalized_model_path" "\$normalized_model_path" 2>/dev/null || true)"
    alt_cache_key="\$(gollek_gemma4_litert_cache_key "\$model" "\$normalized_model_path" 2>/dev/null || true)"
    [ -n "\$cache_key" ] || [ -n "\$alt_cache_key" ] || return 1
    awk -F '\t' -v cache_key="\$cache_key" -v alt_cache_key="\$alt_cache_key" '
        (\$1 == cache_key || (alt_cache_key != "" && \$1 == alt_cache_key)) && \$2 ~ /[.]litertlm$/ {
            found = 1
            exit 0
        }
        END { exit found ? 0 : 1 }
    ' "\$route_cache"
}

gollek_litert_requested() {
    local provider model normalized
    case "\${1:-}" in
        __daemon-stop|__litert-daemon-prewarm) return 0 ;;
        __gguf-daemon-stop|__gguf-daemon-prewarm) return 1 ;;
    esac
    provider="\$(gollek_provider_arg "\$@" 2>/dev/null || true)"
    normalized="\$(gollek_lower "\$provider")"
    case "\$normalized" in
        litert|litertlm|lite-rt|tflite|task) return 0 ;;
        gguf|native|llamacpp|llama.cpp|llama-cpp|java|java-native|jvm) return 1 ;;
    esac
    model="\$(gollek_model_arg "\$@" 2>/dev/null || true)"
    if [ -n "\$model" ] && gollek_litert_model_path "\$model"; then
        return 0
    fi
    if gollek_cached_gemma4_litert_route_requested "\$@"; then
        return 0
    fi
    normalized="\$(gollek_lower "\$(gollek_index_format_for_ref "\$model" 2>/dev/null || true)")"
    case "\$normalized" in
        litert|litertlm|lite-rt|tflite|task) return 0 ;;
    esac
    return 1
}

gollek_gguf_requested() {
    local provider model normalized
    case "\${1:-}" in
        __gguf-daemon-stop|__gguf-daemon-prewarm) return 0 ;;
        __daemon-stop|__litert-daemon-prewarm) return 1 ;;
    esac
    provider="\$(gollek_provider_arg "\$@" 2>/dev/null || true)"
    normalized="\$(gollek_lower "\$provider")"
    case "\$normalized" in
        gguf|native|llamacpp|llama.cpp|llama-cpp|java|java-native|jvm) return 0 ;;
        litert|litertlm|lite-rt|tflite|task) return 1 ;;
    esac
    model="\$(gollek_model_arg "\$@" 2>/dev/null || true)"
    if [ -n "\$model" ] && gollek_gguf_model_path "\$model"; then
        return 0
    fi
    normalized="\$(gollek_lower "\$(gollek_index_format_for_ref "\$model" 2>/dev/null || true)")"
    case "\$normalized" in
        gguf) return 0 ;;
        litert|litertlm|lite-rt|tflite|task) return 1 ;;
    esac
    return 1
}

gollek_safetensor_command() {
    case "\${1:-}" in
        run|chat|prewarm|warmup) return 0 ;;
        *) return 1 ;;
    esac
}

gollek_safetensor_model_path() {
    local normalized
    normalized="\$(gollek_lower "\${1:-}")"
    case "\$normalized" in
        *.safetensors|*.safetensor|*/safetensors/*) return 0 ;;
        *) return 1 ;;
    esac
}

gollek_safetensor_requested() {
    local provider model normalized
    gollek_safetensor_command "\${1:-}" || return 1
    provider="\$(gollek_provider_arg "\$@" 2>/dev/null || true)"
    normalized="\$(gollek_lower "\$provider")"
    case "\$normalized" in
        safetensor|safetensors|safe-tensor|safe-tensors) return 0 ;;
        gguf|native|llamacpp|llama.cpp|llama-cpp|litert|litertlm|lite-rt|tflite|task) return 1 ;;
    esac
    model="\$(gollek_model_arg "\$@" 2>/dev/null || true)"
    if gollek_cached_gemma4_litert_route_requested "\$@"; then
        return 1
    fi
    if [ -n "\$model" ] && gollek_safetensor_model_path "\$model"; then
        return 0
    fi
    normalized="\$(gollek_lower "\$(gollek_index_format_for_ref "\$model" 2>/dev/null || true)")"
    case "\$normalized" in
        safetensor|safetensors|safe-tensor|safe-tensors) return 0 ;;
    esac
    return 1
}

gollek_bootout_fast_daemon_launchctl_jobs() {
    if [ "\$(uname -s)" != "Darwin" ] || ! command -v launchctl >/dev/null 2>&1; then
        return 0
    fi
    local uid
    uid="\$(id -u 2>/dev/null || true)"
    if [ -n "\$uid" ]; then
        launchctl kill TERM "gui/\${uid}/tech.kayys.gollek.gguf-fast-daemon" >/dev/null 2>&1 || true
        launchctl kill TERM "gui/\${uid}/tech.kayys.gollek.litert-fast-daemon" >/dev/null 2>&1 || true
        launchctl bootout "gui/\${uid}/tech.kayys.gollek.gguf-fast-daemon" >/dev/null 2>&1 || true
        launchctl bootout "gui/\${uid}/tech.kayys.gollek.litert-fast-daemon" >/dev/null 2>&1 || true
    fi
    launchctl remove tech.kayys.gollek.gguf-fast-daemon >/dev/null 2>&1 || true
    launchctl remove tech.kayys.gollek.litert-fast-daemon >/dev/null 2>&1 || true
}

gollek_is_daemon_command() {
    local command="\${1:-}"
    local class_name="\${2:-}"
    case " \$command " in
        *" \$class_name __daemon "*) return 0 ;;
        *" \$class_name "*" __daemon "*) return 0 ;;
        *) return 1 ;;
    esac
}

gollek_terminate_daemon_pid() {
    local pid="\${1:-}"
    local attempt
    case "\$pid" in
        ''|*[!0-9]*) return 0 ;;
    esac
    if [ "\$pid" = "\$\$" ] || ! kill -0 "\$pid" >/dev/null 2>&1; then
        return 0
    fi
    kill "\$pid" >/dev/null 2>&1 || true
    for attempt in 1 2 3 4 5; do
        if ! kill -0 "\$pid" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    kill -9 "\$pid" >/dev/null 2>&1 || true
}

gollek_stop_daemon_from_port_file() {
    local port_file="\${1:-}"
    local class_name="\${2:-}"
    local pid command
    [ -f "\$port_file" ] || return 0
    pid="\$(sed -n '2p' "\$port_file" 2>/dev/null | tr -d '[:space:]' || true)"
    case "\$pid" in
        ''|*[!0-9]*) return 0 ;;
    esac
    command="\$(ps -p "\$pid" -o command= 2>/dev/null || true)"
    gollek_is_daemon_command "\$command" "\$class_name" || return 0
    gollek_terminate_daemon_pid "\$pid"
}

gollek_stop_daemons_by_process_scan() {
    local class_name="\${1:-}"
    local line pid command
    command -v ps >/dev/null 2>&1 || return 0
    while IFS= read -r line; do
        line="\${line#"\${line%%[![:space:]]*}"}"
        pid="\${line%%[[:space:]]*}"
        command="\${line#"\$pid"}"
        command="\${command#"\${command%%[![:space:]]*}"}"
        case "\$pid" in
            ''|*[!0-9]*) continue ;;
        esac
        if gollek_is_daemon_command "\$command" "\$class_name"; then
            gollek_terminate_daemon_pid "\$pid"
        fi
    done < <(ps -axo pid=,command= 2>/dev/null || true)
}

gollek_stop_fast_daemons_for_safetensor() {
    if [ "\${GOLLEK_SAFETENSOR_STOP_FAST_DAEMONS:-true}" = "false" ]; then
        return 0
    fi
    gollek_safetensor_requested "\$@" || return 0
    gollek_bootout_fast_daemon_launchctl_jobs
    GOLLEK_GGUF_FAST_RUN=true GOLLEK_LITERT_FAST_RUN=false "\$0" __gguf-daemon-stop >/dev/null 2>&1 || true
    GOLLEK_GGUF_FAST_RUN=false GOLLEK_LITERT_FAST_RUN=true "\$0" __daemon-stop >/dev/null 2>&1 || true
    gollek_stop_daemon_from_port_file "\$HOME/.gollek/run/gguf-fast-daemon.port" "tech.kayys.gollek.cli.commands.GgufFastRun"
    gollek_stop_daemon_from_port_file "\$HOME/.gollek/run/litert-fast-daemon.port" "tech.kayys.gollek.cli.commands.LiteRtLmFastRun"
    gollek_stop_daemons_by_process_scan "tech.kayys.gollek.cli.commands.GgufFastRun"
    gollek_stop_daemons_by_process_scan "tech.kayys.gollek.cli.commands.LiteRtLmFastRun"
    gollek_bootout_fast_daemon_launchctl_jobs
    rm -f "\$HOME/.gollek/run/gguf-fast-daemon.port" "\$HOME/.gollek/run/litert-fast-daemon.port"
}

gollek_prompt_arg() {
    local arg
    while [ "\$#" -gt 0 ]; do
        arg="\$1"
        case "\$arg" in
            --prompt=*|-p=*)
                printf '%s\n' "\${arg#*=}"
                return 0
                ;;
            --prompt|-p)
                shift
                [ "\$#" -gt 0 ] && printf '%s\n' "\$1"
                return 0
                ;;
        esac
        shift
    done
    return 1
}

gollek_max_tokens_arg() {
    local arg
    while [ "\$#" -gt 0 ]; do
        arg="\$1"
        case "\$arg" in
            --max-tokens=*)
                printf '%s\n' "\${arg#*=}"
                return 0
                ;;
            --max-tokens)
                shift
                [ "\$#" -gt 0 ] && printf '%s\n' "\$1"
                return 0
                ;;
        esac
        shift
    done
    return 1
}

gollek_backend_arg() {
    local arg
    while [ "\$#" -gt 0 ]; do
        arg="\$1"
        case "\$arg" in
            --backend=*|--platform=*)
                printf '%s\n' "\${arg#*=}"
                return 0
                ;;
            --backend|--platform)
                shift
                [ "\$#" -gt 0 ] && printf '%s\n' "\$1"
                return 0
                ;;
        esac
        shift
    done
    return 1
}

gollek_has_flag() {
    local wanted="\$1"
    shift
    while [ "\$#" -gt 0 ]; do
        [ "\$1" = "\$wanted" ] && return 0
        shift
    done
    return 1
}

gollek_positional_after_command() {
    [ "\$#" -gt 0 ] || return 1
    shift
    local arg option
    while [ "\$#" -gt 0 ]; do
        arg="\$1"
        option="\$arg"
        case "\$arg" in
            --*=*) option="\${arg%%=*}" ;;
        esac
        case "\$option" in
            --model|--modelFile|--model-file|--model-path|-m|--provider|--prompt|-p|--max-tokens|--backend|--platform|--engine|--gguf-engine)
                case "\$arg" in
                    --*=*) ;;
                    *) shift ;;
                esac
                ;;
            --*) ;;
            -*) ;;
            *)
                printf '%s\n' "\$arg"
                return 0
                ;;
        esac
        shift
    done
    return 1
}

gollek_prewarm_auto_mode() {
    case "\${1:-}" in
        prewarm|warmup) ;;
        *) return 1 ;;
    esac
    local model provider normalized
    model="\$(gollek_model_arg "\$@" 2>/dev/null || true)"
    if [ -z "\$model" ]; then
        model="\$(gollek_positional_after_command "\$@" 2>/dev/null || true)"
    fi
    normalized="\$(gollek_lower "\$model")"
    case "\$normalized" in
        auto|auto:*|*:auto) ;;
        *) return 1 ;;
    esac
    case "\$normalized" in
        auto)
            provider="\$(gollek_lower "\$(gollek_provider_arg "\$@" 2>/dev/null || true)")"
            case "\$provider" in
                gguf|native|llamacpp|llama.cpp|llama-cpp) printf '%s\n' "auto:gguf" ;;
                litert|litertlm|lite-rt|tflite|task) printf '%s\n' "auto:litert" ;;
                *) printf '%s\n' "auto" ;;
            esac
            ;;
        *) printf '%s\n' "\$normalized" ;;
    esac
}

gollek_prewarm_auto() {
    local mode plan status ref prompt max_tokens backend overall
    mode="\$(gollek_prewarm_auto_mode "\$@")" || return 1
    plan="\$(java \\
        \${GOLLEK_JAVA_OPTS:-} \\
        -cp "$JAR_PATH" tech.kayys.gollek.cli.commands.PrewarmAutoPlan "\$mode" \\
        2> >(grep -v -e "WARNING: Using incubator modules" -e "^Picked up JAVA_TOOL_OPTIONS:" >&2))"
    status=\$?
    if [ "\$status" -ne 0 ] || [ -z "\$plan" ]; then
        echo "No local runnable GGUF/LiteRT-LM models were found for prewarm auto mode '\$mode'." >&2
        return 2
    fi

    prompt="\$(gollek_prompt_arg "\$@" 2>/dev/null || true)"
    max_tokens="\$(gollek_max_tokens_arg "\$@" 2>/dev/null || true)"
    backend="\$(gollek_backend_arg "\$@" 2>/dev/null || true)"
    overall=0
    while IFS= read -r ref; do
        [ -n "\$ref" ] || continue
        local args=("prewarm" "--model" "\$ref")
        [ -n "\$prompt" ] && args+=("--prompt" "\$prompt")
        [ -n "\$max_tokens" ] && args+=("--max-tokens" "\$max_tokens")
        [ -n "\$backend" ] && args+=("--backend" "\$backend")
        if gollek_has_flag "--use-cpu" "\$@"; then
            args+=("--use-cpu")
        fi
        "\$0" "\${args[@]}"
        status=\$?
        if [ "\$status" -ne 0 ]; then
            overall="\$status"
        fi
    done <<EOF_PREWARM_AUTO_PLAN
\$plan
EOF_PREWARM_AUTO_PLAN
    return "\$overall"
}

if gollek_prewarm_auto "\$@"; then
    exit 0
else
    PREWARM_AUTO_STATUS=\$?
    if [ "\$PREWARM_AUTO_STATUS" -ne 1 ]; then
        exit "\$PREWARM_AUTO_STATUS"
    fi
fi

# Safetensor Metal runs are bandwidth-bound; clean up before offering the command
# to unrelated fast-run shims so they cannot wake launchctl daemons first.
gollek_stop_fast_daemons_for_safetensor "\$@"

# Simple local text runs avoid the full Quarkus CLI startup by default.
# Fast runners use status 42 as an explicit "not mine; continue to the next
# CLI path" sentinel. Other non-zero statuses are real command failures and
# must not be hidden by silently launching another engine.
if [ "\${GOLLEK_GGUF_FAST_RUN:-true}" != "false" ] && gollek_gguf_fast_command "\${1:-}" && ! gollek_safetensor_requested "\$@" && ! gollek_litert_requested "\$@"; then
    java \\
        -Xmx"\${GOLLEK_JAVA_HEAP}" \\
        -XX:MaxDirectMemorySize="\${GOLLEK_MAX_DIRECT_MEMORY}" \\
        --enable-preview \\
        --add-modules jdk.incubator.vector \\
        --enable-native-access=ALL-UNNAMED \\
        \${GOLLEK_JAVA_OPTS:-} \\
        -Dgollek.gguf.fast_run.daemon="\${GOLLEK_GGUF_FAST_DAEMON}" \\
        -Dgollek.gguf.fast_run.daemon_launcher="\${GOLLEK_GGUF_FAST_DAEMON_LAUNCHER}" \\
        -Dgollek.gguf.fast_run.debug="\${GOLLEK_GGUF_FAST_RUN_DEBUG:-false}" \\
        -Dgollek.gguf.fast_run.hard_exit_after_run="\${GOLLEK_GGUF_FAST_HARD_EXIT_AFTER_RUN}" \\
        -Dgollek.metal.dylib="\$HOME/.gollek/libs/libgollek_metal.dylib" \\
        -Djava.library.path="\$HOME/.gollek/libs:\$HOME/.gollek/libs/llama:\$HOME/.gollek/libs/onnxruntime:\$HOME/.gollek/libs/libtorch" \\
        -cp "$JAR_PATH" tech.kayys.gollek.cli.commands.GgufFastRun "\$@" 2> >(grep -v -e "WARNING: Using incubator modules" -e "^Picked up JAVA_TOOL_OPTIONS:" >&2)
    FAST_STATUS=\$?
    if [ "\$FAST_STATUS" -eq 0 ]; then
        exit 0
    fi
    if [ "\$FAST_STATUS" -ne 42 ]; then
        exit "\$FAST_STATUS"
    fi
    if [ "\${GOLLEK_GGUF_FAST_RUN_DEBUG:-false}" = "true" ]; then
        echo "GGUF fast path returned \$FAST_STATUS; falling back to remaining CLI paths." >&2
    fi
fi

if [ "\${GOLLEK_LITERT_FAST_RUN:-true}" != "false" ] && gollek_litert_fast_command "\${1:-}" && ! gollek_safetensor_requested "\$@" && ! gollek_gguf_requested "\$@"; then
    java \\
        -Xmx"\${GOLLEK_JAVA_HEAP}" \\
        -XX:MaxDirectMemorySize="\${GOLLEK_MAX_DIRECT_MEMORY}" \\
        --enable-preview \\
        --add-modules jdk.incubator.vector \\
        --enable-native-access=ALL-UNNAMED \\
        \${GOLLEK_JAVA_OPTS:-} \\
        -Dgollek.litert.fast_run.daemon="\${GOLLEK_LITERT_FAST_DAEMON}" \\
        -Dgollek.litert.fast_run.daemon_launcher="\${GOLLEK_LITERT_FAST_DAEMON_LAUNCHER}" \\
        -Dgollek.litert.fast_run.dynamic_engine_tokens="\${GOLLEK_LITERT_FAST_DYNAMIC_ENGINE_TOKENS}" \\
        -Dgollek.litert.fast_run.debug="\${GOLLEK_LITERT_FAST_RUN_DEBUG:-false}" \\
        -Dgollek.metal.dylib="\$HOME/.gollek/libs/libgollek_metal.dylib" \\
        -Djava.library.path="\$HOME/.gollek/libs:\$HOME/.gollek/libs/llama:\$HOME/.gollek/libs/onnxruntime:\$HOME/.gollek/libs/libtorch" \\
        -cp "$JAR_PATH" tech.kayys.gollek.cli.commands.LiteRtLmFastRun "\$@" 2> >(grep -v -e "WARNING: Using incubator modules" -e "^Picked up JAVA_TOOL_OPTIONS:" >&2)
    FAST_STATUS=\$?
    if [ "\$FAST_STATUS" -eq 0 ]; then
        exit 0
    fi
    if [ "\$FAST_STATUS" -ne 42 ]; then
        exit "\$FAST_STATUS"
    fi
    if [ "\${GOLLEK_LITERT_FAST_RUN_DEBUG:-false}" = "true" ]; then
        echo "LiteRT fast path returned \$FAST_STATUS; falling back to full CLI." >&2
    fi
fi

# Launch — enabling FFM and Vector API for pure Java converter
exec java \\
    -Xmx"\${GOLLEK_JAVA_HEAP}" \\
    -XX:MaxDirectMemorySize="\${GOLLEK_MAX_DIRECT_MEMORY}" \\
    --enable-preview \\
    --add-modules jdk.incubator.vector \\
    --enable-native-access=ALL-UNNAMED \\
    \${GOLLEK_JAVA_OPTS:-} \\
    -Dgollek.litert.fast_run.daemon="\${GOLLEK_LITERT_FAST_DAEMON}" \\
    -Dgollek.litert.fast_run.daemon_launcher="\${GOLLEK_LITERT_FAST_DAEMON_LAUNCHER}" \\
    -Dgollek.litert.fast_run.dynamic_engine_tokens="\${GOLLEK_LITERT_FAST_DYNAMIC_ENGINE_TOKENS}" \\
    -Dgollek.litert.fast_run.debug="\${GOLLEK_LITERT_FAST_RUN_DEBUG:-false}" \\
    -Dgollek.metal.dylib="\$HOME/.gollek/libs/libgollek_metal.dylib" \\
    -Djava.library.path="\$HOME/.gollek/libs:\$HOME/.gollek/libs/llama:\$HOME/.gollek/libs/onnxruntime:\$HOME/.gollek/libs/libtorch" \\
    -jar "$JAR_PATH" "\$@" 2> >(grep -v -e "WARNING: Using incubator modules" -e "^Picked up JAVA_TOOL_OPTIONS:" >&2)
EOF
fi

chmod +x "$GOLLEK_CLI_BIN"

prewarm_installed_models
verify_installed_fast_paths

# 5. Final Instructions
echo ""
echo -e "${GREEN}✅ Gollek CLI installed locally to ${GOLLEK_CLI_BIN}${NC}"
echo ""

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo -e "${YELLOW}⚠ ${BIN_DIR} is not in your PATH.${NC}"
    echo -e "Add this to your .zshrc or .bashrc:"
    echo -e "  ${BLUE}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
    echo ""
fi

echo -e "Try running: ${YELLOW}gollek --version${NC}"
echo -e "${BLUE}--------------------------------------------------${NC}"
