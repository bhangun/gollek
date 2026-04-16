# ============================================================================
# CMakeLists.txt  —  Gollek Native Core
# Unified build system targeting Android, iOS, Web (Wasm), and Desktop.
#
# Usage examples:
#
#   # Desktop (Linux / macOS / Windows)
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
#   # Android (requires Android NDK r25+)
#   cmake -B build-android \
#     -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a \
#     -DANDROID_PLATFORM=android-26 \
#     -DGOLLEK_PLATFORM=android
#   cmake --build build-android
#
#   # iOS (Xcode required)
#   cmake -B build-ios \
#     -G Xcode \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/ios.toolchain.cmake \
#     -DPLATFORM=OS64 \
#     -DGOLLEK_PLATFORM=ios
#   cmake --build build-ios --config Release
#
#   # WebAssembly (Emscripten required)
#   emcmake cmake -B build-wasm -DGOLLEK_PLATFORM=web
#   cmake --build build-wasm
#
# Copyright (c) 2026 Kayys.tech — MIT License
# ============================================================================

cmake_minimum_required(VERSION 3.22)
project(gollek_native VERSION 1.0.0 LANGUAGES CXX)

# ── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Platform detection ────────────────────────────────────────────────────────
if(NOT DEFINED GOLLEK_PLATFORM)
    if(ANDROID)
        set(GOLLEK_PLATFORM "android")
    elseif(EMSCRIPTEN)
        set(GOLLEK_PLATFORM "web")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(GOLLEK_PLATFORM "ios")
    else()
        set(GOLLEK_PLATFORM "desktop")
    endif()
endif()
message(STATUS "Gollek platform: ${GOLLEK_PLATFORM}")

# ── Options ───────────────────────────────────────────────────────────────────
option(GOLLEK_ENABLE_GPU_DELEGATE "Build with GPU/OpenCL delegate (desktop)" OFF)
option(GOLLEK_ENABLE_ONNX         "Build with ONNX Runtime support"          OFF)
option(GOLLEK_BUILD_TESTS         "Build smoke-test executable"               ON)

# ── TensorFlow Lite dependency ────────────────────────────────────────────────
# We expect LiteRT (TFLite) C API headers and the compiled library to be
# available. Set TFLITE_ROOT or rely on find_package / FetchContent below.
#
# Option A: Pre-built binaries (fastest; see download_litert.sh)
#   -DTFLITE_ROOT=/path/to/litert-install
#
# Option B: Build TFLite from source via FetchContent (slow first build)
#   -DGOLLEK_FETCH_TFLITE=ON

option(GOLLEK_FETCH_TFLITE "Download and build TFLite from source" OFF)

if(GOLLEK_FETCH_TFLITE)
    include(FetchContent)
    FetchContent_Declare(
        tensorflow
        GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
        GIT_TAG        v2.16.1
        GIT_SHALLOW    TRUE
    )
    # Only build the C API target
    set(TFLITE_ENABLE_INSTALL ON  CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(tensorflow)
    set(TFLITE_LIB tensorflowlite_c)
    set(TFLITE_INCLUDE_DIRS ${tensorflow_SOURCE_DIR})
else()
    # Expect pre-built: TFLITE_ROOT should contain include/ and lib/
    if(NOT TFLITE_ROOT)
        message(FATAL_ERROR
            "TFLITE_ROOT not set. Either:\n"
            "  cmake ... -DTFLITE_ROOT=/path/to/litert\n"
            "  cmake ... -DGOLLEK_FETCH_TFLITE=ON\n"
        )
    endif()
    find_library(TFLITE_LIB
        NAMES tensorflowlite_c
        PATHS ${TFLITE_ROOT}/lib
        REQUIRED
    )
    set(TFLITE_INCLUDE_DIRS ${TFLITE_ROOT}/include)
    message(STATUS "Using pre-built TFLite: ${TFLITE_LIB}")
endif()

# ── Core source files ─────────────────────────────────────────────────────────
set(GOLLEK_CORE_SOURCES
    src/gollek_engine.cpp
    src/gollek_format.cpp
    src/gollek_tflite_backend.cpp
)

# Add ONNX backend if enabled
if(GOLLEK_ENABLE_ONNX)
    list(APPEND GOLLEK_CORE_SOURCES src/gollek_onnx_backend.cpp)
endif()

# Delegate source — pick exactly one per platform
if(GOLLEK_PLATFORM STREQUAL "android")
    list(APPEND GOLLEK_CORE_SOURCES platform/delegate_android.cpp)
elseif(GOLLEK_PLATFORM STREQUAL "ios")
    list(APPEND GOLLEK_CORE_SOURCES platform/delegate_apple.mm)
elseif(GOLLEK_PLATFORM STREQUAL "web")
    list(APPEND GOLLEK_CORE_SOURCES platform/delegate_desktop.cpp)
else()
    list(APPEND GOLLEK_CORE_SOURCES platform/delegate_desktop.cpp)
endif()

# ── Main library target ───────────────────────────────────────────────────────
if(GOLLEK_PLATFORM STREQUAL "web")
    # Wasm: produce a single .wasm + .js pair
    add_executable(gollek_core
        ${GOLLEK_CORE_SOURCES}
        platform/web/gollek_wasm_bridge.cpp
    )
    target_link_options(gollek_core PRIVATE
        "SHELL:-s MODULARIZE=1"
        "SHELL:-s EXPORT_ES6=1"
        "SHELL:-s ENVIRONMENT=web,node"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        "SHELL:-s INITIAL_MEMORY=33554432"   # 32 MB initial heap
        "SHELL:-s MAXIMUM_MEMORY=536870912"  # 512 MB max
        "--bind"                              # Enable embind
        "SHELL:-s SINGLE_FILE=0"             # Separate .js and .wasm
    )
    set_target_properties(gollek_core PROPERTIES SUFFIX ".js")
else()
    # Native: shared library (.so / .dylib / .dll)
    add_library(gollek_core SHARED ${GOLLEK_CORE_SOURCES})
endif()

target_include_directories(gollek_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TFLITE_INCLUDE_DIRS}
)

target_link_libraries(gollek_core PUBLIC ${TFLITE_LIB})

# ── ONNX Runtime dependency (optional) ──────────────────────────────────────
if(GOLLEK_ENABLE_ONNX)
    message(STATUS "Building with ONNX Runtime support")
    
    # Try to find ONNX Runtime
    # Option A: Pre-built binaries
    #   -DONNXRUNTIME_ROOT=/path/to/onnxruntime-install
    # Option B: System-installed (find_package)
    
    if(ONNXRUNTIME_ROOT)
        message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOT}")
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime
            PATHS ${ONNXRUNTIME_ROOT}/lib
            REQUIRED
        )
        target_include_directories(gollek_core PRIVATE 
            ${ONNXRUNTIME_ROOT}/include
        )
        target_link_libraries(gollek_core PRIVATE ${ONNXRUNTIME_LIB})
    else()
        # Try system paths
        find_package(ONNXRUNTIME CONFIG QUIET)
        if(ONNXRUNTIME_FOUND)
            message(STATUS "Found system ONNX Runtime")
            target_link_libraries(gollek_core PRIVATE ONNX::onnxruntime)
        else()
            # Try find_library as last resort
            find_library(ONNXRUNTIME_LIB onnxruntime)
            if(ONNXRUNTIME_LIB)
                message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
                target_link_libraries(gollek_core PRIVATE ${ONNXRUNTIME_LIB})
            else()
                message(FATAL_ERROR 
                    "ONNX Runtime not found. Either:\n"
                    "  cmake ... -DONNXRUNTIME_ROOT=/path/to/onnxruntime\n"
                    "  Install ONNX Runtime system-wide\n"
                    "  Or disable ONNX: -DGOLLEK_ENABLE_ONNX=OFF\n"
                )
            endif()
        endif()
    endif()
    
    target_compile_definitions(gollek_core PRIVATE GOLLEK_ENABLE_ONNX)
endif()

# ── Platform-specific additions ───────────────────────────────────────────────

if(GOLLEK_PLATFORM STREQUAL "android")
    # JNI bridge
    target_sources(gollek_core PRIVATE platform/android/gollek_jni.cpp)
    # Android system libraries
    target_link_libraries(gollek_core PRIVATE android log)
    # Export only the JNI and C API symbols
    set_target_properties(gollek_core PROPERTIES
        ANDROID_EXPORT_SYMBOLS   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gollek.map"
    )

elseif(GOLLEK_PLATFORM STREQUAL "ios")
    # Objective-C++ bridge (.mm)
    target_sources(gollek_core PRIVATE
        platform/ios/GollekEngine.mm
    )
    # Metal + CoreML frameworks
    target_link_libraries(gollek_core PRIVATE
        "-framework Metal"
        "-framework CoreML"
        "-framework Foundation"
    )
    # Build a fat static library for XCFramework packaging
    set_target_properties(gollek_core PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )

elseif(GOLLEK_PLATFORM STREQUAL "desktop")
    if(GOLLEK_ENABLE_GPU_DELEGATE)
        target_compile_definitions(gollek_core PRIVATE GOLLEK_ENABLE_GPU_DELEGATE)
    endif()
    # Desktop JNI (optional; requires JDK headers)
    find_package(JNI QUIET)
    if(JNI_FOUND)
        target_sources(gollek_core PRIVATE platform/android/gollek_jni.cpp)
        target_sources(gollek_core PRIVATE platform/desktop/gollek_desktop_runner.cpp)
        target_compile_definitions(gollek_core PRIVATE GOLLEK_ENABLE_DESKTOP_JNI)
        target_include_directories(gollek_core PRIVATE ${JNI_INCLUDE_DIRS})
        target_link_libraries(gollek_core PRIVATE ${JNI_LIBRARIES})
        message(STATUS "JNI found — desktop JVM bridge enabled")
    endif()
endif()

# ── Smoke-test executable ─────────────────────────────────────────────────────
if(GOLLEK_BUILD_TESTS AND NOT GOLLEK_PLATFORM STREQUAL "web"
                      AND NOT GOLLEK_PLATFORM STREQUAL "ios")
    add_executable(gollek_test
        platform/desktop/gollek_desktop_runner.cpp
        tests/smoke_test_main.cpp
    )
    target_include_directories(gollek_test PRIVATE include)
    target_link_libraries(gollek_test PRIVATE gollek_core)
    target_compile_definitions(gollek_test PRIVATE GOLLEK_SMOKE_TEST_MAIN)
endif()

# ── Install targets ───────────────────────────────────────────────────────────
install(TARGETS gollek_core
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib
    RUNTIME  DESTINATION bin
)
install(FILES include/gollek_engine.h DESTINATION include)
