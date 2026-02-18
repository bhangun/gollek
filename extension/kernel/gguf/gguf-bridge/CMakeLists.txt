# CMakeLists.txt for GGUF Bridge
# Enterprise-grade build configuration for cross-platform native library

cmake_minimum_required(VERSION 3.20)
project(gguf_bridge 
    VERSION 1.0.0
    LANGUAGES C CXX
    DESCRIPTION "GGUF Conversion Bridge for Enterprise Inference Engine"
)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(GGUF_BUILD_SHARED "Build shared library" ON)
option(GGUF_BUILD_TESTS "Build tests" ON)
option(GGUF_ENABLE_CUDA "Enable CUDA support" OFF)
option(GGUF_ENABLE_METAL "Enable Metal support" OFF)
option(GGUF_ENABLE_OPENCL "Enable OpenCL support" OFF)
option(GGUF_SANITIZE "Enable address sanitizer" OFF)

# Platform detection
if(APPLE)
    set(GGUF_ENABLE_METAL ON)
elseif(UNIX AND NOT APPLE)
    set(GGUF_ENABLE_OPENCL ON)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -fPIC
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
    else()
        add_compile_options(-O0 -g)
    endif()
    
    if(GGUF_SANITIZE)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
elseif(MSVC)
    add_compile_options(
        /W4
        /MP
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /DNDEBUG)
    else()
        add_compile_options(/Od /Zi)
    endif()
endif()

# Find dependencies
find_package(Threads REQUIRED)

# llama.cpp integration
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)

# Check if llama.cpp is available as subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    add_subdirectory(llama.cpp)
    set(LLAMA_FOUND TRUE)
# Check vendor directory (llama.cpp moved to Inference-gollek-vendor/llama-cpp/llama-cpp)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../Inference-gollek-vendor/llama-cpp/llama-cpp/CMakeLists.txt")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../Inference-gollek-vendor/llama-cpp/llama-cpp llama.cpp)
    set(LLAMA_FOUND TRUE)
else()
    # Try to find installed llama.cpp
    find_package(llama CONFIG)
    if(llama_FOUND)
        set(LLAMA_FOUND TRUE)
    else()
        message(FATAL_ERROR "llama.cpp not found. Please clone it as a subdirectory or install it.")
    endif()
endif()

# CUDA support
if(GGUF_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(GGML_USE_CUDA)
endif()

# Metal support (macOS)
if(GGUF_ENABLE_METAL)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    add_compile_definitions(GGML_USE_METAL)
endif()

# OpenCL support
if(GGUF_ENABLE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        add_compile_definitions(GGML_USE_OPENCL)
    else()
        set(GGUF_ENABLE_OPENCL OFF)
        message(WARNING "OpenCL not found, disabling OpenCL support")
    endif()
endif()

# Main library
set(GGUF_BRIDGE_SOURCES
    gguf_bridge.cpp
)

set(GGUF_BRIDGE_HEADERS
    gguf_bridge.hpp
)

if(GGUF_BUILD_SHARED)
    add_library(gguf_bridge SHARED ${GGUF_BRIDGE_SOURCES})
else()
    add_library(gguf_bridge STATIC ${GGUF_BRIDGE_SOURCES})
endif()

# Set library properties
set_target_properties(gguf_bridge PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${GGUF_BRIDGE_HEADERS}"
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
)

# Include directories
target_include_directories(gguf_bridge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../Inference-gollek-vendor/llama-cpp/llama-cpp
)

# Link libraries
target_link_libraries(gguf_bridge
    PRIVATE
        llama
        Threads::Threads
)

if(GGUF_ENABLE_CUDA)
    target_link_libraries(gguf_bridge PRIVATE CUDA::cudart)
endif()

if(GGUF_ENABLE_METAL)
    target_link_libraries(gguf_bridge PRIVATE
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
    )
endif()

if(GGUF_ENABLE_OPENCL AND OpenCL_FOUND)
    target_link_libraries(gguf_bridge PRIVATE OpenCL::OpenCL)
endif()

# Platform-specific settings
if(WIN32)
    target_compile_definitions(gguf_bridge PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _WIN32_WINNT=0x0600
    )
    target_link_libraries(gguf_bridge PRIVATE ws2_32)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(gguf_bridge PRIVATE dl)
endif()

# Export symbols for shared library
if(GGUF_BUILD_SHARED)
    if(WIN32)
        target_compile_definitions(gguf_bridge PRIVATE GGUF_EXPORTS)
    endif()
endif()

# Tests
if(GGUF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS gguf_bridge
    EXPORT gguf_bridge-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gguf_bridge
)

install(EXPORT gguf_bridge-targets
    FILE gguf_bridge-targets.cmake
    NAMESPACE gguf_bridge::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gguf_bridge
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gguf_bridge-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gguf_bridge-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gguf_bridge
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gguf_bridge-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gguf_bridge-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gguf_bridge-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gguf_bridge
)

# Print configuration summary
message(STATUS "")
message(STATUS "GGUF Bridge Configuration:")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build shared:      ${GGUF_BUILD_SHARED}")
message(STATUS "  Build tests:       ${GGUF_BUILD_TESTS}")
message(STATUS "  CUDA support:      ${GGUF_ENABLE_CUDA}")
message(STATUS "  Metal support:     ${GGUF_ENABLE_METAL}")
message(STATUS "  OpenCL support:    ${GGUF_ENABLE_OPENCL}")
message(STATUS "  Sanitizer:         ${GGUF_SANITIZE}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
