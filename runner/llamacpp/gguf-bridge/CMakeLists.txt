# CMakeLists.txt for GGUF Bridge - Enterprise-grade native library
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
option(GGUF_ENABLE_VULKAN "Enable Vulkan support" OFF)
option(GGUF_SANITIZE "Enable address sanitizer" OFF)
option(GGUF_BUILD_STATIC_LLAMA "Build static llama.cpp" OFF)

# Fetch nlohmann/json
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

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
        -march=native
        -mtune=native
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG -flto)
        add_link_options(-flto)
    else()
        add_compile_options(-O0 -g)
    endif()
    
    if(GGUF_SANITIZE)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
endif()

# Find dependencies
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

# llama.cpp integration
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common utils" FORCE)
set(LLAMA_BUILD_GGUF OFF CACHE BOOL "Build llama.cpp gguf tools" FORCE)

# Try multiple locations for llama.cpp
set(LLAMA_SOURCE_DIR "" CACHE PATH "Path to llama.cpp source directory")

# Common search paths
set(LLAMA_SEARCH_PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp"
    "${CMAKE_SOURCE_DIR}/llama.cpp"
    "$ENV{HOME}/.gollek/source/vendor/llama.cpp/llama.cpp"
    "$ENV{HOME}/.gollek/source/vendor/llama.cpp"
    "/opt/llama.cpp"
)

foreach(SEARCH_PATH ${LLAMA_SEARCH_PATHS})
    if(EXISTS "${SEARCH_PATH}/CMakeLists.txt" AND NOT LLAMA_SOURCE_DIR)
        set(LLAMA_SOURCE_DIR "${SEARCH_PATH}")
        message(STATUS "Found llama.cpp at: ${LLAMA_SOURCE_DIR}")
    endif()
endforeach()

if(EXISTS "${LLAMA_SOURCE_DIR}/CMakeLists.txt")
    # Configure llama.cpp with our settings
    set(LLAMA_ACCELERATE ON CACHE BOOL "Enable Accelerate" FORCE)
    set(LLAMA_METAL ${GGUF_ENABLE_METAL} CACHE BOOL "Enable Metal" FORCE)
    set(LLAMA_CUDA ${GGUF_ENABLE_CUDA} CACHE BOOL "Enable CUDA" FORCE)
    set(LLAMA_OPENCL ${GGUF_ENABLE_OPENCL} CACHE BOOL "Enable OpenCL" FORCE)
    set(LLAMA_VULKAN ${GGUF_ENABLE_VULKAN} CACHE BOOL "Enable Vulkan" FORCE)
    
    add_subdirectory("${LLAMA_SOURCE_DIR}" llama.cpp EXCLUDE_FROM_ALL)
    set(LLAMA_FOUND TRUE)
else()
    # Try to find installed llama.cpp via pkg-config
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(LLAMA llama)
        if(LLAMA_FOUND)
            # Ensure include directories are set from pkg-config
            set(LLAMA_SOURCE_DIR_INCLUDE ${LLAMA_INCLUDE_DIRS})
        endif()
    endif()
    
    if(NOT LLAMA_FOUND)
        message(FATAL_ERROR 
            "llama.cpp not found. Please clone it to one of these locations:\n"
            "  ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp\n"
            "  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp\n"
            "  ~/.gollek/source/vendor/llama.cpp\n"
            "Or set LLAMA_SOURCE_DIR to the correct path.")
    endif()
endif()

# CUDA support
if(GGUF_ENABLE_CUDA AND LLAMA_FOUND)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(GGML_USE_CUDA)
endif()

# Metal support
if(GGUF_ENABLE_METAL)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    add_compile_definitions(GGML_USE_METAL)
endif()

# OpenCL support
if(GGUF_ENABLE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        add_compile_definitions(GGML_USE_OPENCL)
    endif()
endif()

# Main library sources
set(GGUF_BRIDGE_SOURCES
    gguf_bridge.cpp
    gguf_converter.cpp
    gguf_quantizer.cpp
    gguf_utils.cpp
)

set(GGUF_BRIDGE_HEADERS
    gguf_bridge.hpp
    gguf_converter.hpp
    gguf_quantizer.hpp
    gguf_utils.hpp
)

# Create the library
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
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(gguf_bridge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LLAMA_SOURCE_DIR}
        ${LLAMA_SOURCE_DIR}/include
        ${LLAMA_SOURCE_DIR}/ggml/include
        ${LLAMA_SOURCE_DIR}/common
        ${LLAMA_INCLUDE_DIRS}
        ${json_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(gguf_bridge
    PRIVATE
        llama
        ggml
        common
        Threads::Threads
        ZLIB::ZLIB
)

if(GGUF_ENABLE_CUDA)
    target_link_libraries(gguf_bridge PRIVATE CUDA::cudart)
endif()

if(GGUF_ENABLE_METAL)
    target_link_libraries(gguf_bridge PRIVATE ${FOUNDATION_LIBRARY} ${METAL_FRAMEWORK})
endif()

if(GGUF_ENABLE_OPENCL AND OpenCL_FOUND)
    target_link_libraries(gguf_bridge PRIVATE OpenCL::OpenCL)
endif()

# Platform-specific settings
if(WIN32)
    target_compile_definitions(gguf_bridge PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _WIN32_WINNT=0x0600
        NOMINMAX
    )
    target_link_libraries(gguf_bridge PRIVATE ws2_32 bcrypt)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(gguf_bridge PRIVATE dl pthread)
endif()

# Export symbols for shared library
if(GGUF_BUILD_SHARED AND WIN32)
    target_compile_definitions(gguf_bridge PRIVATE GGUF_EXPORTS)
endif()

# Tests
if(GGUF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS gguf_bridge
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gguf_bridge
)

# Configuration summary
message(STATUS "")
message(STATUS "GGUF Bridge Configuration:")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build shared:      ${GGUF_BUILD_SHARED}")
message(STATUS "  Build tests:       ${GGUF_BUILD_TESTS}")
message(STATUS "  llama.cpp source:  ${LLAMA_SOURCE_DIR}")
message(STATUS "  CUDA support:      ${GGUF_ENABLE_CUDA}")
message(STATUS "  Metal support:     ${GGUF_ENABLE_METAL}")
message(STATUS "  OpenCL support:    ${GGUF_ENABLE_OPENCL}")
message(STATUS "  Vulkan support:    ${GGUF_ENABLE_VULKAN}")
message(STATUS "  Sanitizer:         ${GGUF_SANITIZE}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")