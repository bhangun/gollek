cmake_minimum_required(VERSION 3.10)
project(llamacpp_jni)

# Find JNI
find_package(JNI REQUIRED)

# Set source file path
set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/llamacpp_jni.cpp)

# Check if source file exists
if(NOT EXISTS ${SOURCE_FILE})
    message(FATAL_ERROR "Source file not found: ${SOURCE_FILE}")
endif()

# Include directories
include_directories(
    ${JNI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add your source file
add_library(llamacpp_jni SHARED ${SOURCE_FILE})

# Link libraries
target_link_libraries(llamacpp_jni
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libllama.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libggml.a
    pthread
    stdc++
)

# Set output directory
set_target_properties(llamacpp_jni PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../target/native
    OUTPUT_NAME "llamacpp_jni"
)

# Print configuration info
message(STATUS "JNI include directories: ${JNI_INCLUDE_DIRS}")
message(STATUS "Source file: ${SOURCE_FILE}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")