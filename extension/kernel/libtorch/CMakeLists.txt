cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pytorch-quarkus-binding)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find LibTorch
# Set CMAKE_PREFIX_PATH to libtorch location:
# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create shared library
add_library(libtorch_wrapper SHARED
    src/main/cpp/libtorch_wrapper.cpp
)

# Link LibTorch
target_link_libraries(libtorch_wrapper "${TORCH_LIBRARIES}")
target_include_directories(libtorch_wrapper PRIVATE ${TORCH_INCLUDE_DIRS})

# Set output directory
set_target_properties(libtorch_wrapper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Copy library to resources directory
add_custom_command(TARGET libtorch_wrapper POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:libtorch_wrapper>
        ${CMAKE_SOURCE_DIR}/src/main/resources/native/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/
    COMMENT "Copying library to resources directory"
)

# Installation rules
install(TARGETS libtorch_wrapper
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Print configuration
message(STATUS "PyTorch version: ${Torch_VERSION}")
message(STATUS "PyTorch CXX flags: ${TORCH_CXX_FLAGS}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
