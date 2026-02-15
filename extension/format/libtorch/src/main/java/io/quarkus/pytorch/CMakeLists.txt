cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pytorch_java_ffm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch
list(APPEND CMAKE_PREFIX_PATH "/path/to/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add library
add_library(pytorch_java_ffm SHARED
    pytorch_java_ffm.cpp
)

target_include_directories(pytorch_java_ffm PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(pytorch_java_ffm
    "${TORCH_LIBRARIES}"
)

# Set output directory
set_target_properties(pytorch_java_ffm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Install rules
install(TARGETS pytorch_java_ffm
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
)

install(FILES pytorch_java_ffm.h
    DESTINATION include
)
