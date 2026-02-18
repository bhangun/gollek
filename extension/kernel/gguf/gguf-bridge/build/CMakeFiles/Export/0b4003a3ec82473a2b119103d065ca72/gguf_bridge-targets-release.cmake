#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gguf_bridge::gguf_bridge" for configuration "Release"
set_property(TARGET gguf_bridge::gguf_bridge APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gguf_bridge::gguf_bridge PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgguf_bridge.1.0.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libgguf_bridge.1.dylib"
  )

list(APPEND _cmake_import_check_targets gguf_bridge::gguf_bridge )
list(APPEND _cmake_import_check_files_for_gguf_bridge::gguf_bridge "${_IMPORT_PREFIX}/lib/libgguf_bridge.1.0.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
