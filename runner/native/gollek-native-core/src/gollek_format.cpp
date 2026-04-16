/**
 * gollek_format.cpp
 * Model format detection utilities.
 *
 * Detects model format from file extension and optional magic bytes.
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#include "gollek_engine_internal.h"
#include <cctype>
#include <cstdio>
#include <cstring>

namespace gollek {

/**
 * Convert filename suffix to lowercase for case-insensitive comparison.
 */
static std::string to_lower(const char *s) {
  std::string result;
  for (const char *p = s; *p; ++p) {
    result += std::tolower(*p);
  }
  return result;
}

/**
 * Detect model format from file path and optional magic bytes.
 */
ModelFormat detect_format(const char *path) {
  if (!path || path[0] == '\0') {
    return ModelFormat::UNKNOWN;
  }

  // 1. Check file extension
  const char *ext_start = std::strrchr(path, '.');
  if (ext_start) {
    std::string ext = to_lower(ext_start);

    if (ext == ".onnx") {
      return ModelFormat::ONNX;
    }
    if (ext == ".tfl" || ext == ".litertlm" || ext == ".tflite") {
      return ModelFormat::TFLITE;
    }
  }

  // 2. Try to detect by magic bytes
  FILE *fp = std::fopen(path, "rb");
  if (!fp) {
    return ModelFormat::UNKNOWN;
  }

  uint8_t magic[4];
  size_t n = std::fread(magic, 1, sizeof(magic), fp);
  std::fclose(fp);

  if (n >= 4) {
    // TFLite magic: "TFL3" (0x54 0x46 0x4C 0x33)
    if (magic[0] == 0x54 && magic[1] == 0x46 && magic[2] == 0x4C &&
        magic[3] == 0x33) {
      return ModelFormat::TFLITE;
    }
    // ONNX magic: start with protobuf tag 0x0A (field 1, wire type 2)
    // followed by length, then "ONNX" at a specific offset.
    // Simplified check: look for "ONNX" string in first 256 bytes.
  }

  // 3. Check for ONNX magic bytes (more robust)
  fp = std::fopen(path, "rb");
  if (!fp) {
    return ModelFormat::UNKNOWN;
  }

  uint8_t buf[256];
  n = std::fread(buf, 1, sizeof(buf), fp);
  std::fclose(fp);

  // Look for "ONNX" string in the buffer
  for (size_t i = 0; i + 4 <= n; ++i) {
    if (buf[i] == 'O' && buf[i + 1] == 'N' && buf[i + 2] == 'N' &&
        buf[i + 3] == 'X') {
      return ModelFormat::ONNX;
    }
  }

  return ModelFormat::UNKNOWN;
}

}  // namespace gollek
