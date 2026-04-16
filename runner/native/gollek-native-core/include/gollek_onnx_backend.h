/**
 * gollek_onnx_backend.h
 * ONNX Runtime backend implementation for Gollek Engine.
 *
 * This backend wraps the ONNX Runtime C API and implements the RuntimeBackend
 * interface for ONNX models (.onnx files).
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#pragma once

#include "gollek_engine.h"
#include "gollek_engine_internal.h"

// Forward declare ONNX Runtime C API types
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtRunOptions;
struct OrtMemoryInfo;
struct OrtAllocator;
struct OrtValue;
struct OrtApi;
struct OrtApiBase;
struct OrtTensorTypeAndShapeInfo;

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gollek {

/**
 * ONNX Runtime backend implementation.
 * Wraps ONNX Runtime C API and implements RuntimeBackend interface.
 */
class OnnxRuntimeBackend : public RuntimeBackend {
 public:
  explicit OnnxRuntimeBackend(const GollekConfig &config);
  ~OnnxRuntimeBackend() override;

  // RuntimeBackend interface
  GollekStatus load_from_file(const char *path) override;
  GollekStatus load_from_buffer(const void *data, size_t size) override;

  int input_count() const override;
  int output_count() const override;
  GollekStatus input_info(int index, GollekTensorInfo *out) const override;
  GollekStatus output_info(int index, GollekTensorInfo *out) const override;

  GollekStatus resize_input(int index, const int32_t *dims,
                            int num_dims) override;
  GollekStatus allocate_tensors() override;

  GollekStatus set_input(int index, const void *src, size_t bytes) override;
  GollekStatus invoke() override;
  GollekStatus get_output(int index, void *dst, size_t dst_bytes) override;

  GollekStatus set_batch_input(int index, const void *const *inputs,
                               const size_t *input_bytes,
                               int num_inputs) override;
  GollekStatus get_batch_output(int index, void *const *outputs,
                                const size_t *output_bytes,
                                int num_outputs) override;

  const char *last_error() const override { return last_error_.c_str(); }

 private:
  GollekConfig config_;

  // ONNX Runtime opaque handles
  OrtEnv *env_ = nullptr;
  OrtSession *session_ = nullptr;
  OrtSessionOptions *session_opts_ = nullptr;
  OrtRunOptions *run_opts_ = nullptr;
  OrtMemoryInfo *meminfo_ = nullptr;
  OrtAllocator *allocator_ = nullptr;
  const OrtApi *ort_api_ = nullptr;

  // Cached tensor information
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<int, GollekTensorInfo> cached_input_info_;
  std::map<int, GollekTensorInfo> cached_output_info_;
  std::vector<OrtValue *> input_tensors_;
  std::vector<OrtValue *> output_tensors_;

  // Error tracking
  mutable std::string last_error_;

  // Helper methods
  GollekStatus init_onnx_runtime();
  GollekStatus fetch_tensor_names();
  GollekStatus cache_tensor_metadata();
  GollekStatus fill_tensor_info_from_ort(OrtValue *tensor, int tensor_index,
                                         bool is_input,
                                         GollekTensorInfo *out) const;
  
  // Type conversion
  GollekTensorType ort_type_to_gollek(int ort_type) const;
  void set_error(const char *msg) const;
  
  // Execution provider setup
  void setup_execution_providers();
};

}  // namespace gollek
