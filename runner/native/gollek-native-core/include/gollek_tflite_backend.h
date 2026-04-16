/**
 * gollek_tflite_backend.h
 * TensorFlow Lite (LiteRT) backend implementation for Gollek Engine.
 *
 * This backend wraps the TFLite C API and implements the RuntimeBackend
 * interface for TFLite models (.tfl, .litertlm, etc.).
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#pragma once

#include "gollek_engine.h"
#include "gollek_engine_internal.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"

#include <memory>
#include <string>
#include <vector>

namespace gollek {

/**
 * TFLite backend implementation.
 * Wraps TensorFlow Lite C API and implements RuntimeBackend interface.
 */
class TFLiteBackend : public RuntimeBackend {
 public:
  explicit TFLiteBackend(const GollekConfig &config);
  ~TFLiteBackend() override;

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

  // TFLite handles (RAII via unique_ptr + custom deleters)
  struct ModelDeleter {
    void operator()(LitertModel *p) const { LitertModelDelete(p); }
  };
  struct OptionsDeleter {
    void operator()(LitertInterpreterOptions *p) const {
      LitertInterpreterOptionsDelete(p);
    }
  };
  struct InterpDeleter {
    void operator()(LitertInterpreter *p) const { LitertInterpreterDelete(p); }
  };

  std::unique_ptr<LitertModel, ModelDeleter> model_;
  std::unique_ptr<LitertInterpreterOptions, OptionsDeleter> options_;
  std::unique_ptr<LitertInterpreter, InterpDeleter> interpreter_;

  // Optional hardware delegate (raw pointer; lifetime tied to interpreter)
  LitertDelegate *delegate_ = nullptr;

  // Error tracking
  mutable std::string last_error_;

  // Helper methods
  GollekStatus build_interpreter();
  void fill_tensor_info(const LitertTensor *t, GollekTensorInfo *out) const;
  void set_error(const char *msg) const;
};

}  // namespace gollek
