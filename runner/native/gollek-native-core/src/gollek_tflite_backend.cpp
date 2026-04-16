/**
 * gollek_tflite_backend.cpp
 * TensorFlow Lite (LiteRT) backend implementation for Gollek Engine.
 *
 * Wraps TFLite C API and implements the RuntimeBackend interface.
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#include "gollek_tflite_backend.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace gollek {

/* ══════════════════════════════════════════════════════════════════════════
 * TFLiteBackend — Construction / Destruction
 * ══════════════════════════════════════════════════════════════════════════ */

TFLiteBackend::TFLiteBackend(const GollekConfig &config) : config_(config) {}

TFLiteBackend::~TFLiteBackend() {
  // interpreter_ must be destroyed before options_ and model_
  interpreter_.reset();

  // Delegate lifetime is managed by the interpreter in newer TFLite; but
  // if we added it before creating the interpreter we own it.
  // The delegate destroy function is platform-specific (see platform/*.cpp).
  // For safety, we zero the pointer — actual destruction is in
  // create_delegate().
  delegate_ = nullptr;

  options_.reset();
  model_.reset();
}

/* ══════════════════════════════════════════════════════════════════════════
 * TFLiteBackend — Model Loading
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus TFLiteBackend::load_from_file(const char *path) {
  if (!path || path[0] == '\0') {
    set_error("load_from_file: path is null or empty");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  LitertModel *raw = LitertModelCreateFromFile(path);
  if (!raw) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "LitertModelCreateFromFile failed: %s",
                  path);
    set_error(buf);
    return GOLLEK_ERROR_MODEL_LOAD;
  }
  model_.reset(raw);
  return build_interpreter();
}

GollekStatus TFLiteBackend::load_from_buffer(const void *data, size_t size) {
  if (!data || size == 0) {
    set_error("load_from_buffer: data is null or size is 0");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  LitertModel *raw = LitertModelCreate(data, size);
  if (!raw) {
    set_error("LitertModelCreate failed (invalid flatbuffer?)");
    return GOLLEK_ERROR_MODEL_LOAD;
  }
  model_.reset(raw);
  return build_interpreter();
}

/* ══════════════════════════════════════════════════════════════════════════
 * TFLiteBackend — Interpreter Construction (Private)
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus TFLiteBackend::build_interpreter() {
  // 1. Options
  LitertInterpreterOptions *opts = LitertInterpreterOptionsCreate();
  if (!opts) {
    set_error("LitertInterpreterOptionsCreate failed");
    return GOLLEK_ERROR;
  }
  options_.reset(opts);

  int threads = (config_.num_threads <= 0) ? 4 : config_.num_threads;
  LitertInterpreterOptionsSetNumThreads(options_.get(), threads);

  // 2. Delegate auto-selection
  if (config_.delegate != GOLLEK_DELEGATE_NONE) {
    std::string delegate_error;
    delegate_ = create_delegate(config_.delegate, delegate_error);
    if (delegate_) {
      LitertInterpreterOptionsAddDelegate(options_.get(), delegate_);
    } else if (!delegate_error.empty()) {
      // Non-fatal: log and fall back to CPU
      (void)delegate_error;
    }
  }

  // 3. XNNPACK (always beneficial for CPU path)
  if (config_.enable_xnnpack && !delegate_) {
    // XNNPACK is enabled by default in TFLite >= 2.7 when linked.
    // Explicitly request it via options for older builds.
    LitertInterpreterOptionsSetUseNNAPI(options_.get(), 0); // ensure NNAPI off
  }

  // 4. Create interpreter
  LitertInterpreter *interp =
      LitertInterpreterCreate(model_.get(), options_.get());
  if (!interp) {
    set_error("LitertInterpreterCreate failed");
    return GOLLEK_ERROR;
  }
  interpreter_.reset(interp);

  // 5. Allocate tensors
  return allocate_tensors();
}

/* ══════════════════════════════════════════════════════════════════════════
 * TFLiteBackend — Tensor Introspection
 * ══════════════════════════════════════════════════════════════════════════ */

int TFLiteBackend::input_count() const {
  if (!interpreter_)
    return 0;
  return LitertInterpreterGetInputTensorCount(interpreter_.get());
}

int TFLiteBackend::output_count() const {
  if (!interpreter_)
    return 0;
  return LitertInterpreterGetOutputTensorCount(interpreter_.get());
}

void TFLiteBackend::fill_tensor_info(const LitertTensor *t,
                                     GollekTensorInfo *out) const {
  if (!t || !out)
    return;
  out->name = LitertTensorName(t);
  out->type = static_cast<GollekTensorType>(LitertTensorType(t));
  out->byte_size = LitertTensorByteSize(t);
  out->num_dims = LitertTensorNumDims(t);

  for (int i = 0; i < out->num_dims && i < GOLLEK_MAX_DIMS; ++i) {
    out->dims[i] = LitertTensorDim(t, i);
  }

  // Quantization parameters
  auto q_params = LitertTensorQuantizationParams(t);
  out->scale = q_params.scale;
  out->zero_point = q_params.zero_point;
}

GollekStatus TFLiteBackend::input_info(int index, GollekTensorInfo *out) const {
  if (!interpreter_ || !out) {
    return GOLLEK_ERROR_INVALID_ARG;
  }
  if (index < 0 || index >= input_count()) {
    set_error("Input index out of range");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  const LitertTensor *t = LitertInterpreterGetInputTensor(interpreter_.get(), index);
  if (!t) {
    set_error("Failed to get input tensor");
    return GOLLEK_ERROR;
  }

  fill_tensor_info(t, out);
  return GOLLEK_OK;
}

GollekStatus TFLiteBackend::output_info(int index, GollekTensorInfo *out) const {
  if (!interpreter_ || !out) {
    return GOLLEK_ERROR_INVALID_ARG;
  }
  if (index < 0 || index >= output_count()) {
    set_error("Output index out of range");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  const LitertTensor *t = LitertInterpreterGetOutputTensor(interpreter_.get(), index);
  if (!t) {
    set_error("Failed to get output tensor");
    return GOLLEK_ERROR;
  }

  fill_tensor_info(t, out);
  return GOLLEK_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * TFLiteBackend — Dynamic Shape Support
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus TFLiteBackend::resize_input(int index, const int32_t *dims,
                                         int num_dims) {
  if (!interpreter_) {
    set_error("Interpreter not initialized");
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  if (index < 0 || index >= input_count()) {
    set_error("Input index out of range");
    return GOLLEK_ERROR_INVALID_ARG;
  }
  if (!dims || num_dims <= 0 || num_dims > GOLLEK_MAX_DIMS) {
    set_error("Invalid dims or num_dims");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  LitertStatus s = LitertInterpreterResizeInputTensor(interpreter_.get(), index,
                                                      dims, num_dims);
  if (s != LITERT_OK) {
    set_error("LitertInterpreterResizeInputTensor failed");
    return GOLLEK_ERROR;
  }
  return GOLLEK_OK;
}

GollekStatus TFLiteBackend::allocate_tensors() {
  if (!interpreter_) {
    set_error("Interpreter not initialized");
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }

  LitertStatus s = LitertInterpreterAllocateTensors(interpreter_.get());
  if (s != LITERT_OK) {
    set_error("LitertInterpreterAllocateTensors failed");
    return GOLLEK_ERROR_ALLOC_TENSORS;
  }

  return GOLLEK_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * TFLiteBackend — Inference
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus TFLiteBackend::set_input(int index, const void *src,
                                      size_t bytes) {
  if (!interpreter_) {
    set_error("Interpreter not initialized");
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  if (index < 0 || index >= input_count()) {
    set_error("Input index out of range");
    return GOLLEK_ERROR_INVALID_ARG;
  }
  if (!src) {
    set_error("src pointer is null");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  LitertTensor *tensor = LitertInterpreterGetInputTensor(interpreter_.get(), index);
  if (!tensor) {
    set_error("Failed to get input tensor");
    return GOLLEK_ERROR;
  }

  void *tensor_data = LitertTensorData(tensor);
  size_t tensor_bytes = LitertTensorByteSize(tensor);

  if (bytes != tensor_bytes) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "Input size mismatch: got %zu bytes, expected %zu",
                  bytes, tensor_bytes);
    set_error(buf);
    return GOLLEK_ERROR;
  }

  std::memcpy(tensor_data, src, bytes);
  return GOLLEK_OK;
}

GollekStatus TFLiteBackend::invoke() {
  if (!interpreter_) {
    set_error("Interpreter not initialized");
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }

  LitertStatus s = LitertInterpreterInvoke(interpreter_.get());
  if (s != LITERT_OK) {
    set_error("LitertInterpreterInvoke failed");
    return GOLLEK_ERROR_INVOKE;
  }

  return GOLLEK_OK;
}

GollekStatus TFLiteBackend::get_output(int index, void *dst,
                                       size_t dst_bytes) {
  if (!interpreter_) {
    set_error("Interpreter not initialized");
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  if (index < 0 || index >= output_count()) {
    set_error("Output index out of range");
    return GOLLEK_ERROR_INVALID_ARG;
  }
  if (!dst) {
    set_error("dst pointer is null");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  const LitertTensor *tensor = LitertInterpreterGetOutputTensor(interpreter_.get(), index);
  if (!tensor) {
    set_error("Failed to get output tensor");
    return GOLLEK_ERROR;
  }

  const void *tensor_data = LitertTensorData(tensor);
  size_t tensor_bytes = LitertTensorByteSize(tensor);

  if (dst_bytes < tensor_bytes) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "Output buffer too small: need %zu bytes, got %zu",
                  tensor_bytes, dst_bytes);
    set_error(buf);
    return GOLLEK_ERROR;
  }

  std::memcpy(dst, tensor_data, tensor_bytes);
  return GOLLEK_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * TFLiteBackend — Batched Inference (Simple Implementation)
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus TFLiteBackend::set_batch_input(int index,
                                            const void *const *inputs,
                                            const size_t *input_bytes,
                                            int num_inputs) {
  // For simplicity, we concatenate inputs along the batch axis.
  // This is a basic implementation; production code may want KV-cache
  // or other optimizations.

  if (!interpreter_ || !inputs || !input_bytes || num_inputs <= 0) {
    set_error("Invalid batch input parameters");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  LitertTensor *tensor = LitertInterpreterGetInputTensor(interpreter_.get(), index);
  if (!tensor) {
    set_error("Failed to get input tensor");
    return GOLLEK_ERROR;
  }

  void *tensor_data = LitertTensorData(tensor);
  size_t tensor_bytes = LitertTensorByteSize(tensor);

  // Verify total size matches
  size_t total_bytes = 0;
  for (int i = 0; i < num_inputs; ++i) {
    total_bytes += input_bytes[i];
  }

  if (total_bytes != tensor_bytes) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "Batch input size mismatch: got %zu bytes, expected %zu",
                  total_bytes, tensor_bytes);
    set_error(buf);
    return GOLLEK_ERROR;
  }

  // Concatenate all inputs
  size_t offset = 0;
  for (int i = 0; i < num_inputs; ++i) {
    std::memcpy(static_cast<uint8_t *>(tensor_data) + offset,
                inputs[i], input_bytes[i]);
    offset += input_bytes[i];
  }

  return GOLLEK_OK;
}

GollekStatus TFLiteBackend::get_batch_output(int index, void *const *outputs,
                                             const size_t *output_bytes,
                                             int num_outputs) {
  if (!interpreter_ || !outputs || !output_bytes || num_outputs <= 0) {
    set_error("Invalid batch output parameters");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  const LitertTensor *tensor = LitertInterpreterGetOutputTensor(interpreter_.get(), index);
  if (!tensor) {
    set_error("Failed to get output tensor");
    return GOLLEK_ERROR;
  }

  const void *tensor_data = LitertTensorData(tensor);
  size_t tensor_bytes = LitertTensorByteSize(tensor);

  // Verify total size matches
  size_t total_bytes = 0;
  for (int i = 0; i < num_outputs; ++i) {
    total_bytes += output_bytes[i];
  }

  if (total_bytes != tensor_bytes) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "Batch output size mismatch: need %zu bytes, got %zu",
                  tensor_bytes, total_bytes);
    set_error(buf);
    return GOLLEK_ERROR;
  }

  // Split output across buffers
  size_t offset = 0;
  for (int i = 0; i < num_outputs; ++i) {
    std::memcpy(outputs[i], static_cast<const uint8_t *>(tensor_data) + offset,
                output_bytes[i]);
    offset += output_bytes[i];
  }

  return GOLLEK_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * TFLiteBackend — Helper Methods
 * ══════════════════════════════════════════════════════════════════════════ */

void TFLiteBackend::set_error(const char *msg) const {
  if (msg) {
    last_error_ = msg;
  }
}

}  // namespace gollek
