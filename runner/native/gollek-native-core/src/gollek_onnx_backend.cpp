/**
 * gollek_onnx_backend.cpp
 * ONNX Runtime backend implementation for Gollek Engine.
 *
 * Wraps ONNX Runtime C API and implements the RuntimeBackend interface.
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#include "gollek_onnx_backend.h"
#include "onnxruntime_c_api.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace gollek {

/* ══════════════════════════════════════════════════════════════════════════
 * OnnxRuntimeBackend — Construction / Destruction
 * ══════════════════════════════════════════════════════════════════════════ */

OnnxRuntimeBackend::OnnxRuntimeBackend(const GollekConfig &config)
    : config_(config) {
  init_onnx_runtime();
}

OnnxRuntimeBackend::~OnnxRuntimeBackend() {
  // Release output tensors
  if (ort_api_) {
    for (auto *tensor : output_tensors_) {
      if (tensor) {
        ort_api_->ReleaseValue(tensor);
      }
    }
    for (auto *tensor : input_tensors_) {
      if (tensor) {
        ort_api_->ReleaseValue(tensor);
      }
    }
  }

  // Release session and options
  if (ort_api_) {
    if (session_) {
      ort_api_->ReleaseSession(session_);
    }
    if (session_opts_) {
      ort_api_->ReleaseSessionOptions(session_opts_);
    }
    if (run_opts_) {
      ort_api_->ReleaseRunOptions(run_opts_);
    }
    if (meminfo_) {
      ort_api_->ReleaseMemoryInfo(meminfo_);
    }
  }

  // Release environment (typically a singleton)
  if (env_ && ort_api_) {
    ort_api_->ReleaseEnv(env_);
  }
}

/* ══════════════════════════════════════════════════════════════════════════
 * OnnxRuntimeBackend — Initialization
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus OnnxRuntimeBackend::init_onnx_runtime() {
  // Get ONNX Runtime API
  const OrtApiBase *api_base = OrtGetApiBase();
  if (!api_base) {
    set_error("OrtGetApiBase failed");
    return GOLLEK_ERROR;
  }

  // Get the current API version
  ort_api_ = api_base->GetApi(ORT_API_VERSION);
  if (!ort_api_) {
    set_error("GetApi failed");
    return GOLLEK_ERROR;
  }

  // Create environment
  OrtStatus *status = ort_api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                          "gollek_engine", &env_);
  if (status) {
    set_error(ort_api_->GetErrorMessage(status));
    ort_api_->ReleaseStatus(status);
    return GOLLEK_ERROR;
  }

  // Create session options
  status = ort_api_->CreateSessionOptions(&session_opts_);
  if (status) {
    set_error(ort_api_->GetErrorMessage(status));
    ort_api_->ReleaseStatus(status);
    return GOLLEK_ERROR;
  }

  // Set thread options
  int num_threads = (config_.num_threads <= 0) ? 4 : config_.num_threads;
  status = ort_api_->SetIntraOpNumThreads(session_opts_, num_threads);
  if (status) {
    ort_api_->ReleaseStatus(status);
    // Non-fatal
  }

  status = ort_api_->SetInterOpNumThreads(session_opts_, num_threads);
  if (status) {
    ort_api_->ReleaseStatus(status);
    // Non-fatal
  }

  // Set graph optimization level
  status = ort_api_->SetSessionGraphOptimizationLevel(
      session_opts_, GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (status) {
    ort_api_->ReleaseStatus(status);
    // Non-fatal
  }

  // Setup execution providers (GPU, CoreML, etc.)
  setup_execution_providers();

  // Create run options
  status = ort_api_->CreateRunOptions(&run_opts_);
  if (status) {
    set_error(ort_api_->GetErrorMessage(status));
    ort_api_->ReleaseStatus(status);
    return GOLLEK_ERROR;
  }

  // Create memory info (CPU for now; can extend for GPU)
  status = ort_api_->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault, &meminfo_);
  if (status) {
    set_error(ort_api_->GetErrorMessage(status));
    ort_api_->ReleaseStatus(status);
    return GOLLEK_ERROR;
  }

  return GOLLEK_OK;
}

void OnnxRuntimeBackend::setup_execution_providers() {
  if (!ort_api_ || !session_opts_) {
    return;
  }

  // Try to add CUDA provider if available
  if (config_.delegate == GOLLEK_DELEGATE_GPU ||
      config_.delegate == GOLLEK_DELEGATE_AUTO) {
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    OrtStatus *status =
        ort_api_->SessionOptionsAppendExecutionProvider_CUDA(session_opts_,
                                                             &cuda_options);
    if (!status) {
      return;  // Success; don't fall through to CPU
    }
    ort_api_->ReleaseStatus(status);
  }

  // Try CoreML for Apple platforms
  if (config_.delegate == GOLLEK_DELEGATE_COREML ||
      config_.delegate == GOLLEK_DELEGATE_AUTO) {
    OrtStatus *status =
        ort_api_->SessionOptionsAppendExecutionProvider_CoreML(
            session_opts_, nullptr);
    if (!status) {
      return;
    }
    ort_api_->ReleaseStatus(status);
  }

  // CPU provider is always added as fallback
  // (ONNX Runtime adds it automatically if no other providers succeed)
}

/* ══════════════════════════════════════════════════════════════════════════
 * OnnxRuntimeBackend — Model Loading
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus OnnxRuntimeBackend::load_from_file(const char *path) {
  if (!path || path[0] == '\0') {
    set_error("load_from_file: path is null or empty");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  if (!ort_api_ || !env_ || !session_opts_) {
    set_error("ONNX Runtime not properly initialized");
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }

  OrtStatus *status =
      ort_api_->CreateSession(env_, path, session_opts_, &session_);
  if (status) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "CreateSession failed: %s",
                  ort_api_->GetErrorMessage(status));
    set_error(buf);
    ort_api_->ReleaseStatus(status);
    return GOLLEK_ERROR_MODEL_LOAD;
  }

  GollekStatus gs = fetch_tensor_names();
  if (gs != GOLLEK_OK) {
    return gs;
  }

  return cache_tensor_metadata();
}

GollekStatus OnnxRuntimeBackend::load_from_buffer(const void *data,
                                                  size_t size) {
  // ONNX Runtime's C API doesn't directly support loading from buffer.
  // Write to a temporary file or use an alternative approach.
  // For now, return unsupported.
  set_error("load_from_buffer not supported by ONNX Runtime backend");
  return GOLLEK_ERROR;
}

/* ══════════════════════════════════════════════════════════════════════════
 * OnnxRuntimeBackend — Tensor Name & Metadata Fetching
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus OnnxRuntimeBackend::fetch_tensor_names() {
  if (!ort_api_ || !session_) {
    set_error("Session not initialized");
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }

  input_names_.clear();
  output_names_.clear();

  size_t num_inputs = 0;
  OrtStatus *status = ort_api_->SessionGetInputCount(session_, &num_inputs);
  if (status) {
    set_error(ort_api_->GetErrorMessage(status));
    ort_api_->ReleaseStatus(status);
    return GOLLEK_ERROR;
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    char *input_name = nullptr;
    status = ort_api_->SessionGetInputName(session_, i, allocator_,
                                           &input_name);
    if (status) {
      ort_api_->ReleaseStatus(status);
      set_error("Failed to get input name");
      return GOLLEK_ERROR;
    }
    input_names_.push_back(std::string(input_name));
  }

  size_t num_outputs = 0;
  status = ort_api_->SessionGetOutputCount(session_, &num_outputs);
  if (status) {
    set_error(ort_api_->GetErrorMessage(status));
    ort_api_->ReleaseStatus(status);
    return GOLLEK_ERROR;
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    char *output_name = nullptr;
    status = ort_api_->SessionGetOutputName(session_, i, allocator_,
                                            &output_name);
    if (status) {
      ort_api_->ReleaseStatus(status);
      set_error("Failed to get output name");
      return GOLLEK_ERROR;
    }
    output_names_.push_back(std::string(output_name));
  }

  return GOLLEK_OK;
}

GollekStatus OnnxRuntimeBackend::cache_tensor_metadata() {
  // Tensor metadata will be populated on-demand in input_info/output_info
  return GOLLEK_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * OnnxRuntimeBackend — Tensor Introspection
 * ══════════════════════════════════════════════════════════════════════════ */

int OnnxRuntimeBackend::input_count() const {
  return static_cast<int>(input_names_.size());
}

int OnnxRuntimeBackend::output_count() const {
  return static_cast<int>(output_names_.size());
}

GollekStatus OnnxRuntimeBackend::input_info(int index,
                                            GollekTensorInfo *out) const {
  if (!out || index < 0 || index >= input_count()) {
    return GOLLEK_ERROR_INVALID_ARG;
  }

  // For now, return basic info; in production, query shape from session
  if (cached_input_info_.count(index)) {
    *out = cached_input_info_.at(index);
    return GOLLEK_OK;
  }

  // Placeholder: return error (should be populated in load)
  return GOLLEK_ERROR;
}

GollekStatus OnnxRuntimeBackend::output_info(int index,
                                             GollekTensorInfo *out) const {
  if (!out || index < 0 || index >= output_count()) {
    return GOLLEK_ERROR_INVALID_ARG;
  }

  if (cached_output_info_.count(index)) {
    *out = cached_output_info_.at(index);
    return GOLLEK_OK;
  }

  return GOLLEK_ERROR;
}

GollekTensorType OnnxRuntimeBackend::ort_type_to_gollek(int ort_type) const {
  // Map ONNX data types to Gollek types
  // ONNX types: FLOAT=1, UINT8=2, INT8=3, INT16=5, INT32=6, INT64=7,
  //             STRING=8, BOOL=9, FLOAT16=10, ...
  switch (ort_type) {
    case 1:   // FLOAT
      return GOLLEK_TYPE_FLOAT32;
    case 10:  // FLOAT16
      return GOLLEK_TYPE_FLOAT16;
    case 3:   // INT8
      return GOLLEK_TYPE_INT8;
    case 2:   // UINT8
      return GOLLEK_TYPE_UINT8;
    case 6:   // INT32
      return GOLLEK_TYPE_INT32;
    case 7:   // INT64
      return GOLLEK_TYPE_INT64;
    case 9:   // BOOL
      return GOLLEK_TYPE_BOOL;
    default:
      return GOLLEK_TYPE_FLOAT32;  // Default
  }
}

/* ══════════════════════════════════════════════════════════════════════════
 * OnnxRuntimeBackend — Dynamic Shape Support
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus OnnxRuntimeBackend::resize_input(int index, const int32_t *dims,
                                              int num_dims) {
  // ONNX Runtime doesn't have a pre-allocate step; shapes are handled at Run
  set_error("resize_input not supported by ONNX Runtime backend");
  return GOLLEK_ERROR;
}

GollekStatus OnnxRuntimeBackend::allocate_tensors() {
  // ONNX Runtime handles allocation internally
  return GOLLEK_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * OnnxRuntimeBackend — Inference
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus OnnxRuntimeBackend::set_input(int index, const void *src,
                                           size_t bytes) {
  if (!src || index < 0 || index >= input_count()) {
    set_error("Invalid input parameters");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  // Create tensor from input data
  // (In a full implementation, we'd cache and reuse tensors)
  return GOLLEK_OK;
}

GollekStatus OnnxRuntimeBackend::invoke() {
  if (!ort_api_ || !session_) {
    set_error("Session not initialized");
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }

  // Placeholder: actual Run call would go here
  return GOLLEK_OK;
}

GollekStatus OnnxRuntimeBackend::get_output(int index, void *dst,
                                            size_t dst_bytes) {
  if (!dst || index < 0 || index >= output_count()) {
    set_error("Invalid output parameters");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  // Extract output tensor data
  // (In a full implementation, query the output tensor)
  return GOLLEK_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * OnnxRuntimeBackend — Batched Inference
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus OnnxRuntimeBackend::set_batch_input(int index,
                                                 const void *const *inputs,
                                                 const size_t *input_bytes,
                                                 int num_inputs) {
  set_error("Batch inference not yet fully implemented for ONNX");
  return GOLLEK_ERROR;
}

GollekStatus OnnxRuntimeBackend::get_batch_output(int index, void *const *outputs,
                                                  const size_t *output_bytes,
                                                  int num_outputs) {
  set_error("Batch inference not yet fully implemented for ONNX");
  return GOLLEK_ERROR;
}

/* ══════════════════════════════════════════════════════════════════════════
 * OnnxRuntimeBackend — Helper Methods
 * ══════════════════════════════════════════════════════════════════════════ */

void OnnxRuntimeBackend::set_error(const char *msg) const {
  if (msg) {
    last_error_ = msg;
  }
}

}  // namespace gollek
