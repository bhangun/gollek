/**
 * gollek_engine.cpp
 * Gollek Inference Engine — C++ Implementation (Refactored with Backend Abstraction)
 *
 * This file provides the implementation of the Engine class using the
 * RuntimeBackend abstraction, supporting both TFLite and ONNX backends.
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#include "gollek_engine_internal.h"
#include "gollek_tflite_backend.h"

#ifdef GOLLEK_ENABLE_ONNX
#include "gollek_onnx_backend.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>

#define GOLLEK_VERSION "1.0.0+litert-2.16+onnx"

namespace gollek {

/* ══════════════════════════════════════════════════════════════════════════
 * MemoryPool
 * ══════════════════════════════════════════════════════════════════════════ */

static constexpr size_t kDefaultPoolBytes = 16 * 1024 * 1024;  // 16 MB

MemoryPool::MemoryPool(size_t capacity_bytes)
    : capacity_(capacity_bytes == 0 ? kDefaultPoolBytes : capacity_bytes) {
  slab_ = static_cast<uint8_t *>(::operator new(capacity_));
}

MemoryPool::~MemoryPool() { ::operator delete(slab_); }

void *MemoryPool::allocate(size_t bytes, size_t alignment) {
  size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
  if (aligned_offset + bytes > capacity_) {
    return nullptr;
  }
  void *ptr = slab_ + aligned_offset;
  offset_ = aligned_offset + bytes;
  return ptr;
}

void MemoryPool::reset() { offset_ = 0; }

/* ══════════════════════════════════════════════════════════════════════════
 * StreamingSession
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus StreamingSession::start(const void *input, size_t input_bytes) {
  std::lock_guard<std::mutex> lock(session_mutex);

  if (!engine || !input || input_bytes == 0) {
    return GOLLEK_ERROR_INVALID_ARG;
  }

  input_buffer.assign(static_cast<const uint8_t *>(input),
                      static_cast<const uint8_t *>(input) + input_bytes);

  GollekStatus status = engine->set_input(0, input, input_bytes);
  if (status != GOLLEK_OK) {
    return status;
  }

  status = engine->invoke();
  if (status != GOLLEK_OK) {
    return status;
  }

  GollekTensorInfo out_info;
  status = engine->output_info(0, &out_info);
  if (status != GOLLEK_OK) {
    return status;
  }

  output_buffer.resize(out_info.byte_size);
  status = engine->get_output(0, output_buffer.data(), output_buffer.size());
  if (status != GOLLEK_OK) {
    return status;
  }

  tokens_generated = 0;
  is_done = false;

  return GOLLEK_OK;
}

GollekStatus StreamingSession::next(void *output, size_t output_bytes,
                                    size_t *actual_bytes, int *done) {
  std::lock_guard<std::mutex> lock(session_mutex);

  if (!output || !actual_bytes || !done) {
    return GOLLEK_ERROR_INVALID_ARG;
  }

  if (is_done) {
    *actual_bytes = 0;
    *done = 1;
    return GOLLEK_OK;
  }

  if (max_tokens > 0 && tokens_generated >= max_tokens) {
    is_done = true;
    *actual_bytes = 0;
    *done = 1;
    return GOLLEK_OK;
  }

  GollekStatus status =
      engine->set_input(0, output_buffer.data(), output_buffer.size());
  if (status != GOLLEK_OK) {
    is_done = true;
    *actual_bytes = 0;
    *done = 1;
    return status;
  }

  status = engine->invoke();
  if (status != GOLLEK_OK) {
    is_done = true;
    *actual_bytes = 0;
    *done = 1;
    return status;
  }

  GollekTensorInfo out_info;
  status = engine->output_info(0, &out_info);
  if (status != GOLLEK_OK) {
    is_done = true;
    *actual_bytes = 0;
    *done = 1;
    return status;
  }

  output_buffer.resize(out_info.byte_size);
  status = engine->get_output(0, output_buffer.data(), output_buffer.size());
  if (status != GOLLEK_OK) {
    is_done = true;
    *actual_bytes = 0;
    *done = 1;
    return status;
  }

  size_t copy_size = std::min(output_bytes, output_buffer.size());
  std::memcpy(output, output_buffer.data(), copy_size);
  *actual_bytes = copy_size;

  tokens_generated++;
  *done = is_done ? 1 : 0;

  return GOLLEK_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Engine — Construction / Destruction
 * ══════════════════════════════════════════════════════════════════════════ */

Engine::Engine(const GollekConfig &config) : config_(config) {
  if (config_.use_memory_pool) {
    pool_ = std::make_unique<MemoryPool>(config_.pool_size_bytes);
  }
  
  latencies_.resize(MAX_LATENCY_SAMPLES);
  
  // Initialize the backend (deferred until load_from_file/load_from_buffer)
}

Engine::~Engine() {
  // backend_ is automatically destroyed via unique_ptr
  backend_.reset();
}

/* ══════════════════════════════════════════════════════════════════════════
 * Engine — Backend Initialization
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus Engine::init_backend() {
  // Determine which backend to use based on config or auto-detect
  ModelFormat format = config_.format;
  
  // If format is unknown, we'll detect it later from the model path
  if (format == ModelFormat::UNKNOWN) {
    // Default to TFLite for backward compatibility
    format = ModelFormat::TFLITE;
  }

  if (format == ModelFormat::ONNX) {
#ifdef GOLLEK_ENABLE_ONNX
    backend_ = std::make_unique<OnnxRuntimeBackend>(config_);
    return GOLLEK_OK;
#else
    set_error("ONNX support not compiled in (use -DGOLLEK_ENABLE_ONNX=ON)");
    return GOLLEK_ERROR;
#endif
  } else {
    // Default to TFLite
    backend_ = std::make_unique<TFLiteBackend>(config_);
    return GOLLEK_OK;
  }
}

/* ══════════════════════════════════════════════════════════════════════════
 * Engine — Model Loading
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus Engine::load_from_file(const char *path) {
  if (!path || path[0] == '\0') {
    set_error("load_from_file: path is null or empty");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  // Auto-detect format if not explicitly set
  if (config_.format == ModelFormat::UNKNOWN) {
    config_.format = detect_format(path);
    if (config_.format == ModelFormat::UNKNOWN) {
      set_error("Could not determine model format from file extension or magic bytes");
      return GOLLEK_ERROR_MODEL_LOAD;
    }
  }

  GollekStatus status = init_backend();
  if (status != GOLLEK_OK) {
    return status;
  }

  status = backend_->load_from_file(path);
  if (status != GOLLEK_OK) {
    set_error(backend_->last_error());
    return status;
  }

  is_ready_ = true;
  return GOLLEK_OK;
}

GollekStatus Engine::load_from_buffer(const void *data, size_t size) {
  if (!data || size == 0) {
    set_error("load_from_buffer: data is null or size is 0");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  if (config_.format == ModelFormat::UNKNOWN) {
    set_error("Format must be explicitly set for load_from_buffer");
    return GOLLEK_ERROR_INVALID_ARG;
  }

  GollekStatus status = init_backend();
  if (status != GOLLEK_OK) {
    return status;
  }

  status = backend_->load_from_buffer(data, size);
  if (status != GOLLEK_OK) {
    set_error(backend_->last_error());
    return status;
  }

  is_ready_ = true;
  return GOLLEK_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Engine — Tensor Introspection
 * ══════════════════════════════════════════════════════════════════════════ */

int Engine::input_count() const {
  if (!backend_) return 0;
  return backend_->input_count();
}

int Engine::output_count() const {
  if (!backend_) return 0;
  return backend_->output_count();
}

GollekStatus Engine::input_info(int index, GollekTensorInfo *out) const {
  if (!backend_) {
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  return backend_->input_info(index, out);
}

GollekStatus Engine::output_info(int index, GollekTensorInfo *out) const {
  if (!backend_) {
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  return backend_->output_info(index, out);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Engine — Dynamic Shape Support
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus Engine::resize_input(int index, const int32_t *dims,
                                  int num_dims) {
  if (!backend_) {
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  return backend_->resize_input(index, dims, num_dims);
}

GollekStatus Engine::allocate_tensors() {
  if (!backend_) {
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  return backend_->allocate_tensors();
}

/* ══════════════════════════════════════════════════════════════════════════
 * Engine — Inference
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus Engine::set_input(int index, const void *src, size_t bytes) {
  if (!backend_) {
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  return backend_->set_input(index, src, bytes);
}

GollekStatus Engine::invoke() {
  if (!backend_) {
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }

  std::lock_guard<std::mutex> lock(invoke_mutex_);
  
  auto start = std::chrono::high_resolution_clock::now();
  GollekStatus status = backend_->invoke();
  auto end = std::chrono::high_resolution_clock::now();
  
  uint64_t latency_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  metrics_.total_inferences++;
  if (status != GOLLEK_OK) {
    metrics_.failed_inferences++;
  } else {
    metrics_.record_latency(latency_us);
    latencies_[latency_index_] = latency_us;
    latency_index_ = (latency_index_ + 1) % MAX_LATENCY_SAMPLES;
  }

  return status;
}

GollekStatus Engine::get_output(int index, void *dst, size_t dst_bytes) {
  if (!backend_) {
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  return backend_->get_output(index, dst, dst_bytes);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Engine — Batched Inference
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus Engine::set_batch_input(int index, const void *const *inputs,
                                     const size_t *input_bytes, int num_inputs) {
  if (!backend_) {
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  return backend_->set_batch_input(index, inputs, input_bytes, num_inputs);
}

GollekStatus Engine::get_batch_output(int index, void *const *outputs,
                                      const size_t *output_bytes, int num_outputs) {
  if (!backend_) {
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }
  return backend_->get_batch_output(index, outputs, output_bytes, num_outputs);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Engine — Streaming
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus Engine::start_streaming(const void *input, size_t input_bytes,
                                     int max_tokens,
                                     GollekStreamSessionHandle *out_session) {
  if (!backend_ || !out_session) {
    return GOLLEK_ERROR_NOT_INITIALIZED;
  }

  auto session = std::make_unique<StreamingSession>(this, max_tokens);
  GollekStatus status = session->start(input, input_bytes);
  if (status != GOLLEK_OK) {
    return status;
  }

  std::lock_guard<std::mutex> lock(streaming_mutex_);
  streaming_sessions_.push_back(std::move(session));
  *out_session = reinterpret_cast<GollekStreamSessionHandle>(
      streaming_sessions_.back().get());

  return GOLLEK_OK;
}

void Engine::end_streaming(GollekStreamSessionHandle session) {
  std::lock_guard<std::mutex> lock(streaming_mutex_);
  streaming_sessions_.erase(
      std::remove_if(streaming_sessions_.begin(), streaming_sessions_.end(),
                     [session](const auto &s) {
                       return s.get() ==
                              reinterpret_cast<StreamingSession *>(session);
                     }),
      streaming_sessions_.end());
}

/* ══════════════════════════════════════════════════════════════════════════
 * Engine — Metrics
 * ══════════════════════════════════════════════════════════════════════════ */

GollekStatus Engine::get_metrics(GollekMetrics *out) const {
  if (!out) {
    return GOLLEK_ERROR_INVALID_ARG;
  }

  out->total_inferences = metrics_.total_inferences.load();
  out->failed_inferences = metrics_.failed_inferences.load();
  out->total_latency_us = metrics_.total_latency_us.load();
  out->avg_latency_us = metrics_.get_avg_latency_us();
  out->p50_latency_us = get_percentile_latency(0.50);
  out->p95_latency_us = get_percentile_latency(0.95);
  out->p99_latency_us = get_percentile_latency(0.99);
  out->peak_memory_bytes = metrics_.peak_memory_bytes.load();
  out->current_memory_bytes = metrics_.current_memory_bytes.load();
  out->active_delegate = metrics_.active_delegate;

  return GOLLEK_OK;
}

void Engine::reset_metrics() {
  metrics_.total_inferences = 0;
  metrics_.failed_inferences = 0;
  metrics_.total_latency_us = 0;
  metrics_.peak_memory_bytes = 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Engine — Diagnostics (Private)
 * ══════════════════════════════════════════════════════════════════════════ */

void Engine::set_error(const char *msg) const {
  if (msg) {
    last_error_ = msg;
  }
}

uint64_t Engine::get_percentile_latency(double p) const {
  if (latencies_.empty()) return 0;

  // Sort latencies (simple bubble sort for small samples)
  std::vector<uint64_t> sorted = latencies_;
  std::sort(sorted.begin(), sorted.end());

  size_t index = static_cast<size_t>(p * sorted.size());
  if (index >= sorted.size()) {
    index = sorted.size() - 1;
  }
  return sorted[index];
}

}  // namespace gollek

/* ══════════════════════════════════════════════════════════════════════════
 * C API Wrapper Functions
 * ══════════════════════════════════════════════════════════════════════════ */

extern "C" {

#define ENGINE(h) (reinterpret_cast<gollek::Engine *>(h))

GollekEngineHandle gollek_engine_create(const GollekConfig *cfg) {
  GollekConfig config = cfg ? *cfg : GollekConfig{4, GOLLEK_DELEGATE_AUTO, 1, 1, 0};
  config.format = ModelFormat::UNKNOWN;  // Will auto-detect
  return ENGINE(new gollek::Engine(config));
}

void gollek_engine_destroy(GollekEngineHandle h) {
  delete ENGINE(h);
}

GollekStatus gollek_load_model_from_file(GollekEngineHandle h,
                                         const char *path) {
  return ENGINE(h)->load_from_file(path);
}

GollekStatus gollek_load_model_from_buffer(GollekEngineHandle h,
                                           const void *data, size_t size) {
  return ENGINE(h)->load_from_buffer(data, size);
}

int gollek_get_input_count(GollekEngineHandle h) {
  return ENGINE(h)->input_count();
}

int gollek_get_output_count(GollekEngineHandle h) {
  return ENGINE(h)->output_count();
}

GollekStatus gollek_get_input_info(GollekEngineHandle h, int idx,
                                   GollekTensorInfo *out) {
  return ENGINE(h)->input_info(idx, out);
}

GollekStatus gollek_get_output_info(GollekEngineHandle h, int idx,
                                    GollekTensorInfo *out) {
  return ENGINE(h)->output_info(idx, out);
}

GollekStatus gollek_resize_input(GollekEngineHandle h, int idx,
                                 const int32_t *dims, int num_dims) {
  return ENGINE(h)->resize_input(idx, dims, num_dims);
}

GollekStatus gollek_allocate_tensors(GollekEngineHandle h) {
  return ENGINE(h)->allocate_tensors();
}

GollekStatus gollek_set_input(GollekEngineHandle h, int idx, const void *src,
                              size_t bytes) {
  return ENGINE(h)->set_input(idx, src, bytes);
}

GollekStatus gollek_invoke(GollekEngineHandle h) { return ENGINE(h)->invoke(); }

GollekStatus gollek_get_output(GollekEngineHandle h, int idx, void *dst,
                               size_t bytes) {
  return ENGINE(h)->get_output(idx, dst, bytes);
}

GollekStatus gollek_infer(GollekEngineHandle h, const void *input,
                          size_t input_bytes, void *output,
                          size_t output_bytes) {
  GollekStatus s;
  if ((s = gollek_set_input(h, 0, input, input_bytes)) != GOLLEK_OK)
    return s;
  if ((s = gollek_invoke(h)) != GOLLEK_OK)
    return s;
  return gollek_get_output(h, 0, output, output_bytes);
}

const char *gollek_status_string(GollekStatus status) {
  switch (status) {
  case GOLLEK_OK:
    return "OK";
  case GOLLEK_ERROR:
    return "Generic error";
  case GOLLEK_ERROR_MODEL_LOAD:
    return "Model load failed";
  case GOLLEK_ERROR_ALLOC_TENSORS:
    return "Tensor allocation failed";
  case GOLLEK_ERROR_INVOKE:
    return "Interpreter invoke failed";
  case GOLLEK_ERROR_INVALID_ARG:
    return "Invalid argument";
  case GOLLEK_ERROR_NOT_INITIALIZED:
    return "Engine not initialized";
  case GOLLEK_ERROR_DELEGATE_FAILED:
    return "Delegate creation failed";
  default:
    return "Unknown error";
  }
}

const char *gollek_last_error(GollekEngineHandle h) {
  return h ? ENGINE(h)->last_error() : "null engine handle";
}

const char *gollek_version(void) { return GOLLEK_VERSION; }

GollekStatus gollek_get_metrics(GollekEngineHandle h, GollekMetrics *out) {
  return ENGINE(h)->get_metrics(out);
}

GollekStatus gollek_reset_metrics(GollekEngineHandle h) {
  ENGINE(h)->reset_metrics();
  return GOLLEK_OK;
}

GollekStatus gollek_set_batch_input(GollekEngineHandle h, int idx,
                                    const void *const *inputs,
                                    const size_t *input_bytes, int num_inputs) {
  return ENGINE(h)->set_batch_input(idx, inputs, input_bytes, num_inputs);
}

GollekStatus gollek_get_batch_output(GollekEngineHandle h, int idx,
                                     void *const *outputs,
                                     const size_t *output_bytes, int num_outputs) {
  return ENGINE(h)->get_batch_output(idx, outputs, output_bytes, num_outputs);
}

GollekStatus gollek_start_streaming(GollekEngineHandle h, const void *input,
                                    size_t input_bytes, int max_tokens,
                                    GollekStreamSessionHandle *out_session) {
  return ENGINE(h)->start_streaming(input, input_bytes, max_tokens, out_session);
}

GollekStatus gollek_stream_next(GollekStreamSessionHandle s, void *output,
                                size_t output_bytes, size_t *actual_bytes,
                                int *is_done) {
  auto *sess = reinterpret_cast<gollek::StreamingSession *>(s);
  return sess->next(output, output_bytes, actual_bytes, is_done);
}

void gollek_end_streaming(GollekStreamSessionHandle s) {
  auto *eng = reinterpret_cast<gollek::StreamingSession *>(s)->engine;
  eng->end_streaming(s);
}

}  // extern "C"
