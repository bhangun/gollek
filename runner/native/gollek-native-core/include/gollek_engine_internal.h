/**
 * gollek_engine_internal.h
 * Internal C++ class — not part of the public API.
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#pragma once

#include "gollek_engine.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gollek {

/* ── Model format enumeration ─────────────────────────────────────────── */

enum class ModelFormat {
  UNKNOWN = 0,
  TFLITE = 1,   // .litertlm, .tfl
  ONNX = 2,     // .onnx
};

/**
 * Detect model format from file path.
 * Checks file extension and optionally magic bytes.
 */
ModelFormat detect_format(const char *path);

/* ── Abstract runtime backend interface ────────────────────────────────── */

/**
 * RuntimeBackend is the abstract interface that TFLite and ONNX backends
 * must implement. The Engine class delegates all inference operations to
 * the active backend instance.
 */
class RuntimeBackend {
 public:
  virtual ~RuntimeBackend() = default;

  // Model loading
  virtual GollekStatus load_from_file(const char *path) = 0;
  virtual GollekStatus load_from_buffer(const void *data, size_t size) = 0;

  // Tensor introspection
  virtual int input_count() const = 0;
  virtual int output_count() const = 0;
  virtual GollekStatus input_info(int index, GollekTensorInfo *out) const = 0;
  virtual GollekStatus output_info(int index, GollekTensorInfo *out) const = 0;

  // Dynamic shape support
  virtual GollekStatus resize_input(int index, const int32_t *dims,
                                    int num_dims) = 0;
  virtual GollekStatus allocate_tensors() = 0;

  // Inference
  virtual GollekStatus set_input(int index, const void *src,
                                 size_t bytes) = 0;
  virtual GollekStatus invoke() = 0;
  virtual GollekStatus get_output(int index, void *dst,
                                  size_t dst_bytes) = 0;

  // Batched inference (optional; base implementation may return UNSUPPORTED)
  virtual GollekStatus set_batch_input(int index, const void *const *inputs,
                                       const size_t *input_bytes,
                                       int num_inputs) {
    return GOLLEK_ERROR; // Not all backends need to support batching
  }
  virtual GollekStatus get_batch_output(int index, void *const *outputs,
                                        const size_t *output_bytes,
                                        int num_outputs) {
    return GOLLEK_ERROR;
  }

  // Diagnostics
  virtual const char *last_error() const = 0;
};

/* ── Memory pool ──────────────────────────────────────────────────────── */

/**
 * A very simple slab-based memory pool to avoid per-inference allocations.
 * Each allocation returns a pointer into a pre-allocated contiguous buffer.
 * The pool is reset (not freed) between inferences, which is safe because
 * we copy tensors in / out of TFLite immediately.
 */
class MemoryPool {
public:
  explicit MemoryPool(size_t capacity_bytes);
  ~MemoryPool();

  /**
   * Returns a pointer aligned to `alignment` bytes within the slab.
   * Returns nullptr when the slab is exhausted.
   */
  void *allocate(size_t bytes, size_t alignment = 16);

  /** Release all allocations at once — O(1). */
  void reset();

  size_t used() const { return offset_; }
  size_t capacity() const { return capacity_; }

private:
  uint8_t *slab_ = nullptr;
  size_t capacity_ = 0;
  size_t offset_ = 0;

  // Non-copyable
  MemoryPool(const MemoryPool &) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;
};

/* ── Delegate helpers (platform-specific, defined in platform/*.cpp) ─── */

/**
 * Try to create the best available delegate for the current platform/config.
 * Returns nullptr when no suitable delegate is found (CPU fallback).
 * Ownership is transferred to the caller; must call LitertDelegate destroy.
 */
LitertDelegate *create_delegate(GollekDelegate preference,
                                std::string &error_out);

/* ── Performance metrics ─────────────────────────────────────────────── */

struct EngineMetrics {
  std::atomic<uint64_t> total_inferences{0};
  std::atomic<uint64_t> failed_inferences{0};
  std::atomic<uint64_t> total_latency_us{0};
  std::atomic<uint64_t> peak_memory_bytes{0};
  std::atomic<uint64_t> current_memory_bytes{0};
  GollekDelegate active_delegate{GOLLEK_DELEGATE_CPU};

  void record_latency(uint64_t latency_us) { total_latency_us += latency_us; }

  uint64_t get_avg_latency_us() const {
    uint64_t total = total_inferences.load();
    return total > 0 ? total_latency_us.load() / total : 0;
  }
};

/* ── Streaming session ───────────────────────────────────────────────── */

struct StreamingSession {
  Engine *engine;
  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> output_buffer;
  int max_tokens;
  int tokens_generated;
  bool is_done;
  std::mutex session_mutex;

  StreamingSession(Engine *eng, int max_tok)
      : engine(eng), max_tokens(max_tok), tokens_generated(0), is_done(false) {}

  ~StreamingSession() = default;

  GollekStatus start(const void *input, size_t input_bytes);
  GollekStatus next(void *output, size_t output_bytes, size_t *actual_bytes,
                    int *done);
};

/* ── Core engine ──────────────────────────────────────────────────────── */

class Engine {
 public:
  explicit Engine(const GollekConfig &config);
  ~Engine();

  GollekStatus load_from_file(const char *path);
  GollekStatus load_from_buffer(const void *data, size_t size);

  int input_count() const;
  int output_count() const;
  GollekStatus input_info(int index, GollekTensorInfo *out) const;
  GollekStatus output_info(int index, GollekTensorInfo *out) const;

  GollekStatus resize_input(int index, const int32_t *dims, int num_dims);
  GollekStatus allocate_tensors();

  GollekStatus set_input(int index, const void *src, size_t bytes);
  GollekStatus invoke();
  GollekStatus get_output(int index, void *dst, size_t dst_bytes);

  // Batched inference
  GollekStatus set_batch_input(int index, const void *const *inputs,
                               const size_t *input_bytes, int num_inputs);
  GollekStatus get_batch_output(int index, void *const *outputs,
                                const size_t *output_bytes, int num_outputs);

  // Streaming
  GollekStatus start_streaming(const void *input, size_t input_bytes,
                               int max_tokens,
                               GollekStreamSessionHandle *out_session);
  void end_streaming(GollekStreamSessionHandle session);

  // Metrics
  const EngineMetrics &get_metrics() const { return metrics_; }
  GollekStatus get_metrics(GollekMetrics *out) const;
  void reset_metrics();

  const char *last_error() const { return last_error_.c_str(); }

 private:
  GollekConfig config_;

  // The active runtime backend (TFLite or ONNX)
  std::unique_ptr<RuntimeBackend> backend_;

  // Memory pool for scratch buffers
  std::unique_ptr<MemoryPool> pool_;

  // Last error description
  mutable std::string last_error_;

  // Mutex for thread-safe invoke()
  mutable std::mutex invoke_mutex_;

  // Performance metrics
  mutable EngineMetrics metrics_;

  // Latency history for percentile calculations (circular buffer)
  mutable std::vector<uint64_t> latencies_;
  mutable size_t latency_index_ = 0;
  static constexpr size_t MAX_LATENCY_SAMPLES = 10000;

  // Active streaming sessions
  mutable std::vector<std::unique_ptr<StreamingSession>> streaming_sessions_;
  mutable std::mutex streaming_mutex_;

  bool is_ready_ = false; ///< model loaded + tensors allocated

  // Backend initialization
  GollekStatus init_backend();
  void set_error(const char *msg) const;
  uint64_t get_percentile_latency(double p) const;
};

} // namespace gollek
