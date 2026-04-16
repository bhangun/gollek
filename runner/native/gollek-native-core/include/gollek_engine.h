/**
 * gollek_engine.h
 * Gollek Inference Engine — Unified C API
 *
 * This is the single public header that all platform bridges (JNI, ObjC++,
 * Emscripten, Windows DLL) must include. It exposes a flat C ABI so that
 * any language can bind to it without a C++ compiler.
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handle ─────────────────────────────────────────────────────── */

/** Opaque pointer to an engine instance. Never access internals directly. */
typedef struct GollekEngine_t *GollekEngineHandle;

/* ── Status codes (mirror LitertStatus for compatibility) ──────────────── */

typedef enum {
  GOLLEK_OK = 0,
  GOLLEK_ERROR = 1,
  GOLLEK_ERROR_MODEL_LOAD = 2,
  GOLLEK_ERROR_ALLOC_TENSORS = 3,
  GOLLEK_ERROR_INVOKE = 4,
  GOLLEK_ERROR_INVALID_ARG = 5,
  GOLLEK_ERROR_NOT_INITIALIZED = 6,
  GOLLEK_ERROR_DELEGATE_FAILED = 7,
} GollekStatus;

/* ── Delegate / hardware acceleration options ──────────────────────────── */

typedef enum {
  GOLLEK_DELEGATE_NONE = 0,
  GOLLEK_DELEGATE_CPU = 1,     /**< XNNPACK (all platforms)              */
  GOLLEK_DELEGATE_GPU = 2,     /**< Metal (iOS/macOS) / OpenCL (Android) */
  GOLLEK_DELEGATE_NNAPI = 3,   /**< Android NNAPI                        */
  GOLLEK_DELEGATE_HEXAGON = 4, /**< Qualcomm Hexagon DSP                 */
  GOLLEK_DELEGATE_COREML = 5,  /**< Apple Core ML                        */
  GOLLEK_DELEGATE_AUTO = 6,    /**< Let the engine choose                */
} GollekDelegate;

/* ── Engine configuration ─────────────────────────────────────────────── */

typedef struct {
  int num_threads;         /**< CPU thread count; 0 = auto        */
  GollekDelegate delegate; /**< Hardware backend preference        */
  int enable_xnnpack;      /**< 1 = always add XNNPACK on CPU path */
  int use_memory_pool;     /**< 1 = enable internal buffer pooling */
  size_t pool_size_bytes;  /**< Pre-allocated pool size (0 = 16 MB)*/
} GollekConfig;

/* ── Tensor descriptor ────────────────────────────────────────────────── */

#define GOLLEK_MAX_DIMS 8

typedef enum {
  GOLLEK_TYPE_FLOAT32 = 0,
  GOLLEK_TYPE_FLOAT16 = 1,
  GOLLEK_TYPE_INT32 = 2,
  GOLLEK_TYPE_UINT8 = 3,
  GOLLEK_TYPE_INT64 = 4,
  GOLLEK_TYPE_INT8 = 6,
  GOLLEK_TYPE_BOOL = 9,
} GollekTensorType;

typedef struct {
  const char *name;
  GollekTensorType type;
  int32_t dims[GOLLEK_MAX_DIMS];
  int num_dims;
  size_t byte_size;
  float scale;        /**< Quantization scale  (0.0 if none) */
  int32_t zero_point; /**< Quantization offset (0   if none) */
} GollekTensorInfo;

/* ── Lifecycle ────────────────────────────────────────────────────────── */

/**
 * Create an engine instance with the given configuration.
 * Pass NULL for config to use reasonable defaults.
 */
GollekEngineHandle gollek_engine_create(const GollekConfig *config);

/**
 * Destroy an engine instance and free all resources.
 * Safe to call with NULL.
 */
void gollek_engine_destroy(GollekEngineHandle engine);

/* ── Model loading ────────────────────────────────────────────────────── */

/** Load a .litertlm model from an on-disk file path. */
GollekStatus gollek_load_model_from_file(GollekEngineHandle engine,
                                         const char *path);

/**
 * Load a .litertlm model from a memory buffer.
 * The buffer must remain valid for the lifetime of the engine (no copy).
 */
GollekStatus gollek_load_model_from_buffer(GollekEngineHandle engine,
                                           const void *data, size_t size);

/* ── Tensor introspection ─────────────────────────────────────────────── */

int gollek_get_input_count(GollekEngineHandle engine);
int gollek_get_output_count(GollekEngineHandle engine);

GollekStatus gollek_get_input_info(GollekEngineHandle engine, int index,
                                   GollekTensorInfo *out);
GollekStatus gollek_get_output_info(GollekEngineHandle engine, int index,
                                    GollekTensorInfo *out);

/* ── Dynamic shape support ────────────────────────────────────────────── */

/**
 * Resize input tensor at `index` to a new shape before inference.
 * Must call gollek_allocate_tensors() after all resize calls.
 */
GollekStatus gollek_resize_input(GollekEngineHandle engine, int index,
                                 const int32_t *dims, int num_dims);

/** Re-allocate internal buffers after shape changes. */
GollekStatus gollek_allocate_tensors(GollekEngineHandle engine);

/* ── Inference ────────────────────────────────────────────────────────── */

/**
 * Copy `bytes` bytes from `src` into the input tensor at `index`.
 * The engine validates bounds and type size.
 */
GollekStatus gollek_set_input(GollekEngineHandle engine, int index,
                              const void *src, size_t bytes);

/** Run one forward pass. */
GollekStatus gollek_invoke(GollekEngineHandle engine);

/**
 * Copy the output tensor at `index` into `dst`.
 * `dst_bytes` must be >= gollek_get_output_info(...).byte_size.
 */
GollekStatus gollek_get_output(GollekEngineHandle engine, int index, void *dst,
                               size_t dst_bytes);

/* ── Convenience: single-shot inference ─────────────────────────────── */

/**
 * Set one input, invoke, copy one output — the "happy path" for models
 * with a single input/output tensor.
 */
GollekStatus gollek_infer(GollekEngineHandle engine, const void *input,
                          size_t input_bytes, void *output,
                          size_t output_bytes);

/* ── Performance metrics ──────────────────────────────────────────────── */

/** Engine performance metrics structure */
typedef struct {
  uint64_t total_inferences;
  uint64_t failed_inferences;
  uint64_t total_latency_us;
  uint64_t avg_latency_us;
  uint64_t p50_latency_us;
  uint64_t p95_latency_us;
  uint64_t p99_latency_us;
  uint64_t peak_memory_bytes;
  uint64_t current_memory_bytes;
  GollekDelegate active_delegate;
} GollekMetrics;

/** Retrieve engine metrics */
GollekStatus gollek_get_metrics(GollekEngineHandle engine, GollekMetrics *out);

/** Reset engine metrics counters */
GollekStatus gollek_reset_metrics(GollekEngineHandle engine);

/* ── Batched inference ────────────────────────────────────────────────── */

/**
 * Set batched input tensors for a single input index.
 * Inputs are concatenated along the batch dimension (axis 0).
 *
 * @param engine     Engine handle
 * @param index      Input tensor index
 * @param inputs     Array of pointers to input data
 * @param input_bytes Array of input sizes in bytes
 * @param num_inputs Number of inputs in the batch
 * @return GOLLEK_OK on success
 */
GollekStatus gollek_set_batch_input(GollekEngineHandle engine, int index,
                                    const void *const *inputs,
                                    const size_t *input_bytes, int num_inputs);

/**
 * Get batched output tensors for a single output index.
 * Outputs are split along the batch dimension (axis 0).
 *
 * @param engine      Engine handle
 * @param index       Output tensor index
 * @param outputs     Array of pointers to output buffers (pre-allocated)
 * @param output_bytes Array of output buffer sizes in bytes
 * @param num_outputs Number of outputs expected
 * @return GOLLEK_OK on success
 */
GollekStatus gollek_get_batch_output(GollekEngineHandle engine, int index,
                                     void *const *outputs,
                                     const size_t *output_bytes,
                                     int num_outputs);

/* ── Streaming inference ──────────────────────────────────────────────── */

/** Opaque handle to a streaming session */
typedef struct GollekStreamSession_t *GollekStreamSessionHandle;

/**
 * Start a streaming inference session.
 * For auto-regressive models (LLMs), this initializes the decoding loop.
 *
 * @param engine     Engine handle
 * @param input      Input data (e.g., prompt tokens)
 * @param input_bytes Input size in bytes
 * @param max_tokens Maximum tokens to generate (0 = unlimited)
 * @param out_session Output session handle
 * @return GOLLEK_OK on success
 */
GollekStatus gollek_start_streaming(GollekEngineHandle engine,
                                    const void *input, size_t input_bytes,
                                    int max_tokens,
                                    GollekStreamSessionHandle *out_session);

/**
 * Get the next chunk from a streaming session.
 *
 * @param session    Streaming session handle
 * @param output     Output buffer for the chunk
 * @param output_bytes Size of output buffer in bytes
 * @param actual_bytes Actual bytes written to output
 * @param is_done    Pointer to int that will be set to 1 if streaming is
 * complete
 * @return GOLLEK_OK on success
 */
GollekStatus gollek_stream_next(GollekStreamSessionHandle session, void *output,
                                size_t output_bytes, size_t *actual_bytes,
                                int *is_done);

/**
 * End a streaming session and free its resources.
 *
 * @param session Streaming session handle
 */
void gollek_end_streaming(GollekStreamSessionHandle session);

/* ── Diagnostics ──────────────────────────────────────────────────────── */

/** Human-readable string for a status code. */
const char *gollek_status_string(GollekStatus status);

/** Return the last error message (thread-local, valid until next call). */
const char *gollek_last_error(GollekEngineHandle engine);

/** Engine version string (e.g. "1.0.0+litert-2.16"). */
const char *gollek_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
