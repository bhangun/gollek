/**
 * GGUF Bridge - Native C++ bridge for llama.cpp model conversion
 *
 * Provides a stable C API for Java FFM integration with comprehensive
 * error handling, progress tracking, and resource management.
 *
 * @author Bhangun
 * @version 1.0.0
 */

#ifndef GGUF_BRIDGE_HPP
#define GGUF_BRIDGE_HPP

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum {
  GGUF_SUCCESS = 0,
  GGUF_ERROR_INVALID_ARGS = -1,
  GGUF_ERROR_FILE_NOT_FOUND = -2,
  GGUF_ERROR_INVALID_FORMAT = -3,
  GGUF_ERROR_CONVERSION_FAILED = -4,
  GGUF_ERROR_OUT_OF_MEMORY = -5,
  GGUF_ERROR_CANCELLED = -6,
  GGUF_ERROR_UNSUPPORTED_ARCH = -7,
  GGUF_ERROR_INVALID_QUANTIZATION = -8,
  GGUF_ERROR_IO_ERROR = -9,
  GGUF_ERROR_UNKNOWN = -99
} gguf_error_code_t;

// Conversion context handle (opaque)
typedef struct gguf_conversion_ctx *gguf_ctx_t;

// Progress callback: (progress: 0.0-1.0, stage: string, user_data: void*)
typedef void (*gguf_progress_callback_t)(float progress, const char *stage,
                                         void *user_data);

// Log callback: (level: 0=debug,1=info,2=warn,3=error, message: string,
// user_data: void*)
typedef void (*gguf_log_callback_t)(int level, const char *message,
                                    void *user_data);

// Conversion parameters
typedef struct {
  const char *input_path;  // Path to input model (can be directory or file)
  const char *output_path; // Path to output GGUF file
  const char
      *model_type; // Model architecture hint (e.g., "llama", "mistral", "phi")
  const char *quantization; // Quantization type (e.g., "f16", "q4_k_m", "q8_0")
  int vocab_only;           // 1 to convert vocab only, 0 for full model
  int use_mmap;             // 1 to use memory mapping, 0 otherwise
  int num_threads;          // Number of threads for conversion (0 = auto)
  const char *vocab_type;   // Vocab type override (e.g., "bpe", "spm")
  int pad_vocab;            // Pad vocab to multiple of this (0 = no padding)
  const char *
      *metadata_overrides; // NULL-terminated array of "key=value" strings
  gguf_progress_callback_t progress_cb;
  gguf_log_callback_t log_cb;
  void *user_data;
} gguf_conversion_params_t;

// Model info structure
typedef struct {
  char model_type[64];
  char architecture[64];
  uint64_t parameter_count;
  uint32_t num_layers;
  uint32_t hidden_size;
  uint32_t vocab_size;
  uint32_t context_length;
  char quantization[32];
  uint64_t file_size;
} gguf_model_info_t;

/**
 * Get library version string
 */
const char *gguf_version(void);

/**
 * Get last error message (thread-local)
 */
const char *gguf_get_last_error(void);

/**
 * Clear last error
 */
void gguf_clear_error(void);

/**
 * Initialize default conversion parameters
 */
void gguf_default_params(gguf_conversion_params_t *params);

/**
 * Create conversion context
 * Returns NULL on error (check gguf_get_last_error)
 */
gguf_ctx_t gguf_create_context(const gguf_conversion_params_t *params);

/**
 * Validate input model format and extract metadata
 * Returns GGUF_SUCCESS or error code
 */
int gguf_validate_input(gguf_ctx_t ctx, gguf_model_info_t *info);

/**
 * Execute conversion (blocking)
 * Returns GGUF_SUCCESS or error code
 */
int gguf_convert(gguf_ctx_t ctx);

/**
 * Request cancellation of ongoing conversion
 * Thread-safe, can be called from signal handler
 */
void gguf_cancel(gguf_ctx_t ctx);

/**
 * Check if conversion was cancelled
 */
int gguf_is_cancelled(gguf_ctx_t ctx);

/**
 * Get conversion progress (0.0 - 1.0)
 * Returns -1.0 if context is invalid
 */
float gguf_get_progress(gguf_ctx_t ctx);

/**
 * Free conversion context and associated resources
 */
void gguf_free_context(gguf_ctx_t ctx);

/**
 * Detect model format from path
 * Returns format string (e.g., "pytorch", "safetensors", "gguf")
 * Returns NULL if format cannot be detected
 */
const char *gguf_detect_format(const char *path);

/**
 * List available quantization types
 * Returns NULL-terminated array of strings
 * Caller must NOT free the returned array
 */
const char **gguf_available_quantizations(void);

/**
 * Verify GGUF file integrity
 * Returns GGUF_SUCCESS if file is valid
 */
int gguf_verify_file(const char *path, gguf_model_info_t *info);

#ifdef __cplusplus
}
#endif

#endif // GGUF_BRIDGE_HPP
