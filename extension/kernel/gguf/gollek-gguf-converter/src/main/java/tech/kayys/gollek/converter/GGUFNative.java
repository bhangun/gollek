package tech.kayys.gollek.converter;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Low-level FFM bindings to the GGUF bridge native library.
 * 
 * This class provides direct access to the C API using JDK 25's
 * Foreign Function & Memory API for maximum performance and safety.
 * 
 * <p>
 * All methods are thread-safe. Memory management is handled through
 * Arena scopes to prevent leaks.
 * 
 * @author Bhangun
 * @version 1.0.0
 */
public final class GGUFNative {

    private static final String LIBRARY_NAME = "gguf_bridge";

    // Linker and symbol lookup
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP;

    // Method handles (lazily initialized and cached)
    private static volatile MethodHandle gguf_version_handle;
    private static volatile MethodHandle gguf_get_last_error_handle;
    private static volatile MethodHandle gguf_clear_error_handle;
    private static volatile MethodHandle gguf_default_params_handle;
    private static volatile MethodHandle gguf_create_context_handle;
    private static volatile MethodHandle gguf_validate_input_handle;
    private static volatile MethodHandle gguf_convert_handle;
    private static volatile MethodHandle gguf_cancel_handle;
    private static volatile MethodHandle gguf_is_cancelled_handle;
    private static volatile MethodHandle gguf_get_progress_handle;
    private static volatile MethodHandle gguf_free_context_handle;
    private static volatile MethodHandle gguf_detect_format_handle;
    private static volatile MethodHandle gguf_available_quantizations_handle;
    private static volatile MethodHandle gguf_verify_file_handle;

    static {
        // Load native library
        System.loadLibrary(LIBRARY_NAME);

        // Initialize symbol lookup
        LOOKUP = SymbolLookup.loaderLookup();
    }

    // Prevent instantiation
    private GGUFNative() {
        throw new UnsupportedOperationException("Utility class");
    }

    // ========================================================================
    // Layout Definitions
    // ========================================================================

    /**
     * Layout for gguf_conversion_params_t structure
     */
    public static final StructLayout PARAMS_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.ADDRESS.withName("input_path"),
            ValueLayout.ADDRESS.withName("output_path"),
            ValueLayout.ADDRESS.withName("model_type"),
            ValueLayout.ADDRESS.withName("quantization"),
            ValueLayout.JAVA_INT.withName("vocab_only"),
            ValueLayout.JAVA_INT.withName("use_mmap"),
            ValueLayout.JAVA_INT.withName("num_threads"),
            MemoryLayout.paddingLayout(4), // Alignment padding for 8-byte Address
            ValueLayout.ADDRESS.withName("vocab_type"),
            ValueLayout.JAVA_INT.withName("pad_vocab"),
            MemoryLayout.paddingLayout(4), // Alignment padding for 8-byte Address
            ValueLayout.ADDRESS.withName("metadata_overrides"),
            ValueLayout.ADDRESS.withName("progress_cb"),
            ValueLayout.ADDRESS.withName("log_cb"),
            ValueLayout.ADDRESS.withName("user_data"));

    /**
     * Layout for gguf_model_info_t structure
     */
    public static final StructLayout MODEL_INFO_LAYOUT = MemoryLayout.structLayout(
            MemoryLayout.sequenceLayout(64, ValueLayout.JAVA_BYTE).withName("model_type"),
            MemoryLayout.sequenceLayout(64, ValueLayout.JAVA_BYTE).withName("architecture"),
            ValueLayout.JAVA_LONG.withName("parameter_count"),
            ValueLayout.JAVA_INT.withName("num_layers"),
            ValueLayout.JAVA_INT.withName("hidden_size"),
            ValueLayout.JAVA_INT.withName("vocab_size"),
            ValueLayout.JAVA_INT.withName("context_length"),
            MemoryLayout.sequenceLayout(32, ValueLayout.JAVA_BYTE).withName("quantization"),
            ValueLayout.JAVA_LONG.withName("file_size"));

    // ========================================================================
    // Function Descriptors
    // ========================================================================

    private static final FunctionDescriptor DESC_gguf_version = FunctionDescriptor.of(ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_get_last_error = FunctionDescriptor.of(ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_clear_error = FunctionDescriptor.ofVoid();

    private static final FunctionDescriptor DESC_gguf_default_params = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_create_context = FunctionDescriptor.of(ValueLayout.ADDRESS,
            ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_validate_input = FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_convert = FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_cancel = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_is_cancelled = FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_get_progress = FunctionDescriptor.of(ValueLayout.JAVA_FLOAT,
            ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_free_context = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_detect_format = FunctionDescriptor.of(ValueLayout.ADDRESS,
            ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_available_quantizations = FunctionDescriptor
            .of(ValueLayout.ADDRESS);

    private static final FunctionDescriptor DESC_gguf_verify_file = FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS);

    // ========================================================================
    // Callback Descriptors
    // ========================================================================

    /**
     * Progress callback descriptor: void(float, const char*, void*)
     */
    public static final FunctionDescriptor PROGRESS_CALLBACK_DESC = FunctionDescriptor.ofVoid(
            ValueLayout.JAVA_FLOAT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS);

    /**
     * Log callback descriptor: void(int, const char*, void*)
     */
    public static final FunctionDescriptor LOG_CALLBACK_DESC = FunctionDescriptor.ofVoid(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS);

    // ========================================================================
    // Method Handle Getters (with lazy initialization)
    // ========================================================================

    private static MethodHandle getVersionHandle() {
        if (gguf_version_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_version_handle == null) {
                    gguf_version_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_version").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_version")),
                            DESC_gguf_version);
                }
            }
        }
        return gguf_version_handle;
    }

    private static MethodHandle getLastErrorHandle() {
        if (gguf_get_last_error_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_get_last_error_handle == null) {
                    gguf_get_last_error_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_get_last_error").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_get_last_error")),
                            DESC_gguf_get_last_error);
                }
            }
        }
        return gguf_get_last_error_handle;
    }

    private static MethodHandle getClearErrorHandle() {
        if (gguf_clear_error_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_clear_error_handle == null) {
                    gguf_clear_error_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_clear_error").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_clear_error")),
                            DESC_gguf_clear_error);
                }
            }
        }
        return gguf_clear_error_handle;
    }

    private static MethodHandle getDefaultParamsHandle() {
        if (gguf_default_params_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_default_params_handle == null) {
                    gguf_default_params_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_default_params").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_default_params")),
                            DESC_gguf_default_params);
                }
            }
        }
        return gguf_default_params_handle;
    }

    private static MethodHandle getCreateContextHandle() {
        if (gguf_create_context_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_create_context_handle == null) {
                    gguf_create_context_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_create_context").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_create_context")),
                            DESC_gguf_create_context);
                }
            }
        }
        return gguf_create_context_handle;
    }

    private static MethodHandle getValidateInputHandle() {
        if (gguf_validate_input_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_validate_input_handle == null) {
                    gguf_validate_input_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_validate_input").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_validate_input")),
                            DESC_gguf_validate_input);
                }
            }
        }
        return gguf_validate_input_handle;
    }

    private static MethodHandle getConvertHandle() {
        if (gguf_convert_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_convert_handle == null) {
                    gguf_convert_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_convert").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_convert")),
                            DESC_gguf_convert);
                }
            }
        }
        return gguf_convert_handle;
    }

    private static MethodHandle getCancelHandle() {
        if (gguf_cancel_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_cancel_handle == null) {
                    gguf_cancel_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_cancel").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_cancel")),
                            DESC_gguf_cancel);
                }
            }
        }
        return gguf_cancel_handle;
    }

    private static MethodHandle getIsCancelledHandle() {
        if (gguf_is_cancelled_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_is_cancelled_handle == null) {
                    gguf_is_cancelled_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_is_cancelled").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_is_cancelled")),
                            DESC_gguf_is_cancelled);
                }
            }
        }
        return gguf_is_cancelled_handle;
    }

    private static MethodHandle getProgressHandle() {
        if (gguf_get_progress_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_get_progress_handle == null) {
                    gguf_get_progress_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_get_progress").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_get_progress")),
                            DESC_gguf_get_progress);
                }
            }
        }
        return gguf_get_progress_handle;
    }

    private static MethodHandle getFreeContextHandle() {
        if (gguf_free_context_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_free_context_handle == null) {
                    gguf_free_context_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_free_context").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_free_context")),
                            DESC_gguf_free_context);
                }
            }
        }
        return gguf_free_context_handle;
    }

    private static MethodHandle getDetectFormatHandle() {
        if (gguf_detect_format_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_detect_format_handle == null) {
                    gguf_detect_format_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_detect_format").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_detect_format")),
                            DESC_gguf_detect_format);
                }
            }
        }
        return gguf_detect_format_handle;
    }

    private static MethodHandle getAvailableQuantizationsHandle() {
        if (gguf_available_quantizations_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_available_quantizations_handle == null) {
                    gguf_available_quantizations_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_available_quantizations").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_available_quantizations")),
                            DESC_gguf_available_quantizations);
                }
            }
        }
        return gguf_available_quantizations_handle;
    }

    private static MethodHandle getVerifyFileHandle() {
        if (gguf_verify_file_handle == null) {
            synchronized (GGUFNative.class) {
                if (gguf_verify_file_handle == null) {
                    gguf_verify_file_handle = LINKER.downcallHandle(
                            LOOKUP.find("gguf_verify_file").orElseThrow(
                                    () -> new UnsatisfiedLinkError("gguf_verify_file")),
                            DESC_gguf_verify_file);
                }
            }
        }
        return gguf_verify_file_handle;
    }

    // ========================================================================
    // Public API Methods
    // ========================================================================

    /**
     * Get library version string.
     * 
     * @return version string
     */
    public static String getVersion() {
        try {
            MemorySegment result = (MemorySegment) getVersionHandle().invoke();
            return result.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get version", t);
        }
    }

    /**
     * Get last error message (thread-local).
     * 
     * @return error message or empty string if no error
     */
    public static String getLastError() {
        try {
            MemorySegment result = (MemorySegment) getLastErrorHandle().invoke();
            if (result.address() == 0) {
                return "";
            }
            return result.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            return "Failed to get error: " + t.getMessage();
        }
    }

    /**
     * Clear last error.
     */
    public static void clearError() {
        try {
            getClearErrorHandle().invoke();
        } catch (Throwable t) {
            // Ignore
        }
    }

    /**
     * Initialize conversion parameters with defaults.
     * 
     * @param arena memory arena
     * @return memory segment containing default parameters
     */
    public static MemorySegment defaultParams(Arena arena) {
        try {
            MemorySegment params = arena.allocate(PARAMS_LAYOUT);
            getDefaultParamsHandle().invoke(params);
            return params;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to initialize default params", t);
        }
    }

    /**
     * Create conversion context.
     * 
     * @param params conversion parameters
     * @return context handle (address)
     * @throws RuntimeException if creation fails
     */
    public static MemorySegment createContext(MemorySegment params) {
        try {
            MemorySegment ctx = (MemorySegment) getCreateContextHandle().invoke(params);
            if (ctx.address() == 0) {
                throw new RuntimeException("Failed to create context: " + getLastError());
            }
            return ctx;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to create context", t);
        }
    }

    /**
     * Validate input model.
     * 
     * @param ctx  context handle
     * @param info optional model info output (can be NULL)
     * @return error code (0 = success)
     */
    public static int validateInput(MemorySegment ctx, MemorySegment info) {
        try {
            return (int) getValidateInputHandle().invoke(ctx, info);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to validate input", t);
        }
    }

    /**
     * Execute conversion.
     *
     * @param ctx context handle
     * @return error code (0 = success)
     */
    public static int convert(MemorySegment ctx) {
        try {
            return (int) getConvertHandle().invoke(ctx);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to execute conversion", t);
        }
    }

    /**
     * Request cancellation.
     *
     * @param ctx context handle
     */
    public static void cancel(MemorySegment ctx) {
        try {
            getCancelHandle().invoke(ctx);
        } catch (Throwable t) {
            // Ignore
        }
    }

    /**
     * Check if cancelled.
     * 
     * @param ctx context handle
     * @return 1 if cancelled, 0 otherwise
     */
    public static int isCancelled(MemorySegment ctx) {
        try {
            return (int) getIsCancelledHandle().invoke(ctx);
        } catch (Throwable t) {
            return 0;
        }
    }

    /**
     * Get conversion progress.
     * 
     * @param ctx context handle
     * @return progress (0.0 - 1.0) or -1.0 on error
     */
    public static float getProgress(MemorySegment ctx) {
        try {
            return (float) getProgressHandle().invoke(ctx);
        } catch (Throwable t) {
            return -1.0f;
        }
    }

    /**
     * Free conversion context.
     * 
     * @param ctx context handle
     */
    public static void freeContext(MemorySegment ctx) {
        try {
            getFreeContextHandle().invoke(ctx);
        } catch (Throwable t) {
            // Ignore
        }
    }

    /**
     * Detect model format from path.
     * 
     * @param arena memory arena
     * @param path  file or directory path
     * @return format string or null if not detected
     */
    public static String detectFormat(Arena arena, String path) {
        try {
            MemorySegment pathSeg = arena.allocateFrom(path);
            MemorySegment result = (MemorySegment) getDetectFormatHandle().invoke(pathSeg);
            if (result.address() == 0) {
                return null;
            }
            return result.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * Get available quantization types.
     * 
     * @return array of quantization type strings
     */
    public static String[] getAvailableQuantizations() {
        try {
            MemorySegment array = (MemorySegment) getAvailableQuantizationsHandle().invoke();

            // Count strings
            int count = 0;
            while (true) {
                MemorySegment ptr = array.getAtIndex(ValueLayout.ADDRESS, count);
                if (ptr.address() == 0)
                    break;
                count++;
            }

            // Extract strings
            String[] result = new String[count];
            for (int i = 0; i < count; i++) {
                MemorySegment ptr = array.getAtIndex(ValueLayout.ADDRESS, i);
                result[i] = ptr.reinterpret(Long.MAX_VALUE).getString(0);
            }

            return result;
        } catch (Throwable t) {
            return new String[0];
        }
    }

    /**
     * Verify GGUF file integrity.
     * 
     * @param arena memory arena
     * @param path  file path
     * @param info  optional model info output
     * @return error code (0 = success)
     */
    public static int verifyFile(Arena arena, String path, MemorySegment info) {
        try {
            MemorySegment pathSeg = arena.allocateFrom(path);
            return (int) getVerifyFileHandle().invoke(pathSeg, info);
        } catch (Throwable t) {
            return -99;
        }
    }
}
