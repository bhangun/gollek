package tech.kayys.gollek.converter;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;

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
    private static final String LIBRARY_VERSION = "1.0.0";
    private static final System.Logger LOG = System.getLogger(GGUFNative.class.getName());

    // Linker and symbol lookup
    private static final Linker LINKER = Linker.nativeLinker();
    private static SymbolLookup LOOKUP;
    private static volatile boolean AVAILABLE;
    private static volatile Throwable LOAD_ERROR;

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
        AVAILABLE = false;
        LOAD_ERROR = null;

        // Strategy 1: Check system property for explicit library path (highest priority)
        if (!AVAILABLE) {
            try {
                String explicitLibPath = System.getProperty("gollek.gguf.native.library.path");
                if (explicitLibPath != null && !explicitLibPath.isBlank()) {
                    Path libPath = Path.of(explicitLibPath).toAbsolutePath();
                    if (Files.exists(libPath)) {
                        System.load(libPath.toAbsolutePath().toString());
                        LOOKUP = SymbolLookup.loaderLookup();
                        AVAILABLE = true;
                        LOG.log(System.Logger.Level.INFO,
                                "GGUF native bridge loaded from explicit path: " + libPath);
                    }
                }
            } catch (Throwable ignored) { }
        }

        // Strategy 2: Extract from JAR classpath resource to versioned ~/.gollek/libs/
        if (!AVAILABLE) {
            try {
                Path extracted = extractFromClasspath();
                if (extracted != null) {
                    System.load(extracted.toAbsolutePath().toString());
                    LOOKUP = SymbolLookup.loaderLookup();
                    AVAILABLE = true;
                    LOG.log(System.Logger.Level.INFO,
                            "GGUF native bridge loaded from extracted resource: " + extracted);
                }
            } catch (Throwable ignored) { }
        }

        // Strategy 3: Standard ~/.gollek/libs/gguf_bridge/1.0.0/ location (permanent installation)
        if (!AVAILABLE) {
            try {
                Path stdLib = getStandardLibPath();
                if (Files.exists(stdLib)) {
                    System.load(stdLib.toAbsolutePath().toString());
                    LOOKUP = SymbolLookup.loaderLookup();
                    AVAILABLE = true;
                    LOG.log(System.Logger.Level.INFO,
                            "GGUF native bridge loaded from standard location: " + stdLib);
                }
            } catch (Throwable t) {
                LOAD_ERROR = t;
            }
        }

        // Strategy 4: Known build directory (development mode) - explicit fallback
        if (!AVAILABLE) {
            try {
                Path buildLib = findBuildDirectoryLib();
                if (buildLib != null) {
                    System.load(buildLib.toAbsolutePath().toString());
                    LOOKUP = SymbolLookup.loaderLookup();
                    AVAILABLE = true;
                    LOG.log(System.Logger.Level.INFO,
                            "GGUF native bridge loaded from build directory: " + buildLib);
                }
            } catch (Throwable ignored) { }
        }

        // Strategy 5: System.loadLibrary as last resort (works when -Djava.library.path is set)
        if (!AVAILABLE) {
            try {
                System.loadLibrary(LIBRARY_NAME);
                LOOKUP = SymbolLookup.loaderLookup();
                AVAILABLE = true;
                LOG.log(System.Logger.Level.INFO, "GGUF native bridge loaded via System.loadLibrary");
            } catch (Throwable ignored) { }
        }

        if (!AVAILABLE) {
            if (LOAD_ERROR == null) {
                LOAD_ERROR = new UnsatisfiedLinkError(
                        "GGUF native bridge '" + LIBRARY_NAME
                                + "' not found in system library path, classpath resources, "
                                + "build directory, or ~/.gollek/libs/");
            }
            LOG.log(System.Logger.Level.WARNING,
                    "GGUF native library '" + LIBRARY_NAME
                            + "' is unavailable; GGUF conversion will run in degraded mode.");
        }
    }

    /**
     * Extract native library from JAR classpath resource to a versioned directory
     * under {@code ~/.gollek/libs/gguf_bridge/<version>/}.
     * Re-extracts if the SHA-256 checksum differs from the bundled copy.
     */
    private static Path extractFromClasspath() {
        String libFileName = getNativeLibFileName();
        String resourcePath = "native-libs/" + libFileName;

        try (InputStream stream = GGUFNative.class.getClassLoader()
                .getResourceAsStream(resourcePath)) {
            if (stream == null) return null;

            byte[] resourceBytes = stream.readAllBytes();
            String resourceChecksum = sha256(resourceBytes);

            Path extractDir = Path.of(System.getProperty("user.home"),
                    ".gollek", "libs", LIBRARY_NAME, LIBRARY_VERSION);
            Files.createDirectories(extractDir);

            Path targetLib = extractDir.resolve(libFileName);
            Path checksumFile = extractDir.resolve(libFileName + ".sha256");

            boolean needsExtraction = true;
            if (Files.exists(targetLib) && Files.exists(checksumFile)) {
                String existing = Files.readString(checksumFile).trim();
                if (resourceChecksum.equals(existing)) {
                    needsExtraction = false;
                }
            }

            if (needsExtraction) {
                Files.write(targetLib, resourceBytes);
                Files.writeString(checksumFile, resourceChecksum);
                targetLib.toFile().setExecutable(true);
                clearMacQuarantine(extractDir);
                LOG.log(System.Logger.Level.INFO,
                        "Extracted native library to: " + targetLib);
            }
            return targetLib;
        } catch (IOException e) {
            LOG.log(System.Logger.Level.DEBUG,
                    "Failed to extract native library: " + e.getMessage());
            return null;
        }
    }

    /**
     * Find the native library in build directories (development mode).
     * Searches multiple common build output locations.
     */
    private static Path findBuildDirectoryLib() {
        String libFileName = getNativeLibFileName();
        
        // Comprehensive list of build directory paths to search
        String[] devPaths = {
                // Primary build directory
                "gollek/plugins/runner/gguf/gguf-bridge/build/" + libFileName,
                
                // Relative paths from various working directories
                "../gguf-bridge/build/" + libFileName,
                "gguf-bridge/build/" + libFileName,
                "build/" + libFileName,
                
                // Maven/Gradle output directories
                "gollek/plugins/runner/gguf/gguf-bridge/target/" + libFileName,
                "../target/" + libFileName,
                "target/" + libFileName,
                
                // CMake build subdirectories
                "gollek/plugins/runner/gguf/gguf-bridge/build/Release/" + libFileName,
                "gollek/plugins/runner/gguf/gguf-bridge/build/Debug/" + libFileName,
                "gollek/plugins/runner/gguf/gguf-bridge/build/RelWithDebInfo/" + libFileName,
                
                // Alternative build locations
                "gollek/plugins/runner/gguf/build/" + libFileName,
                "gollek/plugins/runner/build/" + libFileName,
                
                // User home build location
                Path.of(System.getProperty("user.home"), ".gollek", "source", "gguf-bridge", "build", libFileName).toString(),
        };
        
        for (String p : devPaths) {
            Path c = Path.of(p);
            if (Files.isRegularFile(c)) {
                LOG.log(System.Logger.Level.DEBUG, "Found native library in build directory: " + c);
                return c;
            }
        }
        
        LOG.log(System.Logger.Level.DEBUG, "Native library not found in any build directory");
        return null;
    }

    /**
     * Install the native library to the standard location (~/.gollek/libs/gguf_bridge/1.0.0/).
     * This method can be called during build or runtime to permanently install the library.
     * 
     * @param sourcePath the source library path (e.g., from build directory)
     * @return the installed library path, or null if installation failed
     */
    public static Path installToStandardLocation(Path sourcePath) {
        if (!Files.exists(sourcePath)) {
            LOG.log(System.Logger.Level.WARNING, "Source library does not exist: " + sourcePath);
            return null;
        }

        try {
            Path stdLibDir = getStandardLibPath().getParent();
            Files.createDirectories(stdLibDir);

            Path targetLib = stdLibDir.resolve(sourcePath.getFileName());
            Files.copy(sourcePath, targetLib, java.nio.file.StandardCopyOption.REPLACE_EXISTING);

            // Set executable on Unix-like systems
            if (!isWindows()) {
                targetLib.toFile().setExecutable(true);
            }

            // Clear macOS quarantine
            clearMacQuarantine(stdLibDir);

            // Write checksum for future verification
            byte[] libBytes = Files.readAllBytes(targetLib);
            String checksum = sha256(libBytes);
            Path checksumFile = stdLibDir.resolve(sourcePath.getFileName() + ".sha256");
            Files.writeString(checksumFile, checksum);

            LOG.log(System.Logger.Level.INFO,
                    "Installed GGUF native bridge to standard location: " + targetLib);
            return targetLib;
        } catch (IOException e) {
            LOG.log(System.Logger.Level.ERROR,
                    "Failed to install native library to standard location: " + e.getMessage());
            return null;
        }
    }

    /**
     * Check if the native library exists in the standard location.
     * 
     * @return true if the library exists at ~/.gollek/libs/gguf_bridge/1.0.0/
     */
    public static boolean isInstalledInStandardLocation() {
        return Files.exists(getStandardLibPath());
    }

    private static Path getStandardLibPath() {
        return Path.of(System.getProperty("user.home"),
                ".gollek", "libs", LIBRARY_NAME, LIBRARY_VERSION, getNativeLibFileName());
    }

    private static String getNativeLibFileName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("mac") || os.contains("darwin")) return "lib" + LIBRARY_NAME + ".dylib";
        if (os.contains("win")) return LIBRARY_NAME + ".dll";
        return "lib" + LIBRARY_NAME + ".so";
    }

    private static String sha256(byte[] data) {
        try {
            byte[] hash = MessageDigest.getInstance("SHA-256").digest(data);
            return HexFormat.of().formatHex(hash);
        } catch (Exception e) { return "unknown"; }
    }

    private static void clearMacQuarantine(Path dir) {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (!(os.contains("mac") || os.contains("darwin"))) return;
        try {
            Path xattr = Path.of("/usr/bin/xattr");
            if (Files.isExecutable(xattr)) {
                new ProcessBuilder(xattr.toString(), "-dr",
                        "com.apple.quarantine", dir.toAbsolutePath().toString())
                        .redirectErrorStream(true).start().waitFor();
            }
        } catch (Exception ignored) { }
    }

    private static boolean isWindows() {
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("win");
    }

    // Prevent instantiation
    private GGUFNative() {
        throw new UnsupportedOperationException("Utility class");
    }

    public static boolean isAvailable() {
        return AVAILABLE;
    }

    public static String getUnavailableReason() {
        Throwable t = LOAD_ERROR;
        return t == null ? "" : t.getClass().getSimpleName() + ": " + t.getMessage();
    }

    private static void ensureAvailable() {
        if (!AVAILABLE) {
            throw new IllegalStateException("Native GGUF bridge is unavailable: " + getUnavailableReason(), LOAD_ERROR);
        }
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
            ensureAvailable();
            MemorySegment result = (MemorySegment) getVersionHandle().invoke();
            return result.reinterpret(Long.MAX_VALUE).getString(0L);
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
        if (!AVAILABLE) {
            return "GGUF native bridge unavailable: " + getUnavailableReason();
        }
        try {
            MemorySegment result = (MemorySegment) getLastErrorHandle().invoke();
            if (result.address() == 0) {
                return "";
            }
            return result.reinterpret(Long.MAX_VALUE).getString(0L);
        } catch (Throwable t) {
            return "Failed to get error: " + t.getMessage();
        }
    }

    /**
     * Clear last error.
     */
    public static void clearError() {
        if (!AVAILABLE) {
            return;
        }
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
            ensureAvailable();
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
            ensureAvailable();
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
            ensureAvailable();
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
            ensureAvailable();
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
        if (!AVAILABLE) {
            return;
        }
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
        if (!AVAILABLE) {
            return 0;
        }
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
        if (!AVAILABLE) {
            return -1.0f;
        }
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
        if (!AVAILABLE) {
            return;
        }
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
        if (!AVAILABLE) {
            return null;
        }
        try {
            MemorySegment pathSeg = arena.allocateFrom(path);
            MemorySegment result = (MemorySegment) getDetectFormatHandle().invoke(pathSeg);
            if (result.address() == 0) {
                return null;
            }
            return result.reinterpret(Long.MAX_VALUE).getString(0L);
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
        if (!AVAILABLE) {
            return new String[0];
        }
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
                result[i] = ptr.reinterpret(Long.MAX_VALUE).getString(0L);
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
        if (!AVAILABLE) {
            return -99;
        }
        try {
            MemorySegment pathSeg = arena.allocateFrom(path);
            return (int) getVerifyFileHandle().invoke(pathSeg, info);
        } catch (Throwable t) {
            return -99;
        }
    }
}
