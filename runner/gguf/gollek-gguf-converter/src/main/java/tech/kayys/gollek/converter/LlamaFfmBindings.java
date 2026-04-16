/*
 * Gollek Inference Engine — GGUF Converter Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * LlamaFfmBindings.java
 * ─────────────────────
 * Low-level FFM bindings for llama.cpp model quantization functions.
 *
 * This class provides direct access to llama.cpp's quantization API
 * using JDK 25's Foreign Function & Memory API (JEP 454).
 *
 * Functions exposed:
 * - llama_model_quantize(): Quantize a GGUF model
 * - llama_get_last_error(): Get error message from failed operations
 *
 * Thread Safety:
 * ══════════════
 * All methods are thread-safe. Memory management is handled through
 * Arena scopes to prevent leaks.
 */
package tech.kayys.gollek.converter;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Low-level FFM bindings for llama.cpp quantization functions.
 *
 * <p>
 * This class loads the llama.cpp native library and provides
 * method handles for calling quantization functions directly
 * from Java without JNI overhead.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public final class LlamaFfmBindings {

    private static final String LIB_BASE_NAME = "llama";
    private static final System.Logger LOG = System.getLogger(LlamaFfmBindings.class.getName());

    private static volatile boolean AVAILABLE = false;
    private static volatile Throwable LOAD_ERROR = null;
    private static SymbolLookup LOOKUP;
    private static Linker LINKER;

    // Method handles
    private static volatile MethodHandle llama_model_quantize_handle;
    private static volatile MethodHandle llama_get_last_error_handle;
    private static volatile MethodHandle llama_backend_init_handle;
    private static volatile MethodHandle llama_backend_free_handle;

    static {
        AVAILABLE = false;
        LOAD_ERROR = null;
        LINKER = Linker.nativeLinker();

        // Try to load llama.cpp library
        loadLibrary();

        if (AVAILABLE) {
            initializeFunctionHandles();
        }
    }

    private LlamaFfmBindings() {
        throw new UnsupportedOperationException("Utility class");
    }

    /**
     * Check if llama.cpp FFM bindings are available.
     *
     * @return true if native library is loaded and functions are linked
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Get reason why bindings are unavailable (if applicable).
     *
     * @return error message or empty string if available
     */
    public static String getUnavailableReason() {
        if (AVAILABLE) {
            return "";
        }
        if (LOAD_ERROR == null) {
            return "llama.cpp native library not found";
        }
        return LOAD_ERROR.getClass().getSimpleName() + ": " + LOAD_ERROR.getMessage();
    }

    /**
     * Get last error message from llama.cpp.
     *
     * @return error message string
     */
    public static String getLastError() {
        if (!AVAILABLE || llama_get_last_error_handle == null) {
            return "Native bindings not available";
        }

        try (Arena arena = Arena.ofConfined()) {
            try {
                MemorySegment errorSegment = (MemorySegment) llama_get_last_error_handle.invokeExact();
                if (errorSegment.address() == 0) {
                    return "No error";
                }

                // Read null-terminated string
                return errorSegment.reinterpret(Long.MAX_VALUE).getString(0L);
            } catch (Throwable t) {
                return "Failed to get error message: " + t.getMessage();
            }
        }
    }

    /**
     * Initialize llama.cpp backend (required before quantization).
     */
    public static void initBackend() {
        if (!AVAILABLE || llama_backend_init_handle == null) {
            throw new IllegalStateException("llama.cpp bindings not available");
        }

        try {
            llama_backend_init_handle.invokeExact();
            LOG.log(System.Logger.Level.DEBUG, "llama.cpp backend initialized");
        } catch (Throwable t) {
            throw new RuntimeException("Failed to initialize llama.cpp backend", t);
        }
    }

    /**
     * Free llama.cpp backend resources.
     */
    public static void freeBackend() {
        if (!AVAILABLE || llama_backend_free_handle == null) {
            return;
        }

        try {
            llama_backend_free_handle.invokeExact();
            LOG.log(System.Logger.Level.DEBUG, "llama.cpp backend freed");
        } catch (Throwable t) {
            LOG.log(System.Logger.Level.WARNING, "Failed to free llama.cpp backend: " + t.getMessage());
        }
    }

    /**
     * Quantize a GGUF model file.
     *
     * @param arena      Arena for memory allocation
     * @param inputPath  Path to input GGUF file (f16)
     * @param outputPath Path to output GGUF file (quantized)
     * @param params     Quantization parameters struct
     * @return 0 on success, negative error code on failure
     */
    public static int quantizeModel(
            Arena arena,
            String inputPath,
            String outputPath,
            MemorySegment params) {

        if (!AVAILABLE || llama_model_quantize_handle == null) {
            throw new IllegalStateException("llama.cpp quantization function not available");
        }

        try {
            // Allocate strings in native memory
            MemorySegment inputSegment = arena.allocateFrom(inputPath);
            MemorySegment outputSegment = arena.allocateFrom(outputPath);

            // Call llama_model_quantize
            int result = (int) llama_model_quantize_handle.invokeExact(
                    inputSegment,
                    outputSegment,
                    params);

            if (result != 0) {
                LOG.log(System.Logger.Level.ERROR,
                        "llama_model_quantize failed with code " + result + ": " + getLastError());
            }

            return result;

        } catch (Throwable t) {
            LOG.log(System.Logger.Level.ERROR, "Quantization failed: " + t.getMessage());
            throw new RuntimeException("Quantization failed", t);
        }
    }

    /**
     * Load llama.cpp native library.
     */
    private static void loadLibrary() {
        // Strategy 1: System property for explicit path
        if (!AVAILABLE) {
            try {
                String explicitPath = System.getProperty("gollek.llama.library.path");
                if (explicitPath != null && !explicitPath.isBlank()) {
                    Path libPath = Path.of(explicitPath).toAbsolutePath();
                    if (Files.exists(libPath)) {
                        System.load(libPath.toString());
                        LOOKUP = SymbolLookup.loaderLookup();
                        AVAILABLE = true;
                        LOG.log(System.Logger.Level.INFO,
                                "llama.cpp loaded from explicit path: " + libPath);
                        return;
                    }
                }
            } catch (Throwable ignored) {
            }
        }

        // Strategy 2: Standard location (~/.gollek/libs/llama/)
        if (!AVAILABLE) {
            try {
                Path stdLib = getStandardLibPath();
                if (Files.exists(stdLib)) {
                    System.load(stdLib.toString());
                    LOOKUP = SymbolLookup.loaderLookup();
                    AVAILABLE = true;
                    LOG.log(System.Logger.Level.INFO,
                            "llama.cpp loaded from standard location: " + stdLib);
                    return;
                }
            } catch (Throwable ignored) {
            }
        }

        // Strategy 3: Build directory (development)
        if (!AVAILABLE) {
            try {
                Path buildLib = findBuildDirectoryLib();
                if (buildLib != null && Files.exists(buildLib)) {
                    System.load(buildLib.toString());
                    LOOKUP = SymbolLookup.loaderLookup();
                    AVAILABLE = true;
                    LOG.log(System.Logger.Level.INFO,
                            "llama.cpp loaded from build directory: " + buildLib);
                    return;
                }
            } catch (Throwable ignored) {
            }
        }

        // Strategy 4: System.loadLibrary (uses java.library.path)
        if (!AVAILABLE) {
            try {
                System.loadLibrary(LIB_BASE_NAME);
                LOOKUP = SymbolLookup.loaderLookup();
                AVAILABLE = true;
                LOG.log(System.Logger.Level.INFO, "llama.cpp loaded via System.loadLibrary");
                return;
            } catch (Throwable ignored) {
            }
        }

        // Failed to load
        LOAD_ERROR = new UnsatisfiedLinkError(
                "llama.cpp library '" + LIB_BASE_NAME + "' not found");
        LOG.log(System.Logger.Level.WARNING,
                "llama.cpp native library is unavailable: " + getUnavailableReason());
    }

    /**
     * Initialize function handles.
     */
    private static void initializeFunctionHandles() {
        if (LOOKUP == null) {
            LOOKUP = SymbolLookup.loaderLookup();
        }

        try {
            // llama_model_quantize
            LOOKUP.find("llama_model_quantize").ifPresentOrElse(
                    addr -> {
                        FunctionDescriptor desc = FunctionDescriptor.of(
                                ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS);
                        llama_model_quantize_handle = LINKER.downcallHandle(addr, desc);
                        LOG.log(System.Logger.Level.DEBUG, "Linked llama_model_quantize");
                    },
                    () -> LOG.log(System.Logger.Level.WARNING, "llama_model_quantize not found"));

            // llama_get_last_error
            LOOKUP.find("llama_get_last_error").ifPresentOrElse(
                    addr -> {
                        FunctionDescriptor desc = FunctionDescriptor.of(ValueLayout.ADDRESS);
                        llama_get_last_error_handle = LINKER.downcallHandle(addr, desc);
                        LOG.log(System.Logger.Level.DEBUG, "Linked llama_get_last_error");
                    },
                    () -> LOG.log(System.Logger.Level.WARNING, "llama_get_last_error not found"));

            // llama_backend_init
            LOOKUP.find("llama_backend_init").ifPresentOrElse(
                    addr -> {
                        FunctionDescriptor desc = FunctionDescriptor.ofVoid();
                        llama_backend_init_handle = LINKER.downcallHandle(addr, desc);
                        LOG.log(System.Logger.Level.DEBUG, "Linked llama_backend_init");
                    },
                    () -> LOG.log(System.Logger.Level.DEBUG, "llama_backend_init not found (optional)"));

            // llama_backend_free
            LOOKUP.find("llama_backend_free").ifPresentOrElse(
                    addr -> {
                        FunctionDescriptor desc = FunctionDescriptor.ofVoid();
                        llama_backend_free_handle = LINKER.downcallHandle(addr, desc);
                        LOG.log(System.Logger.Level.DEBUG, "Linked llama_backend_free");
                    },
                    () -> LOG.log(System.Logger.Level.DEBUG, "llama_backend_free not found (optional)"));

        } catch (Exception e) {
            LOG.log(System.Logger.Level.ERROR, "Failed to initialize function handles: " + e.getMessage());
            AVAILABLE = false;
            LOAD_ERROR = e;
        }
    }

    /**
     * Get standard library path (~/.gollek/libs/llama/).
     */
    private static Path getStandardLibPath() {
        String libFileName = getNativeLibFileName();
        return Path.of(System.getProperty("user.home"),
                ".gollek", "libs", "llama", libFileName);
    }

    private static Path findBuildDirectoryLib() {
        String libFileName = getNativeLibFileName();
        String userHome = System.getProperty("user.home");

        String[] devPaths = {
                userHome + "/.gollek/source/vendor/llama.cpp/" + libFileName,
                userHome + "/.gollek/source/vendor/llama.cpp/build/bin/" + libFileName,
                "/opt/homebrew/opt/llama.cpp/lib/" + libFileName,
                "/usr/local/opt/llama.cpp/lib/" + libFileName,
                "gollek/plugins/runner/gguf/gollek-ext-runner-gguf/build/" + libFileName,
                "gollek/plugins/runner/gguf/gollek-ext-runner-gguf/target/" + libFileName,
                "gollek/plugins/runner/gguf/build/" + libFileName,
                "gollek/plugins/runner/build/" + libFileName,
                "build/" + libFileName,
                "target/" + libFileName,
        };

        for (String p : devPaths) {
            Path path = Path.of(p);
            if (Files.isRegularFile(path)) {
                return path;
            }
        }

        return null;
    }

    /**
     * Get native library file name for current platform.
     */
    private static String getNativeLibFileName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("mac") || os.contains("darwin")) {
            return "lib" + LIB_BASE_NAME + ".dylib";
        }
        if (os.contains("win")) {
            return LIB_BASE_NAME + ".dll";
        }
        return "lib" + LIB_BASE_NAME + ".so";
    }
}
