package tech.kayys.gollek.provider.litert;

import lombok.extern.slf4j.Slf4j;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

import static java.lang.foreign.ValueLayout.*;

/**
 * Native bindings for LiteRT 2.0 C API using Java Foreign Function & Memory API (FFM).
 *
 * <p>This class binds the <b>modern LiteRT 2.0 CompiledModel API</b> which uses:
 * <ul>
 *   <li>{@code LiteRtEnvironment} — runtime context with accelerator registry</li>
 *   <li>{@code LiteRtModel} — loaded model representation</li>
 *   <li>{@code LiteRtOptions} — compilation options (CPU/GPU/NPU selection)</li>
 *   <li>{@code LiteRtCompiledModel} — compiled model ready for inference</li>
 *   <li>{@code LiteRtTensorBuffer} — input/output data buffers with lock/unlock</li>
 * </ul>
 *
 * <p>The legacy TfLite Interpreter API symbols exist in the library but are <b>private</b>
 * (local symbols, not dynamically linkable). Only {@code LiteRt*} prefixed symbols are public.
 *
 * @see <a href="https://github.com/google-ai-edge/LiteRT/tree/main/litert/c">LiteRT C API Headers</a>
 */
@Slf4j
public class LiteRTNativeBindings {

    // ========================================================================
    // Status Codes (from litert_common.h)
    // ========================================================================

    public static final int kLiteRtStatusOk = 0;
    public static final int kLiteRtStatusErrorInvalidArgument = 1;
    public static final int kLiteRtStatusErrorMemoryAllocationFailure = 2;
    public static final int kLiteRtStatusErrorRuntimeFailure = 3;
    public static final int kLiteRtStatusErrorMissingInputTensor = 4;
    public static final int kLiteRtStatusErrorUnsupported = 5;
    public static final int kLiteRtStatusErrorNotFound = 6;

    // ========================================================================
    // Hardware Accelerator Constants (from litert_common.h)
    // ========================================================================

    public static final int kLiteRtHwAcceleratorNone = 0;
    public static final int kLiteRtHwAcceleratorCpu = 1;   // 1 << 0
    public static final int kLiteRtHwAcceleratorGpu = 2;   // 1 << 1
    public static final int kLiteRtHwAcceleratorNpu = 4;   // 1 << 2

    // ========================================================================
    // TensorBuffer Lock Modes (from litert_common.h)
    // ========================================================================

    public static final int kLiteRtTensorBufferLockModeRead = 0;
    public static final int kLiteRtTensorBufferLockModeWrite = 1;
    public static final int kLiteRtTensorBufferLockModeReadWrite = 2;

    // ========================================================================
    // Method Handles — Environment
    // ========================================================================

    private final MethodHandle hCreateEnvironment;
    private final MethodHandle hDestroyEnvironment;

    // ========================================================================
    // Method Handles — Model
    // ========================================================================

    private final MethodHandle hCreateModelFromFile;
    private final MethodHandle hCreateModelFromBuffer;
    private final MethodHandle hDestroyModel;

    // ========================================================================
    // Method Handles — Model Introspection
    // ========================================================================

    private final MethodHandle hGetNumModelSubgraphs;
    private final MethodHandle hGetModelSubgraph;
    private final MethodHandle hGetNumModelSignatures;
    private final MethodHandle hGetModelSignature;
    private final MethodHandle hGetMainModelSubgraphIndex;

    // ========================================================================
    // Method Handles — Signature Introspection
    // ========================================================================

    private final MethodHandle hGetSignatureKey;
    private final MethodHandle hGetNumSignatureInputs;
    private final MethodHandle hGetNumSignatureOutputs;
    private final MethodHandle hGetSignatureInputName;
    private final MethodHandle hGetSignatureOutputName;
    private final MethodHandle hGetSignatureInputTensorByIndex;
    private final MethodHandle hGetSignatureOutputTensorByIndex;

    // ========================================================================
    // Method Handles — Subgraph Introspection
    // ========================================================================

    private final MethodHandle hGetNumSubgraphInputs;
    private final MethodHandle hGetNumSubgraphOutputs;
    private final MethodHandle hGetSubgraphInput;
    private final MethodHandle hGetSubgraphOutput;

    // ========================================================================
    // Method Handles — Tensor Introspection
    // ========================================================================

    private final MethodHandle hGetTensorName;
    private final MethodHandle hGetTensorTypeId;
    private final MethodHandle hGetRankedTensorType;

    // ========================================================================
    // Method Handles — Options
    // ========================================================================

    private final MethodHandle hCreateOptions;
    private final MethodHandle hDestroyOptions;
    private final MethodHandle hSetOptionsHardwareAccelerators;

    // ========================================================================
    // Method Handles — CompiledModel
    // ========================================================================

    private final MethodHandle hCreateCompiledModel;
    private final MethodHandle hDestroyCompiledModel;
    private final MethodHandle hRunCompiledModel;
    private final MethodHandle hRunCompiledModelWithOptions;
    private final MethodHandle hCompiledModelIsFullyAccelerated;
    private final MethodHandle hGetCompiledModelInputBufferRequirements;
    private final MethodHandle hGetCompiledModelOutputBufferRequirements;
    private final MethodHandle hGetCompiledModelInputTensorLayout;
    private final MethodHandle hCompiledModelResizeInputTensor;

    // ========================================================================
    // Method Handles — TensorBuffer
    // ========================================================================

    private final MethodHandle hCreateTensorBufferFromHostMemory;
    private MethodHandle hCreateManagedTensorBuffer;
    private MethodHandle hCreateManagedTensorBufferFromRequirements;
    private final MethodHandle hDestroyTensorBuffer;
    private final MethodHandle hLockTensorBuffer;
    private final MethodHandle hUnlockTensorBuffer;
    private final MethodHandle hGetTensorBufferSize;
    private final MethodHandle hGetTensorBufferHostMemory;
    private MethodHandle hGetTensorBufferType;
    private MethodHandle hGetTensorBufferTensorType;

    // ========================================================================
    // Method Handles — TensorBufferRequirements
    // ========================================================================

    private MethodHandle hGetTensorBufferRequirementsBufferSize;
    private MethodHandle hDestroyTensorBufferRequirements;

    // ========================================================================
    // Method Handles — Status / Utility
    // ========================================================================

    private final MethodHandle hGetStatusString;

    private final SymbolLookup symbolLookup;
    private final Linker linker;

    /**
     * Initialize LiteRT 2.0 native bindings from the given library path.
     *
     * @param libraryPath path to libLiteRt.dylib / libLiteRt.so
     */
    public LiteRTNativeBindings(Path libraryPath) {
        log.info("Initializing LiteRT 2.0 native bindings from: {}", libraryPath);

        try {
            Arena globalArena = Arena.global();
            this.symbolLookup = SymbolLookup.libraryLookup(libraryPath, globalArena);
            this.linker = Linker.nativeLinker();

            // ===== Status Utility =====
            this.hGetStatusString = bind("LiteRtGetStatusString",
                    FunctionDescriptor.of(ADDRESS, JAVA_INT));

            // ===== Environment =====
            this.hCreateEnvironment = bind("LiteRtCreateEnvironment",
                    FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));
            this.hDestroyEnvironment = bind("LiteRtDestroyEnvironment",
                    FunctionDescriptor.ofVoid(ADDRESS));

            // ===== Model =====
            this.hCreateModelFromFile = bind("LiteRtCreateModelFromFile",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hCreateModelFromBuffer = bind("LiteRtCreateModelFromBuffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            this.hDestroyModel = bind("LiteRtDestroyModel",
                    FunctionDescriptor.ofVoid(ADDRESS));

            // ===== Model Introspection =====
            this.hGetNumModelSubgraphs = bind("LiteRtGetNumModelSubgraphs",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetModelSubgraph = bind("LiteRtGetModelSubgraph",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            this.hGetNumModelSignatures = bind("LiteRtGetNumModelSignatures",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetModelSignature = bind("LiteRtGetModelSignature",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            this.hGetMainModelSubgraphIndex = bind("LiteRtGetMainModelSubgraphIndex",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

            // ===== Signature Introspection =====
            this.hGetSignatureKey = bind("LiteRtGetSignatureKey",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetNumSignatureInputs = bind("LiteRtGetNumSignatureInputs",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetNumSignatureOutputs = bind("LiteRtGetNumSignatureOutputs",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetSignatureInputName = bind("LiteRtGetSignatureInputName",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            this.hGetSignatureOutputName = bind("LiteRtGetSignatureOutputName",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            this.hGetSignatureInputTensorByIndex = bind("LiteRtGetSignatureInputTensorByIndex",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            this.hGetSignatureOutputTensorByIndex = bind("LiteRtGetSignatureOutputTensorByIndex",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));

            // ===== Subgraph Introspection =====
            this.hGetNumSubgraphInputs = bind("LiteRtGetNumSubgraphInputs",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetNumSubgraphOutputs = bind("LiteRtGetNumSubgraphOutputs",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetSubgraphInput = bind("LiteRtGetSubgraphInput",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            this.hGetSubgraphOutput = bind("LiteRtGetSubgraphOutput",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));

            // ===== Tensor Introspection =====
            this.hGetTensorName = bind("LiteRtGetTensorName",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetTensorTypeId = bind("LiteRtGetTensorTypeId",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetRankedTensorType = bind("LiteRtGetRankedTensorType",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

            // ===== Options =====
            this.hCreateOptions = bind("LiteRtCreateOptions",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));
            this.hDestroyOptions = bind("LiteRtDestroyOptions",
                    FunctionDescriptor.ofVoid(ADDRESS));
            this.hSetOptionsHardwareAccelerators = bind("LiteRtSetOptionsHardwareAccelerators",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));

            // ===== CompiledModel =====
            this.hCreateCompiledModel = bind("LiteRtCreateCompiledModel",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            this.hDestroyCompiledModel = bind("LiteRtDestroyCompiledModel",
                    FunctionDescriptor.ofVoid(ADDRESS));
            this.hRunCompiledModel = bind("LiteRtRunCompiledModel",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
            this.hRunCompiledModelWithOptions = bind("LiteRtRunCompiledModelWithOptions",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS));
            this.hCompiledModelIsFullyAccelerated = bind("LiteRtCompiledModelIsFullyAccelerated",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetCompiledModelInputBufferRequirements = bind("LiteRtGetCompiledModelInputBufferRequirements",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS));
            this.hGetCompiledModelOutputBufferRequirements = bind("LiteRtGetCompiledModelOutputBufferRequirements",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS));
            this.hGetCompiledModelInputTensorLayout = bind("LiteRtGetCompiledModelInputTensorLayout",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS));
            this.hCompiledModelResizeInputTensor = bind("LiteRtCompiledModelResizeInputTensor",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG));

            // ===== TensorBuffer =====
            this.hCreateTensorBufferFromHostMemory = bind("LiteRtCreateTensorBufferFromHostMemory",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS));
            this.hDestroyTensorBuffer = bind("LiteRtDestroyTensorBuffer",
                    FunctionDescriptor.ofVoid(ADDRESS));
            this.hLockTensorBuffer = bind("LiteRtLockTensorBuffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT));
            this.hUnlockTensorBuffer = bind("LiteRtUnlockTensorBuffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));
            this.hGetTensorBufferSize = bind("LiteRtGetTensorBufferSize",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            this.hGetTensorBufferHostMemory = bind("LiteRtGetTensorBufferHostMemory",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

            // ===== Optional TensorBuffer bindings =====
            try {
                this.hCreateManagedTensorBuffer = bind("LiteRtCreateManagedTensorBuffer",
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
                this.hCreateManagedTensorBufferFromRequirements = bind("LiteRtCreateManagedTensorBufferFromRequirements",
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
                this.hGetTensorBufferType = bind("LiteRtGetTensorBufferType",
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
                this.hGetTensorBufferTensorType = bind("LiteRtGetTensorBufferTensorType",
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
                log.info("✅ Managed TensorBuffer bindings available");
            } catch (Exception e) {
                log.warn("⚠️  Managed TensorBuffer bindings not available");
            }

            // ===== Optional TensorBufferRequirements =====
            try {
                this.hGetTensorBufferRequirementsBufferSize = bind("LiteRtGetTensorBufferRequirementsBufferSize",
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
                this.hDestroyTensorBufferRequirements = bind("LiteRtDestroyTensorBufferRequirements",
                        FunctionDescriptor.ofVoid(ADDRESS));
                log.info("✅ TensorBufferRequirements bindings available");
            } catch (Exception e) {
                log.warn("⚠️  TensorBufferRequirements bindings not available");
            }

            log.info("✅ LiteRT 2.0 native bindings initialized successfully");
        } catch (Exception e) {
            log.error("Failed to initialize LiteRT 2.0 native bindings", e);
            throw new IllegalStateException("Failed to load LiteRT library from: " + libraryPath, e);
        }
    }

    private MethodHandle bind(String name, FunctionDescriptor descriptor) {
        return symbolLookup.find(name)
                .map(addr -> linker.downcallHandle(addr, descriptor))
                .orElseThrow(() -> new IllegalStateException(String.format("Symbol not found in LiteRT library: %s. This library might be an older TFLite version or incompatible. Ensure you are using LiteRT 2.0 (libLiteRt.dylib).", name)));
    }

    // ====================================================================
    // PUBLIC API — Status
    // ====================================================================

    /** Returns a human-readable string for the given status code. */
    public String getStatusString(int status) {
        try {
            MemorySegment ptr = (MemorySegment) hGetStatusString.invoke(status);
            return (ptr != null && ptr.address() != 0)
                    ? ptr.reinterpret(256).getString(0)
                    : "unknown(" + status + ")";
        } catch (Throwable e) {
            return "unknown(" + status + ")";
        }
    }

    private void check(int status, String func) {
        if (status != kLiteRtStatusOk) {
            throw new RuntimeException(func + " failed: " + getStatusString(status));
        }
    }

    // ====================================================================
    // PUBLIC API — Environment
    // ====================================================================

    /** Creates a LiteRT environment with default options (CPU). */
    public MemorySegment createEnvironment(Arena arena) {
        return createEnvironment(arena, 0, MemorySegment.NULL);
    }

    /** Creates a LiteRT environment with the specified options. */
    public MemorySegment createEnvironment(Arena arena, int numOptions, MemorySegment options) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hCreateEnvironment.invoke(numOptions, options, out);
            check(status, "LiteRtCreateEnvironment");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create environment", e);
        }
    }

    public void destroyEnvironment(MemorySegment env) {
        if (env == null || env.address() == 0) return;
        try {
            hDestroyEnvironment.invoke(env);
        } catch (Throwable e) {
            log.error("Failed to destroy environment", e);
        }
    }

    // ====================================================================
    // PUBLIC API — Model
    // ====================================================================

    /** Loads a model from a file path. */
    public MemorySegment createModelFromFile(String path, Arena arena) {
        try {
            MemorySegment pathSeg = arena.allocateFrom(path);
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hCreateModelFromFile.invoke(pathSeg, out);
            check(status, "LiteRtCreateModelFromFile");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create model from file: " + path, e);
        }
    }

    /** Loads a model from an in-memory buffer. */
    public MemorySegment createModelFromBuffer(MemorySegment buffer, long size, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hCreateModelFromBuffer.invoke(buffer, size, out);
            check(status, "LiteRtCreateModelFromBuffer");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create model from buffer", e);
        }
    }

    public void destroyModel(MemorySegment model) {
        if (model == null || model.address() == 0) return;
        try {
            hDestroyModel.invoke(model);
        } catch (Throwable e) {
            log.error("Failed to destroy model", e);
        }
    }

    // ====================================================================
    // PUBLIC API — Model Introspection
    // ====================================================================

    public int getNumModelSubgraphs(MemorySegment model, Arena arena) {
        try {
            MemorySegment out = arena.allocate(JAVA_LONG);
            int status = (int) hGetNumModelSubgraphs.invoke(model, out);
            check(status, "LiteRtGetNumModelSubgraphs");
            return (int) out.get(JAVA_LONG, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get subgraph count", e);
        }
    }

    public MemorySegment getModelSubgraph(MemorySegment model, int index, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hGetModelSubgraph.invoke(model, (long) index, out);
            check(status, "LiteRtGetModelSubgraph");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get subgraph at " + index, e);
        }
    }

    public int getNumModelSignatures(MemorySegment model, Arena arena) {
        try {
            MemorySegment out = arena.allocate(JAVA_LONG);
            int status = (int) hGetNumModelSignatures.invoke(model, out);
            check(status, "LiteRtGetNumModelSignatures");
            return (int) out.get(JAVA_LONG, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get signature count", e);
        }
    }

    public MemorySegment getModelSignature(MemorySegment model, int index, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hGetModelSignature.invoke(model, (long) index, out);
            check(status, "LiteRtGetModelSignature");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get signature at " + index, e);
        }
    }

    // ====================================================================
    // PUBLIC API — Signature Introspection
    // ====================================================================

    public String getSignatureKey(MemorySegment signature, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hGetSignatureKey.invoke(signature, out);
            check(status, "LiteRtGetSignatureKey");
            MemorySegment ptr = out.get(ADDRESS, 0);
            return ptr.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get signature key", e);
        }
    }

    public int getNumSignatureInputs(MemorySegment signature, Arena arena) {
        try {
            MemorySegment out = arena.allocate(JAVA_LONG);
            int status = (int) hGetNumSignatureInputs.invoke(signature, out);
            check(status, "LiteRtGetNumSignatureInputs");
            return (int) out.get(JAVA_LONG, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get signature input count", e);
        }
    }

    public int getNumSignatureOutputs(MemorySegment signature, Arena arena) {
        try {
            MemorySegment out = arena.allocate(JAVA_LONG);
            int status = (int) hGetNumSignatureOutputs.invoke(signature, out);
            check(status, "LiteRtGetNumSignatureOutputs");
            return (int) out.get(JAVA_LONG, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get signature output count", e);
        }
    }

    public String getSignatureInputName(MemorySegment signature, int index, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hGetSignatureInputName.invoke(signature, (long) index, out);
            check(status, "LiteRtGetSignatureInputName");
            MemorySegment ptr = out.get(ADDRESS, 0);
            return ptr.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get input name at " + index, e);
        }
    }

    public String getSignatureOutputName(MemorySegment signature, int index, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hGetSignatureOutputName.invoke(signature, (long) index, out);
            check(status, "LiteRtGetSignatureOutputName");
            MemorySegment ptr = out.get(ADDRESS, 0);
            return ptr.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get output name at " + index, e);
        }
    }

    public MemorySegment getSignatureInputTensorByIndex(MemorySegment signature, int index, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hGetSignatureInputTensorByIndex.invoke(signature, (long) index, out);
            check(status, "LiteRtGetSignatureInputTensorByIndex");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get input tensor at " + index, e);
        }
    }

    public MemorySegment getSignatureOutputTensorByIndex(MemorySegment signature, int index, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hGetSignatureOutputTensorByIndex.invoke(signature, (long) index, out);
            check(status, "LiteRtGetSignatureOutputTensorByIndex");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get output tensor at " + index, e);
        }
    }

    // ====================================================================
    // PUBLIC API — Tensor Introspection
    // ====================================================================

    public String getTensorName(MemorySegment tensor, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hGetTensorName.invoke(tensor, out);
            check(status, "LiteRtGetTensorName");
            MemorySegment ptr = out.get(ADDRESS, 0);
            return ptr.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor name", e);
        }
    }

    // ====================================================================
    // PUBLIC API — Options
    // ====================================================================

    /** Creates compilation options. */
    public MemorySegment createOptions(Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hCreateOptions.invoke(out);
            check(status, "LiteRtCreateOptions");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create options", e);
        }
    }

    public void destroyOptions(MemorySegment options) {
        if (options == null || options.address() == 0) return;
        try {
            hDestroyOptions.invoke(options);
        } catch (Throwable e) {
            log.error("Failed to destroy options", e);
        }
    }

    /** Sets the hardware accelerators to use (bitmask of kLiteRtHwAccelerator* values). */
    public void setOptionsHardwareAccelerators(MemorySegment options, int accelerators) {
        try {
            int status = (int) hSetOptionsHardwareAccelerators.invoke(options, accelerators);
            if (status != 0) {
                String error = getStatusString(status);
                log.warn("LiteRtSetOptionsHardwareAccelerators status {}: {}", status, error);
            }
            check(status, "LiteRtSetOptionsHardwareAccelerators");
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to set hardware accelerators: " + e.getMessage(), e);
        }
    }

    // ====================================================================
    // PUBLIC API — CompiledModel
    // ====================================================================

    /** Creates a compiled model ready for inference. */
    public MemorySegment createCompiledModel(MemorySegment env, MemorySegment model,
                                              MemorySegment options, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hCreateCompiledModel.invoke(env, model, options, out);
            check(status, "LiteRtCreateCompiledModel");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create compiled model", e);
        }
    }

    public void destroyCompiledModel(MemorySegment compiledModel) {
        if (compiledModel == null || compiledModel.address() == 0) return;
        try {
            hDestroyCompiledModel.invoke(compiledModel);
        } catch (Throwable e) {
            log.error("Failed to destroy compiled model", e);
        }
    }

    /**
     * Runs inference on the compiled model.
     *
     * @param compiledModel   the compiled model handle
     * @param signatureIndex  signature index (usually 0)
     * @param inputBuffers    array of LiteRtTensorBuffer handles for inputs
     * @param outputBuffers   array of LiteRtTensorBuffer handles for outputs
     */
    public void runCompiledModel(MemorySegment compiledModel, int signatureIndex,
                                  MemorySegment inputBuffers, int numInputs,
                                  MemorySegment outputBuffers, int numOutputs) {
        try {
            int status = (int) hRunCompiledModel.invoke(compiledModel,
                    (long) signatureIndex, (long) numInputs, inputBuffers,
                    (long) numOutputs, outputBuffers);
            check(status, "LiteRtRunCompiledModel");
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to run compiled model", e);
        }
    }

    /** Checks if the compiled model is fully accelerated by hardware. */
    public boolean isFullyAccelerated(MemorySegment compiledModel, Arena arena) {
        try {
            MemorySegment out = arena.allocate(JAVA_INT);
            int status = (int) hCompiledModelIsFullyAccelerated.invoke(compiledModel, out);
            if (status != kLiteRtStatusOk) return false;
            return out.get(JAVA_INT, 0) != 0;
        } catch (Throwable e) {
            return false;
        }
    }

    /** Gets the buffer requirements for the given input tensor of the compiled model. */
    public MemorySegment getCompiledModelInputBufferRequirements(MemorySegment compiledModel,
                                                                  int signatureIndex, int inputIndex,
                                                                  Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hGetCompiledModelInputBufferRequirements.invoke(
                    compiledModel, (long) signatureIndex, (long) inputIndex, out);
            check(status, "LiteRtGetCompiledModelInputBufferRequirements");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get input buffer requirements", e);
        }
    }

    /** Gets the buffer requirements for the given output tensor of the compiled model. */
    public MemorySegment getCompiledModelOutputBufferRequirements(MemorySegment compiledModel,
                                                                   int signatureIndex, int outputIndex,
                                                                   Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hGetCompiledModelOutputBufferRequirements.invoke(
                    compiledModel, (long) signatureIndex, (long) outputIndex, out);
            check(status, "LiteRtGetCompiledModelOutputBufferRequirements");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get output buffer requirements", e);
        }
    }

    /** Resizes the input tensor to support dynamic shapes. */
    public void resizeInputTensor(MemorySegment compiledModel, int signatureIndex,
                                   int inputIndex, int[] dims, Arena arena) {
        try {
            MemorySegment dimsSeg = arena.allocateFrom(JAVA_INT, dims);
            int status = (int) hCompiledModelResizeInputTensor.invoke(
                    compiledModel, (long) signatureIndex, (long) inputIndex,
                    dimsSeg, (long) dims.length);
            check(status, "LiteRtCompiledModelResizeInputTensor");
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to resize input tensor", e);
        }
    }

    // ====================================================================
    // PUBLIC API — TensorBuffer
    // ====================================================================

    /** Creates a tensor buffer wrapping existing host memory. */
    public MemorySegment createTensorBufferFromHostMemory(MemorySegment tensorType,
                                                           MemorySegment hostBuffer,
                                                           long bufferSize, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hCreateTensorBufferFromHostMemory.invoke(
                    tensorType, hostBuffer, bufferSize, MemorySegment.NULL, out);
            check(status, "LiteRtCreateTensorBufferFromHostMemory");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create tensor buffer from host memory", e);
        }
    }

    /** Creates a managed tensor buffer from buffer requirements. */
    public MemorySegment createManagedTensorBufferFromRequirements(MemorySegment env,
                                                                     MemorySegment tensorType,
                                                                     MemorySegment requirements,
                                                                     Arena arena) {
        if (hCreateManagedTensorBufferFromRequirements == null) {
            throw new UnsupportedOperationException("Managed TensorBuffer from requirements not available");
        }
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hCreateManagedTensorBufferFromRequirements.invoke(
                    env, tensorType, requirements, out);
            check(status, "LiteRtCreateManagedTensorBufferFromRequirements");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create managed tensor buffer", e);
        }
    }

    public void destroyTensorBuffer(MemorySegment buffer) {
        if (buffer == null || buffer.address() == 0) return;
        try {
            hDestroyTensorBuffer.invoke(buffer);
        } catch (Throwable e) {
            log.error("Failed to destroy tensor buffer", e);
        }
    }

    /** Locks a tensor buffer and returns the host memory address. */
    public MemorySegment lockTensorBuffer(MemorySegment buffer, int lockMode, Arena arena) {
        try {
            MemorySegment out = arena.allocate(ADDRESS);
            int status = (int) hLockTensorBuffer.invoke(buffer, out, lockMode);
            check(status, "LiteRtLockTensorBuffer");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to lock tensor buffer", e);
        }
    }

    /** Unlocks a previously locked tensor buffer. */
    public void unlockTensorBuffer(MemorySegment buffer) {
        try {
            int status = (int) hUnlockTensorBuffer.invoke(buffer);
            check(status, "LiteRtUnlockTensorBuffer");
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to unlock tensor buffer", e);
        }
    }

    /** Gets the size of a tensor buffer. */
    public long getTensorBufferSize(MemorySegment buffer, Arena arena) {
        try {
            MemorySegment out = arena.allocate(JAVA_LONG);
            int status = (int) hGetTensorBufferSize.invoke(buffer, out);
            check(status, "LiteRtGetTensorBufferSize");
            return out.get(JAVA_LONG, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor buffer size", e);
        }
    }

    // ====================================================================
    // Enums
    // ====================================================================

    /** LiteRT tensor element types. */
    public enum LitertType {
        NO_TYPE(0), FLOAT32(1), INT32(2), UINT8(3), INT64(4), STRING(5), BOOL(6), INT16(7),
        COMPLEX64(8), INT8(9), FLOAT16(10), FLOAT64(11), COMPLEX128(12), UINT64(13),
        RESOURCE(14), VARIANT(15), UINT32(16), UINT16(17), INT4(18), FLOAT8E5M2(19);

        public final int value;

        LitertType(int value) {
            this.value = value;
        }

        public static LitertType fromInt(int value) {
            for (LitertType type : values()) {
                if (type.value == value) return type;
            }
            return NO_TYPE;
        }
    }

    /** LiteRT status codes. */
    public enum LitertStatus {
        OK(0, "Success"),
        ERROR_INVALID_ARGUMENT(1, "Invalid argument"),
        ERROR_MEMORY_ALLOCATION(2, "Memory allocation failure"),
        ERROR_RUNTIME_FAILURE(3, "Runtime failure"),
        ERROR_MISSING_INPUT(4, "Missing input tensor"),
        ERROR_UNSUPPORTED(5, "Unsupported"),
        ERROR_NOT_FOUND(6, "Not found"),
        ERROR_TIMEOUT(7, "Timeout expired"),
        ERROR_WRONG_VERSION(8, "Wrong version"),
        ERROR_UNKNOWN(9, "Unknown error");

        public final int value;
        private final String message;

        LitertStatus(int value, String message) {
            this.value = value;
            this.message = message;
        }

        public String getErrorMessage() {
            return message;
        }

        public static LitertStatus fromInt(int value) {
            for (LitertStatus s : values()) {
                if (s.value == value) return s;
            }
            return ERROR_UNKNOWN;
        }
    }
}
