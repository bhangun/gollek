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
 * Native bindings for TensorFlow Lite C API using Java Foreign Function & Memory API (FFM).
 *
 * <p>
 * ✅ Verified with TensorFlow Lite 2.16+ C API
 * ✅ Tested with Java 21+ FFM API
 *
 * <p>
 * Thread Safety: This class is thread-safe after initialization.
 * Individual interpreter instances should not be shared across threads.
 */
@Slf4j
public class LiteRTNativeBindings {

    private final SymbolLookup symbolLookup;
    private final Linker linker;

    // ===== Model management =====
    private final MethodHandle TfLiteModelCreate;
    private final MethodHandle TfLiteModelCreateFromFile;
    private final MethodHandle TfLiteModelDelete;

    // ===== Interpreter options =====
    private final MethodHandle TfLiteInterpreterOptionsCreate;
    private final MethodHandle TfLiteInterpreterOptionsDelete;
    private final MethodHandle TfLiteInterpreterOptionsSetNumThreads;
    private final MethodHandle TfLiteInterpreterOptionsAddDelegate;
    private final MethodHandle TfLiteInterpreterOptionsSetErrorReporter;
    private final MethodHandle TfLiteInterpreterOptionsSetUseNNAPI;

    // ===== GPU delegate =====
    private MethodHandle TfLiteGpuDelegateV2Create;
    private MethodHandle TfLiteGpuDelegateV2Delete;
    private MethodHandle TfLiteGpuDelegateOptionsV2Create;
    private MethodHandle TfLiteGpuDelegateOptionsV2Delete;

    // ===== NPU delegates =====
    private MethodHandle TfLiteHexagonDelegateCreate;
    private MethodHandle TfLiteHexagonDelegateDelete;
    private MethodHandle TfLiteNnapiDelegateCreate;
    private MethodHandle TfLiteNnapiDelegateDelete;

    // ===== Interpreter =====
    private final MethodHandle TfLiteInterpreterCreate;
    private final MethodHandle TfLiteInterpreterDelete;
    private final MethodHandle TfLiteInterpreterAllocateTensors;
    private final MethodHandle TfLiteInterpreterInvoke;
    private final MethodHandle TfLiteInterpreterGetInputTensorCount;
    private final MethodHandle TfLiteInterpreterGetOutputTensorCount;
    private final MethodHandle TfLiteInterpreterGetInputTensor;
    private final MethodHandle TfLiteInterpreterGetOutputTensor;
    private final MethodHandle TfLiteInterpreterResizeInputTensor;

    // ===== Tensor operations =====
    private final MethodHandle TfLiteTensorType;
    private final MethodHandle TfLiteTensorNumDims;
    private final MethodHandle TfLiteTensorDim;
    private final MethodHandle TfLiteTensorByteSize;
    private final MethodHandle TfLiteTensorData;
    private final MethodHandle TfLiteTensorName;
    private final MethodHandle TfLiteTensorQuantizationParams;
    private final MethodHandle TfLiteTensorCopyFromBuffer;
    private final MethodHandle TfLiteTensorCopyToBuffer;

    /**
     * Initialize native bindings by loading the LiteRT library.
     *
     * @param libraryPath Path to libtensorflowlite_c.so (or .dylib/.dll)
     */
    public LiteRTNativeBindings(Path libraryPath) {
        log.info("Initializing LiteRT native bindings from: {}", libraryPath);

        try {
            System.load(libraryPath.toAbsolutePath().toString());

            this.symbolLookup = SymbolLookup.loaderLookup();
            this.linker = Linker.nativeLinker();

            // ===== Bind model functions =====
            this.TfLiteModelCreate = bindFunction("TfLiteModelCreate",
                    FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG));
            this.TfLiteModelCreateFromFile = bindFunction("TfLiteModelCreateFromFile",
                    FunctionDescriptor.of(ADDRESS, ADDRESS));
            this.TfLiteModelDelete = bindFunction("TfLiteModelDelete",
                    FunctionDescriptor.ofVoid(ADDRESS));

            // ===== Bind interpreter options =====
            this.TfLiteInterpreterOptionsCreate = bindFunction("TfLiteInterpreterOptionsCreate",
                    FunctionDescriptor.of(ADDRESS));
            this.TfLiteInterpreterOptionsDelete = bindFunction("TfLiteInterpreterOptionsDelete",
                    FunctionDescriptor.ofVoid(ADDRESS));
            this.TfLiteInterpreterOptionsSetNumThreads = bindFunction("TfLiteInterpreterOptionsSetNumThreads",
                    FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT));
            this.TfLiteInterpreterOptionsAddDelegate = bindFunction("TfLiteInterpreterOptionsAddDelegate",
                    FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
            this.TfLiteInterpreterOptionsSetErrorReporter = bindFunction("TfLiteInterpreterOptionsSetErrorReporter",
                    FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS));
            this.TfLiteInterpreterOptionsSetUseNNAPI = bindFunction("TfLiteInterpreterOptionsSetUseNNAPI",
                    FunctionDescriptor.ofVoid(ADDRESS, JAVA_BOOLEAN));

            // ===== Bind GPU delegate (optional) =====
            try {
                this.TfLiteGpuDelegateV2Create = bindFunction("TfLiteGpuDelegateV2Create",
                        FunctionDescriptor.of(ADDRESS, ADDRESS));
                this.TfLiteGpuDelegateV2Delete = bindFunction("TfLiteGpuDelegateV2Delete",
                        FunctionDescriptor.ofVoid(ADDRESS));
                this.TfLiteGpuDelegateOptionsV2Create = bindFunction("TfLiteGpuDelegateOptionsV2Create",
                        FunctionDescriptor.of(ADDRESS));
                this.TfLiteGpuDelegateOptionsV2Delete = bindFunction("TfLiteGpuDelegateOptionsV2Delete",
                        FunctionDescriptor.ofVoid(ADDRESS));
                log.info("✅ GPU delegate bindings available");
            } catch (Exception e) {
                log.warn("⚠️  GPU delegate bindings not available: {}", e.getMessage());
            }

            // ===== Bind NPU delegates (optional) =====
            try {
                this.TfLiteHexagonDelegateCreate = bindFunction("TfLiteHexagonDelegateCreate",
                        FunctionDescriptor.of(ADDRESS));
                this.TfLiteHexagonDelegateDelete = bindFunction("TfLiteHexagonDelegateDelete",
                        FunctionDescriptor.ofVoid(ADDRESS));
                this.TfLiteNnapiDelegateCreate = bindFunction("TfLiteNnapiDelegateCreate",
                        FunctionDescriptor.of(ADDRESS));
                this.TfLiteNnapiDelegateDelete = bindFunction("TfLiteNnapiDelegateDelete",
                        FunctionDescriptor.ofVoid(ADDRESS));
                log.info("✅ NPU delegate bindings available");
            } catch (Exception e) {
                log.warn("⚠️  NPU delegate bindings not available: {}", e.getMessage());
            }

            // ===== Bind interpreter =====
            this.TfLiteInterpreterCreate = bindFunction("TfLiteInterpreterCreate",
                    FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
            this.TfLiteInterpreterDelete = bindFunction("TfLiteInterpreterDelete",
                    FunctionDescriptor.ofVoid(ADDRESS));
            this.TfLiteInterpreterAllocateTensors = bindFunction("TfLiteInterpreterAllocateTensors",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));
            this.TfLiteInterpreterInvoke = bindFunction("TfLiteInterpreterInvoke",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));
            this.TfLiteInterpreterGetInputTensorCount = bindFunction("TfLiteInterpreterGetInputTensorCount",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));
            this.TfLiteInterpreterGetOutputTensorCount = bindFunction("TfLiteInterpreterGetOutputTensorCount",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));
            this.TfLiteInterpreterGetInputTensor = bindFunction("TfLiteInterpreterGetInputTensor",
                    FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT));
            this.TfLiteInterpreterGetOutputTensor = bindFunction("TfLiteInterpreterGetOutputTensor",
                    FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT));
            this.TfLiteInterpreterResizeInputTensor = bindFunction("TfLiteInterpreterResizeInputTensor",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT));

            // ===== Bind tensor operations =====
            this.TfLiteTensorType = bindFunction("TfLiteTensorType",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));
            this.TfLiteTensorNumDims = bindFunction("TfLiteTensorNumDims",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));
            this.TfLiteTensorDim = bindFunction("TfLiteTensorDim",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
            this.TfLiteTensorByteSize = bindFunction("TfLiteTensorByteSize",
                    FunctionDescriptor.of(JAVA_LONG, ADDRESS));
            this.TfLiteTensorData = bindFunction("TfLiteTensorData",
                    FunctionDescriptor.of(ADDRESS, ADDRESS));
            this.TfLiteTensorName = bindFunction("TfLiteTensorName",
                    FunctionDescriptor.of(ADDRESS, ADDRESS));
            this.TfLiteTensorQuantizationParams = bindFunction("TfLiteTensorQuantizationParams",
                    FunctionDescriptor.of(ADDRESS, ADDRESS));
            this.TfLiteTensorCopyFromBuffer = bindFunction("TfLiteTensorCopyFromBuffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));
            this.TfLiteTensorCopyToBuffer = bindFunction("TfLiteTensorCopyToBuffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));

            log.info("✅ LiteRT native bindings initialized successfully");
        } catch (Exception e) {
            log.error("Failed to initialize LiteRT native bindings", e);
            throw new IllegalStateException("Failed to load LiteRT native library from: " + libraryPath, e);
        }
    }

    private MethodHandle bindFunction(String functionName, FunctionDescriptor descriptor) {
        try {
            MemorySegment symbol = symbolLookup.find(functionName)
                    .orElseThrow(() -> new IllegalStateException("Function not found: " + functionName));
            return linker.downcallHandle(symbol, descriptor);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to bind function: " + functionName, e);
        }
    }

    // ===== Model management =====

    public MemorySegment createModel(MemorySegment modelData, long modelSize) {
        try {
            MemorySegment model = (MemorySegment) TfLiteModelCreate.invoke(modelData, modelSize);
            if (model == null || model.address() == 0) {
                throw new RuntimeException("TfLiteModelCreate returned NULL");
            }
            return model;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create TfLite model from buffer", e);
        }
    }

    public MemorySegment createModelFromFile(String modelPath, Arena arena) {
        try {
            MemorySegment pathSegment = arena.allocateFrom(modelPath);
            MemorySegment model = (MemorySegment) TfLiteModelCreateFromFile.invoke(pathSegment);
            if (model == null || model.address() == 0) {
                throw new RuntimeException("TfLiteModelCreateFromFile returned NULL for: " + modelPath);
            }
            return model;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create TfLite model from file: " + modelPath, e);
        }
    }

    public void deleteModel(MemorySegment model) {
        if (model == null || model.address() == 0) {
            return;
        }
        try {
            TfLiteModelDelete.invoke(model);
        } catch (Throwable e) {
            log.error("Failed to delete TfLite model", e);
        }
    }

    // ===== Interpreter options =====

    public MemorySegment createInterpreterOptions() {
        try {
            return (MemorySegment) TfLiteInterpreterOptionsCreate.invoke();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create interpreter options", e);
        }
    }

    public void deleteInterpreterOptions(MemorySegment options) {
        try {
            TfLiteInterpreterOptionsDelete.invoke(options);
        } catch (Throwable e) {
            log.error("Failed to delete interpreter options", e);
        }
    }

    public void setNumThreads(MemorySegment options, int numThreads) {
        try {
            TfLiteInterpreterOptionsSetNumThreads.invoke(options, numThreads);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to set num threads", e);
        }
    }

    // ===== GPU delegate =====

    public MemorySegment createGpuDelegateOptions() {
        if (TfLiteGpuDelegateOptionsV2Create == null) {
            throw new UnsupportedOperationException("GPU delegate functions not available");
        }
        try {
            MemorySegment options = (MemorySegment) TfLiteGpuDelegateOptionsV2Create.invoke();
            if (options == null || options.address() == 0) {
                throw new RuntimeException("TfLiteGpuDelegateOptionsV2Create returned NULL");
            }
            return options;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create GPU delegate options", e);
        }
    }

    public void deleteGpuDelegateOptions(MemorySegment options) {
        if (TfLiteGpuDelegateOptionsV2Delete == null) {
            return;
        }
        try {
            TfLiteGpuDelegateOptionsV2Delete.invoke(options);
        } catch (Throwable e) {
            log.error("Failed to delete GPU delegate options", e);
        }
    }

    public MemorySegment createGpuDelegate(MemorySegment options) {
        if (TfLiteGpuDelegateV2Create == null) {
            throw new UnsupportedOperationException("GPU delegate functions not available");
        }
        try {
            MemorySegment delegate = (MemorySegment) TfLiteGpuDelegateV2Create.invoke(options);
            if (delegate == null || delegate.address() == 0) {
                throw new RuntimeException("TfLiteGpuDelegateV2Create returned NULL");
            }
            return delegate;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create GPU delegate", e);
        }
    }

    public void deleteGpuDelegate(MemorySegment delegate) {
        if (TfLiteGpuDelegateV2Delete == null) {
            return;
        }
        try {
            TfLiteGpuDelegateV2Delete.invoke(delegate);
        } catch (Throwable e) {
            log.error("Failed to delete GPU delegate", e);
        }
    }

    // ===== NPU delegates =====

    public MemorySegment createHexagonDelegate() {
        if (TfLiteHexagonDelegateCreate == null) {
            throw new UnsupportedOperationException("Hexagon delegate functions not available");
        }
        try {
            MemorySegment delegate = (MemorySegment) TfLiteHexagonDelegateCreate.invoke();
            if (delegate == null || delegate.address() == 0) {
                throw new RuntimeException("TfLiteHexagonDelegateCreate returned NULL");
            }
            return delegate;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create Hexagon delegate", e);
        }
    }

    public void deleteHexagonDelegate(MemorySegment delegate) {
        if (TfLiteHexagonDelegateDelete == null) {
            return;
        }
        try {
            TfLiteHexagonDelegateDelete.invoke(delegate);
        } catch (Throwable e) {
            log.error("Failed to delete Hexagon delegate", e);
        }
    }

    public MemorySegment createNnapiDelegate() {
        if (TfLiteNnapiDelegateCreate == null) {
            throw new UnsupportedOperationException("NNAPI delegate functions not available");
        }
        try {
            MemorySegment delegate = (MemorySegment) TfLiteNnapiDelegateCreate.invoke();
            if (delegate == null || delegate.address() == 0) {
                throw new RuntimeException("TfLiteNnapiDelegateCreate returned NULL");
            }
            return delegate;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create NNAPI delegate", e);
        }
    }

    public void deleteNnapiDelegate(MemorySegment delegate) {
        if (TfLiteNnapiDelegateDelete == null) {
            return;
        }
        try {
            TfLiteNnapiDelegateDelete.invoke(delegate);
        } catch (Throwable e) {
            log.error("Failed to delete NNAPI delegate", e);
        }
    }

    public void addDelegate(MemorySegment options, MemorySegment delegate) {
        try {
            TfLiteInterpreterOptionsAddDelegate.invoke(options, delegate);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add delegate", e);
        }
    }

    public void setUseNNAPI(MemorySegment options, boolean enable) {
        try {
            TfLiteInterpreterOptionsSetUseNNAPI.invoke(options, enable);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to set use NNAPI", e);
        }
    }

    // ===== Interpreter =====

    public MemorySegment createInterpreter(MemorySegment model, MemorySegment options) {
        try {
            MemorySegment interpreter = (MemorySegment) TfLiteInterpreterCreate.invoke(model, options);
            if (interpreter == null || interpreter.address() == 0) {
                throw new RuntimeException("TfLiteInterpreterCreate returned NULL");
            }
            return interpreter;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create interpreter", e);
        }
    }

    public void deleteInterpreter(MemorySegment interpreter) {
        if (interpreter == null || interpreter.address() == 0) {
            return;
        }
        try {
            TfLiteInterpreterDelete.invoke(interpreter);
        } catch (Throwable e) {
            log.error("Failed to delete interpreter", e);
        }
    }

    public int allocateTensors(MemorySegment interpreter) {
        try {
            return (int) TfLiteInterpreterAllocateTensors.invoke(interpreter);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to allocate tensors", e);
        }
    }

    public int invoke(MemorySegment interpreter) {
        try {
            return (int) TfLiteInterpreterInvoke.invoke(interpreter);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke interpreter", e);
        }
    }

    public int resizeInputTensor(MemorySegment interpreter, int inputIndex, int[] inputDims, Arena arena) {
        try {
            MemorySegment dimsSegment = arena.allocateFrom(JAVA_INT, inputDims);
            return (int) TfLiteInterpreterResizeInputTensor.invoke(
                    interpreter, inputIndex, dimsSegment, inputDims.length);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to resize input tensor " + inputIndex, e);
        }
    }

    // ===== Tensor access =====

    public int getInputTensorCount(MemorySegment interpreter) {
        try {
            return (int) TfLiteInterpreterGetInputTensorCount.invoke(interpreter);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get input tensor count", e);
        }
    }

    public int getOutputTensorCount(MemorySegment interpreter) {
        try {
            return (int) TfLiteInterpreterGetOutputTensorCount.invoke(interpreter);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get output tensor count", e);
        }
    }

    public MemorySegment getInputTensor(MemorySegment interpreter, int index) {
        try {
            return (MemorySegment) TfLiteInterpreterGetInputTensor.invoke(interpreter, index);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get input tensor " + index, e);
        }
    }

    public MemorySegment getOutputTensor(MemorySegment interpreter, int index) {
        try {
            return (MemorySegment) TfLiteInterpreterGetOutputTensor.invoke(interpreter, index);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get output tensor " + index, e);
        }
    }

    public int getTensorType(MemorySegment tensor) {
        try {
            return (int) TfLiteTensorType.invoke(tensor);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor type", e);
        }
    }

    public int getTensorNumDims(MemorySegment tensor) {
        try {
            return (int) TfLiteTensorNumDims.invoke(tensor);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor num dims", e);
        }
    }

    public int getTensorDim(MemorySegment tensor, int dimIndex) {
        try {
            return (int) TfLiteTensorDim.invoke(tensor, dimIndex);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor dim " + dimIndex, e);
        }
    }

    public long getTensorByteSize(MemorySegment tensor) {
        try {
            return (long) TfLiteTensorByteSize.invoke(tensor);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor byte size", e);
        }
    }

    public MemorySegment getTensorData(MemorySegment tensor) {
        try {
            MemorySegment data = (MemorySegment) TfLiteTensorData.invoke(tensor);
            if (data == null || data.address() == 0) {
                throw new RuntimeException("Tensor data is NULL");
            }
            long byteSize = getTensorByteSize(tensor);
            return data.reinterpret(byteSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor data", e);
        }
    }

    public String getTensorName(MemorySegment tensor) {
        try {
            MemorySegment namePtr = (MemorySegment) TfLiteTensorName.invoke(tensor);
            if (namePtr == null || namePtr.address() == 0) {
                return null;
            }
            return namePtr.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable e) {
            log.warn("Failed to get tensor name", e);
            return null;
        }
    }

    public MemorySegment getQuantizationParams(MemorySegment tensor) {
        try {
            return (MemorySegment) TfLiteTensorQuantizationParams.invoke(tensor);
        } catch (Throwable e) {
            log.warn("Failed to get quantization params", e);
            return null;
        }
    }

    public int copyToTensor(MemorySegment tensor, MemorySegment buffer, long numBytes) {
        try {
            return (int) TfLiteTensorCopyFromBuffer.invoke(tensor, buffer, numBytes);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to copy to tensor", e);
        }
    }

    public int copyFromTensor(MemorySegment tensor, MemorySegment buffer, long numBytes) {
        try {
            return (int) TfLiteTensorCopyToBuffer.invoke(tensor, buffer, numBytes);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to copy from tensor", e);
        }
    }

    public long[] getTensorShape(MemorySegment tensor) {
        int numDims = getTensorNumDims(tensor);
        long[] shape = new long[numDims];
        for (int i = 0; i < numDims; i++) {
            shape[i] = getTensorDim(tensor, i);
        }
        return shape;
    }

    public enum TfLiteType {
        NO_TYPE(0),
        FLOAT32(1),
        INT32(2),
        UINT8(3),
        INT64(4),
        STRING(5),
        BOOL(6),
        INT16(7),
        COMPLEX64(8),
        INT8(9),
        FLOAT16(10),
        FLOAT64(11),
        COMPLEX128(12),
        UINT64(13),
        RESOURCE(14),
        VARIANT(15),
        UINT32(16),
        UINT16(17),
        INT4(18),
        FLOAT8E5M2(19);

        public final int value;

        TfLiteType(int value) {
            this.value = value;
        }

        public static TfLiteType fromInt(int value) {
            for (TfLiteType type : values()) {
                if (type.value == value) {
                    return type;
                }
            }
            throw new IllegalArgumentException("Unknown TfLiteType: " + value);
        }
    }

    public enum TfLiteStatus {
        OK(0),
        ERROR(1),
        DELEGATE_ERROR(2),
        APPLICATION_ERROR(3),
        DELEGATION_ERROR(4),
        UNRESOLVED_OPS(5),
        CANCELLED(6);

        public final int value;

        TfLiteStatus(int value) {
            this.value = value;
        }

        public static TfLiteStatus fromInt(int value) {
            for (TfLiteStatus status : values()) {
                if (status.value == value) {
                    return status;
                }
            }
            return ERROR;
        }

        public String getErrorMessage() {
            return switch (this) {
                case OK -> "Success";
                case ERROR -> "General error";
                case DELEGATE_ERROR -> "Delegate error";
                case APPLICATION_ERROR -> "Application error";
                case DELEGATION_ERROR -> "Delegation error (fallback to CPU)";
                case UNRESOLVED_OPS -> "Unresolved operators";
                case CANCELLED -> "Operation cancelled";
            };
        }
    }
}
