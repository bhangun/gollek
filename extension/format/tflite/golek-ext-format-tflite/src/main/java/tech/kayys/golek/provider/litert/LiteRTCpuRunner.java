package tech.kayys.golek.provider.litert;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.spi.exception.InferenceException;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * LiteRT CPU runner backed by TensorFlow Lite C API.
 *
 * This runner executes inference with native bindings and returns
 * a JSON-encoded content payload plus structured metadata.
 */
@Slf4j
public class LiteRTCpuRunner implements AutoCloseable {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private LiteRTNativeBindings nativeBindings;
    private LiteRTDelegateManager delegateManager;

    private Arena arena;
    private MemorySegment model;
    private MemorySegment interpreter;

    private boolean initialized = false;
    private int numThreads = 4;
    private boolean useGpu = false;
    private boolean useNpu = false;
    private String gpuBackend = "auto";
    private String npuType = "auto";

    private final Map<Integer, TensorInfo> inputTensors = new HashMap<>();
    private final Map<Integer, TensorInfo> outputTensors = new HashMap<>();

    private final AtomicLong totalInferences = new AtomicLong(0);
    private final AtomicLong failedInferences = new AtomicLong(0);
    private final AtomicLong totalLatencyMs = new AtomicLong(0);

    public void initialize(Path modelPath, LiteRTRunnerConfig config) {
        if (initialized) {
            return;
        }

        try {
            this.numThreads = config.numThreads();
            this.useGpu = config.useGpu();
            this.useNpu = config.useNpu();
            this.gpuBackend = config.gpuBackend();
            this.npuType = config.npuType();

            this.arena = Arena.ofConfined();
            String libraryPath = findLiteRTLibrary();
            this.nativeBindings = new LiteRTNativeBindings(Paths.get(libraryPath));
            this.delegateManager = new LiteRTDelegateManager(nativeBindings, arena);

            if (!Files.exists(modelPath)) {
                throw new InferenceException(
                        ErrorCode.INIT_MODEL_LOAD_FAILED,
                        "Model file not found: " + modelPath);
            }

            this.model = nativeBindings.createModelFromFile(modelPath.toString(), arena);

            MemorySegment options = nativeBindings.createInterpreterOptions();
            nativeBindings.setNumThreads(options, numThreads);

            initializeDelegates(options);

            this.interpreter = nativeBindings.createInterpreter(model, options);
            nativeBindings.deleteInterpreterOptions(options);

            int status = nativeBindings.allocateTensors(interpreter);
            if (status != LiteRTNativeBindings.TfLiteStatus.OK.value) {
                throw new InferenceException(ErrorCode.INIT_RUNNER_FAILED,
                        "Tensor allocation failed with status: " + status);
            }

            inspectTensors();
            initialized = true;
            log.info("✅ LiteRT CPU runner initialized (inputs: {}, outputs: {})",
                    inputTensors.size(), outputTensors.size());
        } catch (Exception e) {
            cleanup();
            throw e instanceof InferenceException
                    ? (InferenceException) e
                    : new InferenceException(ErrorCode.INIT_RUNNER_FAILED, "LiteRT init failed", e);
        }
    }

    public InferenceResponse infer(InferenceRequest request) {
        if (!initialized) {
            throw new InferenceException(ErrorCode.RUNTIME_INVALID_STATE, "Runner not initialized");
        }

        long start = System.currentTimeMillis();

        try {
            Map<String, TensorData> inputs = resolveInputs(request);
            Map<String, TensorData> outputs = runInference(inputs);

            long latencyMs = System.currentTimeMillis() - start;
            totalInferences.incrementAndGet();
            totalLatencyMs.addAndGet(latencyMs);

            String content = serializeOutputs(outputs);
            Map<String, Object> metadata = Map.of(
                    "runner", "litert-cpu",
                    "outputs", serializeOutputsForMetadata(outputs));

            return InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .model(request.getModel())
                    .durationMs(latencyMs)
                    .content(content)
                    .metadata(metadata)
                    .build();
        } catch (Exception e) {
            failedInferences.incrementAndGet();
            if (e instanceof InferenceException ie) {
                throw ie;
            }
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED, "Inference failed", e);
        }
    }

    public CompletableFuture<InferenceResponse> inferAsync(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> infer(request));
    }

    public boolean health() {
        return initialized && interpreter != null && interpreter.address() != 0;
    }

    @Override
    public void close() {
        cleanup();
    }

    // ===== Internal helpers =====

    private void initializeDelegates(MemorySegment options) {
        if (!useGpu && !useNpu) {
            return;
        }

        delegateManager.autoDetectAndInitializeDelegates();

        if (useGpu) {
            tryInitializeGpuDelegate(options);
        }

        if (useNpu) {
            tryInitializeNpuDelegate(options);
        }
    }

    private void tryInitializeGpuDelegate(MemorySegment options) {
        try {
            LiteRTDelegateManager.GpuBackend backend = switch (gpuBackend.toLowerCase()) {
                case "opencl" -> LiteRTDelegateManager.GpuBackend.OPENCL;
                case "vulkan" -> LiteRTDelegateManager.GpuBackend.VULKAN;
                case "metal" -> LiteRTDelegateManager.GpuBackend.METAL;
                default -> LiteRTDelegateManager.GpuBackend.OPENCL;
            };
            boolean ok = delegateManager.tryInitializeGpuDelegate(backend, backend.name());
            if (ok) {
                LiteRTDelegateManager.DelegateType type = delegateManager.getBestAvailableDelegate();
                if (type != null) {
                    delegateManager.addDelegateToOptions(options, type);
                }
            }
        } catch (Exception e) {
            log.warn("GPU delegate initialization failed: {}", e.getMessage());
        }
    }

    private void tryInitializeNpuDelegate(MemorySegment options) {
        try {
            LiteRTDelegateManager.NpuType type = switch (npuType.toLowerCase()) {
                case "hexagon" -> LiteRTDelegateManager.NpuType.HEXAGON;
                case "nnapi" -> LiteRTDelegateManager.NpuType.NNAPI;
                case "ethos" -> LiteRTDelegateManager.NpuType.ETHOS;
                case "neuron" -> LiteRTDelegateManager.NpuType.NEURON;
                default -> LiteRTDelegateManager.NpuType.NNAPI;
            };
            boolean ok = delegateManager.tryInitializeNpuDelegate(type, type.name());
            if (ok) {
                LiteRTDelegateManager.DelegateType best = delegateManager.getBestAvailableDelegate();
                if (best != null) {
                    delegateManager.addDelegateToOptions(options, best);
                }
            }
        } catch (Exception e) {
            log.warn("NPU delegate initialization failed: {}", e.getMessage());
        }
    }

    private Map<String, TensorData> runInference(Map<String, TensorData> inputs) {
        try (Arena inferenceArena = Arena.ofConfined()) {
            for (int i = 0; i < inputTensors.size(); i++) {
                TensorInfo info = inputTensors.get(i);
                TensorData input = findInputForIndex(i, info, inputs);
                validateInput(info, input);

                byte[] bytes = TensorConverter.toNativeBytes(input);
                MemorySegment buffer = inferenceArena.allocateFrom(ValueLayout.JAVA_BYTE, bytes);
                MemorySegment tensor = nativeBindings.getInputTensor(interpreter, i);
                nativeBindings.copyToTensor(tensor, buffer, bytes.length);
            }

            int status = nativeBindings.invoke(interpreter);
            if (status != LiteRTNativeBindings.TfLiteStatus.OK.value) {
                throw new InferenceException(
                        ErrorCode.RUNTIME_INFERENCE_FAILED,
                        "Inference failed: " + LiteRTNativeBindings.TfLiteStatus.fromInt(status).getErrorMessage());
            }

            Map<String, TensorData> outputs = new LinkedHashMap<>();
            for (int i = 0; i < outputTensors.size(); i++) {
                TensorInfo info = outputTensors.get(i);
                MemorySegment tensor = nativeBindings.getOutputTensor(interpreter, i);
                TensorData output = readTensor(info, tensor);
                outputs.put(info.name != null ? info.name : "output_" + i, output);
            }

            return outputs;
        }
    }

    private void inspectTensors() {
        inputTensors.clear();
        outputTensors.clear();

        int inputCount = nativeBindings.getInputTensorCount(interpreter);
        for (int i = 0; i < inputCount; i++) {
            MemorySegment tensor = nativeBindings.getInputTensor(interpreter, i);
            TensorInfo info = inspectTensor(tensor);
            inputTensors.put(i, info);
        }

        int outputCount = nativeBindings.getOutputTensorCount(interpreter);
        for (int i = 0; i < outputCount; i++) {
            MemorySegment tensor = nativeBindings.getOutputTensor(interpreter, i);
            TensorInfo info = inspectTensor(tensor);
            outputTensors.put(i, info);
        }
    }

    private TensorInfo inspectTensor(MemorySegment tensor) {
        String name = nativeBindings.getTensorName(tensor);
        LiteRTNativeBindings.TfLiteType type = LiteRTNativeBindings.TfLiteType.fromInt(
                nativeBindings.getTensorType(tensor));
        long[] shape = nativeBindings.getTensorShape(tensor);
        long byteSize = nativeBindings.getTensorByteSize(tensor);
        return new TensorInfo(name, type, shape, byteSize);
    }

    private TensorData readTensor(TensorInfo info, MemorySegment tensor) {
        long byteSize = nativeBindings.getTensorByteSize(tensor);
        if (byteSize > Integer.MAX_VALUE) {
            throw new InferenceException(ErrorCode.TENSOR_INVALID_DATA, "Tensor too large to read");
        }

        MemorySegment data = nativeBindings.getTensorData(tensor);
        byte[] bytes = new byte[(int) byteSize];
        ByteBuffer buffer = data.asByteBuffer();
        buffer.get(bytes);

        TensorDataType dtype = mapDataType(info.type);
        TensorData output = TensorData.builder()
                .name(info.name)
                .shape(info.shape)
                .dtype(dtype)
                .data(bytes)
                .build();

        decodeTypedData(output);
        return output;
    }

    private TensorDataType mapDataType(LiteRTNativeBindings.TfLiteType type) {
        return switch (type) {
            case FLOAT32 -> TensorDataType.FLOAT32;
            case FLOAT16 -> TensorDataType.FLOAT16;
            case FLOAT64 -> TensorDataType.FLOAT64;
            case INT8 -> TensorDataType.INT8;
            case UINT8 -> TensorDataType.UINT8;
            case INT16 -> TensorDataType.INT16;
            case UINT16 -> TensorDataType.UINT16;
            case INT32 -> TensorDataType.INT32;
            case UINT32 -> TensorDataType.UINT32;
            case INT64 -> TensorDataType.INT64;
            case UINT64 -> TensorDataType.UINT64;
            case BOOL -> TensorDataType.BOOL;
            case STRING -> TensorDataType.STRING;
            default -> TensorDataType.FLOAT32;
        };
    }

    private void decodeTypedData(TensorData output) {
        if (output.getData() == null || output.getDtype() == null) {
            return;
        }

        ByteBuffer buffer = ByteBuffer.wrap(output.getData()).order(ByteOrder.nativeOrder());
        int count = (int) (output.getData().length / Math.max(1, output.getDtype().getByteSize()));

        switch (output.getDtype()) {
            case FLOAT32 -> {
                float[] values = new float[count];
                for (int i = 0; i < count; i++) {
                    values[i] = buffer.getFloat();
                }
                output.setFloatData(values);
            }
            case FLOAT16 -> {
                float[] values = new float[count];
                for (int i = 0; i < count; i++) {
                    values[i] = buffer.getShort();
                }
                output.setFloatData(values);
            }
            case INT8, UINT8, BOOL -> {
                int[] values = new int[count];
                for (int i = 0; i < count; i++) {
                    values[i] = buffer.get();
                }
                output.setIntData(values);
            }
            case INT16, UINT16 -> {
                int[] values = new int[count];
                for (int i = 0; i < count; i++) {
                    values[i] = buffer.getShort();
                }
                output.setIntData(values);
            }
            case INT32, UINT32 -> {
                int[] values = new int[count];
                for (int i = 0; i < count; i++) {
                    values[i] = buffer.getInt();
                }
                output.setIntData(values);
            }
            case INT64, UINT64 -> {
                long[] values = new long[count];
                for (int i = 0; i < count; i++) {
                    values[i] = buffer.getLong();
                }
                output.setLongData(values);
            }
            default -> {
                // leave raw bytes
            }
        }
    }

    private Map<String, TensorData> resolveInputs(InferenceRequest request) {
        Object raw = request.getParameters().get("inputs");
        if (raw instanceof List<?> list) {
            return parseTensorList(list);
        }
        if (raw instanceof Map<?, ?> map) {
            return parseTensorMap(map);
        }

        Object single = request.getParameters().get("input");
        if (single instanceof TensorData td) {
            return Map.of(td.getName() != null ? td.getName() : "input_0", td);
        }

        if (!request.getMessages().isEmpty()) {
            TensorInfo info = inputTensors.get(0);
            if (info != null && info.type == LiteRTNativeBindings.TfLiteType.STRING) {
                String text = request.getMessages().stream()
                        .map(m -> m.getContent())
                        .reduce("", (a, b) -> a + " " + b)
                        .trim();
                TensorData td = TensorData.builder()
                        .name(info.name != null ? info.name : "input_0")
                        .shape(info.shape != null ? info.shape : new long[] { 1 })
                        .dtype(TensorDataType.STRING)
                        .data(text.getBytes(java.nio.charset.StandardCharsets.UTF_8))
                        .build();
                return Map.of(td.getName(), td);
            }
        }

        throw new InferenceException(ErrorCode.TENSOR_MISSING_INPUT, "No inputs provided");
    }

    private Map<String, TensorData> parseTensorList(List<?> list) {
        Map<String, TensorData> inputs = new LinkedHashMap<>();
        for (int i = 0; i < list.size(); i++) {
            Object item = list.get(i);
            TensorData td = coerceTensorData(item, "input_" + i);
            inputs.put(td.getName() != null ? td.getName() : "input_" + i, td);
        }
        return inputs;
    }

    private Map<String, TensorData> parseTensorMap(Map<?, ?> map) {
        Map<String, TensorData> inputs = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            String name = String.valueOf(entry.getKey());
            TensorData td = coerceTensorData(entry.getValue(), name);
            inputs.put(name, td);
        }
        return inputs;
    }

    private TensorData coerceTensorData(Object value, String fallbackName) {
        if (value instanceof TensorData td) {
            if (td.getName() == null) {
                td.setName(fallbackName);
            }
            return td;
        }
        if (value instanceof Map<?, ?> map) {
            TensorData td = new TensorData();
            Object nameObj = map.get("name");
            td.setName(nameObj != null ? String.valueOf(nameObj) : fallbackName);
            td.setShape(coerceLongArray(map.get("shape")));
            td.setDtype(coerceDtype(map.get("dtype")));
            td.setData(coerceByteArray(map.get("data")));
            td.setFloatData(coerceFloatArray(map.get("floatData")));
            td.setIntData(coerceIntArray(map.get("intData")));
            td.setLongData(coerceLongArray(map.get("longData")));
            td.setBoolData(coerceBoolArray(map.get("boolData")));
            return td;
        }
        throw new InferenceException(ErrorCode.TENSOR_INVALID_DATA, "Invalid input tensor format");
    }

    private long[] coerceLongArray(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof long[] arr) {
            return arr;
        }
        if (value instanceof List<?> list) {
            long[] arr = new long[list.size()];
            for (int i = 0; i < list.size(); i++) {
                arr[i] = ((Number) list.get(i)).longValue();
            }
            return arr;
        }
        return null;
    }

    private int[] coerceIntArray(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof int[] arr) {
            return arr;
        }
        if (value instanceof List<?> list) {
            int[] arr = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                arr[i] = ((Number) list.get(i)).intValue();
            }
            return arr;
        }
        return null;
    }

    private float[] coerceFloatArray(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof float[] arr) {
            return arr;
        }
        if (value instanceof List<?> list) {
            float[] arr = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                arr[i] = ((Number) list.get(i)).floatValue();
            }
            return arr;
        }
        return null;
    }

    private boolean[] coerceBoolArray(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof boolean[] arr) {
            return arr;
        }
        if (value instanceof List<?> list) {
            boolean[] arr = new boolean[list.size()];
            for (int i = 0; i < list.size(); i++) {
                arr[i] = (Boolean) list.get(i);
            }
            return arr;
        }
        return null;
    }

    private byte[] coerceByteArray(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof byte[] arr) {
            return arr;
        }
        if (value instanceof String s) {
            return Base64.getDecoder().decode(s);
        }
        return null;
    }

    private TensorDataType coerceDtype(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof TensorDataType dtype) {
            return dtype;
        }
        return TensorDataType.valueOf(String.valueOf(value).toUpperCase());
    }

    private TensorData findInputForIndex(int index, TensorInfo info, Map<String, TensorData> inputs) {
        if (info.name != null && inputs.containsKey(info.name)) {
            return inputs.get(info.name);
        }

        String fallback = "input_" + index;
        if (inputs.containsKey(fallback)) {
            return inputs.get(fallback);
        }

        if (inputs.size() == 1) {
            return inputs.values().iterator().next();
        }

        throw new InferenceException(ErrorCode.TENSOR_MISSING_INPUT,
                "Missing input tensor: " + (info.name != null ? info.name : fallback));
    }

    private void validateInput(TensorInfo info, TensorData input) {
        if (input.getDtype() == null) {
            input.setDtype(mapDataType(info.type));
        }
        if (input.getShape() == null) {
            input.setShape(info.shape);
        }

        if (info.shape != null && input.getShape() != null &&
                input.getShape().length == info.shape.length) {
            for (int i = 0; i < info.shape.length; i++) {
                if (info.shape[i] != input.getShape()[i]) {
                    throw new InferenceException(ErrorCode.TENSOR_SHAPE_MISMATCH,
                            "Input shape mismatch for " + (info.name != null ? info.name : "input_" + i));
                }
            }
        }

        if (input.getData() == null && input.getFloatData() == null
                && input.getIntData() == null && input.getLongData() == null
                && input.getBoolData() == null) {
            throw new InferenceException(ErrorCode.TENSOR_INVALID_DATA,
                    "Input tensor has no data: " + input.getName());
        }
    }

    private String serializeOutputs(Map<String, TensorData> outputs) {
        try {
            return OBJECT_MAPPER.writeValueAsString(serializeOutputsForMetadata(outputs));
        } catch (JsonProcessingException e) {
            log.warn("Failed to serialize outputs to JSON", e);
            return "{}";
        }
    }

    private Map<String, Object> serializeOutputsForMetadata(Map<String, TensorData> outputs) {
        Map<String, Object> payload = new LinkedHashMap<>();
        for (Map.Entry<String, TensorData> entry : outputs.entrySet()) {
            TensorData td = entry.getValue();
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("dtype", td.getDtype() != null ? td.getDtype().name() : null);
            item.put("shape", td.getShape() != null ? td.getShape() : new long[0]);

            if (td.getFloatData() != null) {
                item.put("floatData", td.getFloatData());
            } else if (td.getIntData() != null) {
                item.put("intData", td.getIntData());
            } else if (td.getLongData() != null) {
                item.put("longData", td.getLongData());
            } else if (td.getBoolData() != null) {
                item.put("boolData", td.getBoolData());
            } else if (td.getData() != null) {
                item.put("data", Base64.getEncoder().encodeToString(td.getData()));
            } else {
                item.put("data", null);
            }
            payload.put(entry.getKey(), item);
        }
        return payload;
    }

    private String findLiteRTLibrary() {
        String override = System.getProperty("LITERT_LIBRARY_PATH");
        if (override == null || override.isBlank()) {
            override = System.getenv("LITERT_LIBRARY_PATH");
        }
        if (override != null && !override.isBlank() && Files.exists(Paths.get(override))) {
            return override;
        }

        String os = System.getProperty("os.name").toLowerCase();
        String libName = os.contains("linux") ? "libtensorflowlite_c.so"
                : os.contains("mac") ? "libtensorflowlite_c.dylib"
                        : os.contains("win") ? "tensorflowlite_c.dll"
                                : null;

        if (libName == null) {
            throw new UnsupportedOperationException("Unsupported OS: " + os);
        }

        String[] searchPaths = {
                "/usr/local/lib/" + libName,
                "/usr/lib/" + libName,
                System.getProperty("user.home") + "/lib/" + libName,
                "./lib/" + libName
        };

        for (String path : searchPaths) {
            if (Files.exists(Paths.get(path))) {
                return path;
            }
        }

        throw new IllegalStateException("LiteRT library not found: " + libName);
    }

    private void cleanup() {
        try {
            if (interpreter != null && interpreter.address() != 0) {
                nativeBindings.deleteInterpreter(interpreter);
                interpreter = null;
            }
            if (model != null && model.address() != 0) {
                nativeBindings.deleteModel(model);
                model = null;
            }
            if (delegateManager != null) {
                delegateManager.cleanup();
                delegateManager = null;
            }
            if (arena != null) {
                arena.close();
                arena = null;
            }
            initialized = false;
        } catch (Exception e) {
            log.error("LiteRT cleanup failed", e);
        }
    }

    private record TensorInfo(
            String name,
            LiteRTNativeBindings.TfLiteType type,
            long[] shape,
            long byteSize) {
    }
}
