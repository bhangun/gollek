package tech.kayys.gollek.provider.litert;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

import static java.lang.foreign.ValueLayout.*;

/**
 * LiteRT 2.0 CPU runner using the modern CompiledModel API.
 *
 * <p>Lifecycle:
 * <ol>
 *   <li>Create Environment (with accelerator auto-registration)</li>
 *   <li>Load Model from file</li>
 *   <li>Create Options with hardware accelerator selection</li>
 *   <li>Create CompiledModel (JIT compilation with XNNPACK)</li>
 *   <li>For each inference: create TensorBuffers → Run → read outputs</li>
 * </ol>
 */
@Slf4j
public class LiteRTCpuRunner implements AutoCloseable {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private LiteRTNativeBindings bindings;

    private Arena arena;
    private MemorySegment environment;
    private MemorySegment model;
    private MemorySegment options;
    private MemorySegment compiledModel;

    // Model introspection results
    private MemorySegment primarySignature;
    private String signatureKey;
    private final Map<Integer, TensorInfo> inputTensors = new HashMap<>();
    private final Map<Integer, TensorInfo> outputTensors = new HashMap<>();

    // LLM path (unchanged)
    private LiteRTInferenceRunner llmRunner;
    private LiteRTTokenizer tokenizer;

    private boolean initialized = false;
    private int numThreads = 4;
    private boolean useGpu = false;
    private boolean useNpu = false;

    private final AtomicLong totalInferences = new AtomicLong(0);
    private final AtomicLong failedInferences = new AtomicLong(0);
    private final AtomicLong totalLatencyMs = new AtomicLong(0);

    public void initialize(Path modelPath, LiteRTRunnerConfig config) {
        if (initialized) return;

        try {
            this.numThreads = config.numThreads();
            this.useGpu = config.useGpu();
            this.useNpu = config.useNpu();

            this.arena = Arena.ofConfined();
            String libraryPath = findLiteRTLibrary();
            this.bindings = new LiteRTNativeBindings(Paths.get(libraryPath));

            if (!Files.exists(modelPath)) {
                throw new InferenceException(
                        ErrorCode.INIT_MODEL_LOAD_FAILED,
                        "Model file not found: " + modelPath);
            }

            // Check for LLM / specialized model formats
            try {
                LiteRTContainerParser.ContainerInfo containerInfo = LiteRTContainerParser.parse(modelPath);
                if (containerInfo.isLlmModel()) {
                    handleLlmModel(modelPath, containerInfo);
                    return;
                }
            } catch (InferenceException ie) {
                throw ie;
            } catch (Exception e) {
                log.warn("Container check failed, proceeding with standard path: {}", e.getMessage());
            }

            // ===== LiteRT 2.0 CompiledModel Pipeline =====

            // Step 1: Create Environment
            log.info("Creating LiteRT 2.0 environment...");
            this.environment = bindings.createEnvironment(arena);
            log.info("✓ Environment created");

            // Step 2: Load Model
            log.info("Loading model: {}", modelPath);
            this.model = bindings.createModelFromFile(modelPath.toAbsolutePath().toString(), arena);

            // Introspect model
            int numSigs = bindings.getNumModelSignatures(model, arena);
            int numSubgraphs = bindings.getNumModelSubgraphs(model, arena);
            log.info("Model loaded: {} subgraphs, {} signatures", numSubgraphs, numSigs);

            // Step 3: Create Options with hardware accelerator
            this.options = bindings.createOptions(arena);
            int accelerators = resolveAccelerators();
            bindings.setOptionsHardwareAccelerators(options, accelerators);
            log.info("Options created with accelerators: {}", describeAccelerators(accelerators));

            // Step 4: Create CompiledModel
            long t0 = System.currentTimeMillis();
            this.compiledModel = bindings.createCompiledModel(environment, model, options, arena);
            long compilationMs = System.currentTimeMillis() - t0;

            boolean fullyAccelerated = bindings.isFullyAccelerated(compiledModel, arena);
            log.info("✓ CompiledModel created in {}ms (fully accelerated: {})", compilationMs, fullyAccelerated);

            // Step 5: Introspect I/O via model signatures
            if (numSigs > 0) {
                this.primarySignature = bindings.getModelSignature(model, 0, arena);
                this.signatureKey = bindings.getSignatureKey(primarySignature, arena);
                log.info("Primary signature: '{}'", signatureKey);
                introspectSignature(primarySignature);
            } else {
                log.warn("No signatures found; inspecting subgraph 0 directly");
                MemorySegment subgraph = bindings.getModelSubgraph(model, 0, arena);
                introspectSubgraph(subgraph);
            }

            initialized = true;
            log.info("LiteRT 2.0 CPU runner initialized successfully.");
            if (inputTensors.isEmpty() && outputTensors.isEmpty()) {
                log.warn("No input/output tensors discovered. Model might require a specialized runner.");
            }
        } catch (Exception e) {
            log.error("LiteRT Runner Init Error: {}", e.getMessage(), e);
            cleanup();
            throw e instanceof InferenceException
                    ? (InferenceException) e
                    : new InferenceException(ErrorCode.INIT_RUNNER_FAILED, "LiteRT init failed: " + e.getMessage(), e);
        }
    }

    public InferenceResponse infer(InferenceRequest request) {
        if (!initialized) {
            throw new InferenceException(ErrorCode.RUNTIME_INVALID_STATE, "Runner not initialized");
        }

        long start = System.currentTimeMillis();

        try {
            if (llmRunner != null) {
                return runLlmInference(request, start);
            }

            Map<String, TensorData> inputs = resolveInputs(request);
            Map<String, TensorData> outputs = runInference(inputs);

            long latencyMs = System.currentTimeMillis() - start;
            totalInferences.incrementAndGet();
            totalLatencyMs.addAndGet(latencyMs);

            String content = serializeOutputs(outputs);
            Map<String, Object> metadata = Map.of(
                    "runner", "litert-2.0-cpu",
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
            if (e instanceof InferenceException ie) throw ie;
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED, "Inference failed", e);
        }
    }

    public CompletableFuture<InferenceResponse> inferAsync(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> infer(request));
    }

    public boolean health() {
        return initialized && compiledModel != null && compiledModel.address() != 0;
    }

    @Override
    public void close() {
        if (llmRunner != null) llmRunner.close();
        if (tokenizer != null) tokenizer.close();
        cleanup();
    }

    // ===== Internal: Inference =====

    private Map<String, TensorData> runInference(Map<String, TensorData> inputs) {
        try (Arena inferArena = Arena.ofConfined()) {
            int numInputs = inputTensors.size();
            int numOutputs = outputTensors.size();

            // Allocate pointer arrays for input/output TensorBuffers
            MemorySegment inputBufArray = inferArena.allocate(ADDRESS, numInputs);
            MemorySegment outputBufArray = inferArena.allocate(ADDRESS, numOutputs);

            List<MemorySegment> managedBuffers = new ArrayList<>();

            try {
                // Create input TensorBuffers with data
                for (int i = 0; i < numInputs; i++) {
                    TensorInfo info = inputTensors.get(i);
                    TensorData input = findInputForIndex(i, info, inputs);
                    validateInput(info, input);

                    byte[] bytes = TensorConverter.toNativeBytes(input);

                    // Allocate aligned host memory and copy input data
                    MemorySegment hostMem = inferArena.allocate(bytes.length, 64);
                    hostMem.copyFrom(MemorySegment.ofArray(bytes));

                    // Build a minimal LiteRtRankedTensorType struct for Host buffer creation
                    // For CPU host memory, the tensor type layout varies by LiteRT version
                    // We use the buffer requirements from the compiled model when available
                    MemorySegment bufReq = bindings.getCompiledModelInputBufferRequirements(
                            compiledModel, 0, i, inferArena);

                    MemorySegment tensorBuf = bindings.createManagedTensorBufferFromRequirements(
                            environment, MemorySegment.NULL, bufReq, inferArena);
                    managedBuffers.add(tensorBuf);

                    // Lock → write data → unlock
                    MemorySegment locked = bindings.lockTensorBuffer(tensorBuf,
                            LiteRTNativeBindings.kLiteRtTensorBufferLockModeWrite, inferArena);
                    MemorySegment.copy(MemorySegment.ofArray(bytes), 0, locked.reinterpret(bytes.length), 0, bytes.length);
                    bindings.unlockTensorBuffer(tensorBuf);

                    inputBufArray.setAtIndex(ADDRESS, i, tensorBuf);
                }

                // Create output TensorBuffers from requirements
                for (int i = 0; i < numOutputs; i++) {
                    MemorySegment bufReq = bindings.getCompiledModelOutputBufferRequirements(
                            compiledModel, 0, i, inferArena);

                    MemorySegment tensorBuf = bindings.createManagedTensorBufferFromRequirements(
                            environment, MemorySegment.NULL, bufReq, inferArena);
                    managedBuffers.add(tensorBuf);
                    outputBufArray.setAtIndex(ADDRESS, i, tensorBuf);
                }

                // Run inference!
                bindings.runCompiledModel(compiledModel, 0, inputBufArray, numInputs, outputBufArray, numOutputs);

                // Read output data
                Map<String, TensorData> outputs = new LinkedHashMap<>();
                for (int i = 0; i < numOutputs; i++) {
                    TensorInfo info = outputTensors.get(i);
                    MemorySegment tensorBuf = outputBufArray.getAtIndex(ADDRESS, i);

                    long bufSize = bindings.getTensorBufferSize(tensorBuf, inferArena);
                    MemorySegment locked = bindings.lockTensorBuffer(tensorBuf,
                            LiteRTNativeBindings.kLiteRtTensorBufferLockModeRead, inferArena);

                    byte[] rawBytes = locked.reinterpret(bufSize).toArray(JAVA_BYTE);
                    bindings.unlockTensorBuffer(tensorBuf);

                    TensorDataType dtype = mapDataType(info.type);
                    TensorData output = TensorData.builder()
                            .name(info.name)
                            .shape(info.shape)
                            .dtype(dtype)
                            .data(rawBytes)
                            .build();
                    decodeTypedData(output);
                    outputs.put(info.name != null ? info.name : "output_" + i, output);
                }

                return outputs;
            } finally {
                // Cleanup tensor buffers
                for (MemorySegment buf : managedBuffers) {
                    bindings.destroyTensorBuffer(buf);
                }
            }
        }
    }

    // ===== Internal: Introspection =====

    private void introspectSignature(MemorySegment signature) {
        inputTensors.clear();
        outputTensors.clear();

        int numInputs = bindings.getNumSignatureInputs(signature, arena);
        for (int i = 0; i < numInputs; i++) {
            String name = bindings.getSignatureInputName(signature, i, arena);
            inputTensors.put(i, new TensorInfo(name, LiteRTNativeBindings.LitertType.FLOAT32, null, 0));
        }

        int numOutputs = bindings.getNumSignatureOutputs(signature, arena);
        for (int i = 0; i < numOutputs; i++) {
            String name = bindings.getSignatureOutputName(signature, i, arena);
            outputTensors.put(i, new TensorInfo(name, LiteRTNativeBindings.LitertType.FLOAT32, null, 0));
        }

        log.info("Signature '{}': {} inputs, {} outputs", signatureKey, numInputs, numOutputs);
    }

    private void introspectSubgraph(MemorySegment subgraph) {
        inputTensors.clear();
        outputTensors.clear();
        // Subgraph introspection uses LiteRtGetNumSubgraphInputs etc.
        // For now, minimal: just log
        log.info("Subgraph introspected (details require buffer requirements from CompiledModel)");
    }

    private void handleLlmModel(Path modelPath, LiteRTContainerParser.ContainerInfo containerInfo) {
        if (containerInfo.format() == LiteRTContainerParser.ContainerFormat.LITERTLM) {
            log.info("Detected .litertlm container, delegating to LiteRTInferenceRunner for potential Native Metal fallback");
        }

        // .task, .litertlm or standalone .tflite LLM
        log.info("Detected LLM model, initializing LiteRTInferenceRunner");
        this.tokenizer = LiteRTTokenizer.create(modelPath);
        this.llmRunner = new LiteRTInferenceRunner(bindings, modelPath, tokenizer, useGpu, numThreads);
        this.llmRunner.initialize();
        this.initialized = true;
    }

    private InferenceResponse runLlmInference(InferenceRequest request, long start) {
        StringBuilder result = new StringBuilder();
        String prompt = request.getPrompt();
        if (prompt == null) prompt = "";

        llmRunner.generate(prompt, result::append);

        long latencyMs = System.currentTimeMillis() - start;
        totalInferences.incrementAndGet();
        totalLatencyMs.addAndGet(latencyMs);

        return InferenceResponse.builder()
                .requestId(request.getRequestId())
                .model(request.getModel())
                .durationMs(latencyMs)
                .content(result.toString())
                .metadata(Map.of("runner", "litert-llm"))
                .build();
    }

    // ===== Internal: Accelerator Resolution =====

    private int resolveAccelerators() {
        int accel = LiteRTNativeBindings.kLiteRtHwAcceleratorCpu;
        if (useGpu) accel |= LiteRTNativeBindings.kLiteRtHwAcceleratorGpu;
        if (useNpu) accel |= LiteRTNativeBindings.kLiteRtHwAcceleratorNpu;
        return accel;
    }

    private String describeAccelerators(int accel) {
        List<String> names = new ArrayList<>();
        if ((accel & LiteRTNativeBindings.kLiteRtHwAcceleratorCpu) != 0) names.add("CPU");
        if ((accel & LiteRTNativeBindings.kLiteRtHwAcceleratorGpu) != 0) names.add("GPU");
        if ((accel & LiteRTNativeBindings.kLiteRtHwAcceleratorNpu) != 0) names.add("NPU");
        return String.join("+", names);
    }

    // ===== Internal: Input Resolution =====

    private Map<String, TensorData> resolveInputs(InferenceRequest request) {
        Object inputsObj = request.getParameters().get("inputs");
        if (inputsObj != null) {
            if (inputsObj instanceof List<?> list) return parseTensorList(list);
            if (inputsObj instanceof Map<?, ?> map) return parseTensorMap(map);
        }

        if (!request.getMessages().isEmpty()) {
            if (inputTensors.isEmpty()) {
                throw new InferenceException(ErrorCode.TENSOR_MISSING_INPUT,
                        "Model has no input tensors. Check signatures.");
            }

            TensorInfo info = inputTensors.values().stream()
                    .filter(t -> t.type == LiteRTNativeBindings.LitertType.STRING)
                    .findFirst().orElse(null);

            if (info != null) {
                String text = request.getMessages().get(0).getContent();
                TensorData td = TensorData.builder()
                        .name(info.name).dtype(TensorDataType.STRING)
                        .data(text.getBytes(java.nio.charset.StandardCharsets.UTF_8))
                        .shape(new long[]{1})
                        .build();
                return Map.of(info.name != null ? info.name : "input_0", td);
            }

            // Default: single float tensor from first message
            return createDefaultInputFromMessage(request);
        }

        throw new InferenceException(ErrorCode.TENSOR_MISSING_INPUT, "No inputs provided");
    }

    private Map<String, TensorData> createDefaultInputFromMessage(InferenceRequest request) {
        TensorInfo info = inputTensors.get(0);
        if (info == null) {
            throw new InferenceException(ErrorCode.TENSOR_MISSING_INPUT, "No input tensors");
        }
        long totalElements = 1;
        if (info.shape != null) {
            for (long d : info.shape) totalElements *= d;
        }
        float[] zeros = new float[(int) totalElements];
        TensorData td = TensorData.builder()
                .name(info.name).dtype(TensorDataType.FLOAT32)
                .shape(info.shape).floatData(zeros)
                .build();
        return Map.of(info.name != null ? info.name : "input_0", td);
    }

    @SuppressWarnings("unchecked")
    private Map<String, TensorData> parseTensorList(List<?> list) {
        Map<String, TensorData> result = new LinkedHashMap<>();
        for (int i = 0; i < list.size(); i++) {
            Map<String, Object> item = (Map<String, Object>) list.get(i);
            TensorData td = parseSingleTensorInput(item, i);
            result.put(td.getName() != null ? td.getName() : "input_" + i, td);
        }
        return result;
    }

    @SuppressWarnings("unchecked")
    private Map<String, TensorData> parseTensorMap(Map<?, ?> map) {
        Map<String, TensorData> result = new LinkedHashMap<>();
        int idx = 0;
        for (var entry : map.entrySet()) {
            Map<String, Object> item = (Map<String, Object>) entry.getValue();
            TensorData td = parseSingleTensorInput(item, idx++);
            td.setName(entry.getKey().toString());
            result.put(td.getName(), td);
        }
        return result;
    }

    @SuppressWarnings("unchecked")
    private TensorData parseSingleTensorInput(Map<String, Object> item, int idx) {
        TensorData.TensorDataBuilder builder = TensorData.builder();
        builder.name(item.containsKey("name") ? item.get("name").toString() : "input_" + idx);

        if (item.containsKey("data") && item.get("data") instanceof String b64) {
            builder.data(Base64.getDecoder().decode(b64));
        }
        if (item.containsKey("floatData") && item.get("floatData") instanceof List<?> flist) {
            float[] arr = new float[flist.size()];
            for (int i = 0; i < flist.size(); i++) arr[i] = ((Number) flist.get(i)).floatValue();
            builder.floatData(arr);
            builder.dtype(TensorDataType.FLOAT32);
        }
        if (item.containsKey("shape") && item.get("shape") instanceof List<?> sList) {
            long[] shape = new long[sList.size()];
            for (int i = 0; i < sList.size(); i++) shape[i] = ((Number) sList.get(i)).longValue();
            builder.shape(shape);
        }
        return builder.build();
    }

    private TensorData findInputForIndex(int index, TensorInfo info, Map<String, TensorData> inputs) {
        if (info.name != null && inputs.containsKey(info.name)) {
            return inputs.get(info.name);
        }
        String fallbackKey = "input_" + index;
        if (inputs.containsKey(fallbackKey)) {
            return inputs.get(fallbackKey);
        }
        if (inputs.size() == 1) {
            return inputs.values().iterator().next();
        }
        throw new InferenceException(ErrorCode.TENSOR_MISSING_INPUT, "Missing input for: " + info.name);
    }

    private void validateInput(TensorInfo info, TensorData input) {
        if (input == null) {
            throw new InferenceException(ErrorCode.TENSOR_MISSING_INPUT,
                    "Input tensor is null: " + info.name);
        }
        if (input.getData() == null && input.getFloatData() == null
                && input.getIntData() == null && input.getLongData() == null
                && input.getBoolData() == null) {
            throw new InferenceException(ErrorCode.TENSOR_INVALID_DATA,
                    "Input tensor has no data: " + input.getName());
        }
    }

    // ===== Internal: Output Processing =====

    private TensorDataType mapDataType(LiteRTNativeBindings.LitertType type) {
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
        if (output.getData() == null || output.getDtype() == null) return;

        ByteBuffer buffer = ByteBuffer.wrap(output.getData()).order(ByteOrder.nativeOrder());
        int count = (int) (output.getData().length / Math.max(1, output.getDtype().getByteSize()));

        switch (output.getDtype()) {
            case FLOAT32 -> {
                float[] values = new float[count];
                for (int i = 0; i < count; i++) values[i] = buffer.getFloat();
                output.setFloatData(values);
            }
            case INT8, UINT8, BOOL -> {
                int[] values = new int[count];
                for (int i = 0; i < count; i++) values[i] = buffer.get();
                output.setIntData(values);
            }
            case INT32, UINT32 -> {
                int[] values = new int[count];
                for (int i = 0; i < count; i++) values[i] = buffer.getInt();
                output.setIntData(values);
            }
            case INT64, UINT64 -> {
                long[] values = new long[count];
                for (int i = 0; i < count; i++) values[i] = buffer.getLong();
                output.setLongData(values);
            }
            default -> { /* leave raw bytes */ }
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

            if (td.getFloatData() != null) item.put("floatData", td.getFloatData());
            else if (td.getIntData() != null) item.put("intData", td.getIntData());
            else if (td.getLongData() != null) item.put("longData", td.getLongData());
            else if (td.getBoolData() != null) item.put("boolData", td.getBoolData());
            else if (td.getData() != null) item.put("data", Base64.getEncoder().encodeToString(td.getData()));
            else item.put("data", null);

            payload.put(entry.getKey(), item);
        }
        return payload;
    }

    // ===== Internal: Library Resolution =====

    private String findLiteRTLibrary() {
        String override = System.getProperty("LITERT_LIBRARY_PATH");
        if (override == null || override.isBlank()) override = System.getenv("LITERT_LIBRARY_PATH");
        if (override != null && !override.isBlank() && Files.exists(Paths.get(override))) return override;

        String os = System.getProperty("os.name").toLowerCase();
        // LiteRT 2.0 uses libLiteRt.dylib / libLiteRt.so (new naming)
        String[] libNames;
        if (os.contains("mac")) {
            libNames = new String[]{"libLiteRt.dylib", "libtensorflowlite_c.dylib"};
        } else if (os.contains("linux")) {
            libNames = new String[]{"libLiteRt.so", "libtensorflowlite_c.so"};
        } else if (os.contains("win")) {
            libNames = new String[]{"LiteRt.dll", "tensorflowlite_c.dll"};
        } else {
            throw new UnsupportedOperationException("Unsupported OS: " + os);
        }

        String[] searchDirs = {
                System.getProperty("user.home") + "/.gollek/libs/",
                "/usr/local/lib/",
                "/usr/lib/",
                System.getProperty("user.home") + "/lib/",
                "./lib/"
        };

        for (String dir : searchDirs) {
            for (String libName : libNames) {
                Path candidate = Paths.get(dir + libName);
                if (Files.exists(candidate)) {
                    log.info("Found LiteRT library: {}", candidate);
                    // On macOS, if we found libtensorflowlite_c.dylib but libLiteRt.dylib ALSO exists, we MUST pick LiteRt
                    if (os.contains("mac") && libName.contains("tensorflow")) {
                        Path litertCandidate = Paths.get(dir + "libLiteRt.dylib");
                        if (Files.exists(litertCandidate)) {
                            log.warn("Skipping legacy library {} in favor of LiteRT 2.0 at {}", candidate, litertCandidate);
                            return litertCandidate.toString();
                        }
                    }
                    return candidate.toString();
                }
            }
        }

        throw new IllegalStateException("LiteRT library not found. Install libLiteRt.dylib to ~/.gollek/libs/");
    }

    // ===== Internal: Cleanup =====

    private void cleanup() {
        try {
            if (compiledModel != null && compiledModel.address() != 0) {
                bindings.destroyCompiledModel(compiledModel);
                compiledModel = null;
            }
            if (options != null && options.address() != 0) {
                bindings.destroyOptions(options);
                options = null;
            }
            if (model != null && model.address() != 0) {
                bindings.destroyModel(model);
                model = null;
            }
            if (environment != null && environment.address() != 0) {
                bindings.destroyEnvironment(environment);
                environment = null;
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

    // ===== Internal: Types =====

    private record TensorInfo(
            String name,
            LiteRTNativeBindings.LitertType type,
            long[] shape,
            long byteSize) {
    }
}
