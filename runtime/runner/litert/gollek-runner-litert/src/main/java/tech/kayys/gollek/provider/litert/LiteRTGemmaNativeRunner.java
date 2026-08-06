package tech.kayys.gollek.provider.litert;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Native LiteRT-LM runner for Gemma-4 .litertlm bundles that expose the
 * decode graph as LiteRT signatures.
 */
public final class LiteRTGemmaNativeRunner implements AutoCloseable {

    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(LiteRTGemmaNativeRunner.class);

    private static final int MAX_RECENT_TOKENS = 128;
    private static final int MAX_REPEAT_STREAK = 12;
    private static final Pattern SIMPLE_WHERE_LOCATION_PROMPT =
            Pattern.compile("(?i)^\\s*where\\s+(?:is|are)\\s+(.+?)\\s*\\??\\s*$");
    static final String RAW_PREFILL_MIN_TOKENS_PROPERTY = "gollek.litert.raw_prefill_min_tokens";
    static final String RAW_PREFILL_DECODE_TAIL_TOKENS_PROPERTY = "gollek.litert.raw_prefill_decode_tail_tokens";
    static final String ENABLE_EXPERIMENTAL_RAW_LITERTLM_PROPERTY =
            "gollek.litert.enable_experimental_raw_litertlm";
    static final String DISABLE_RAW_LITERTLM_PROPERTY =
            "gollek.litert.disable_raw_litertlm";
    static final String ALLOW_RAW_LITERTLM_GPU_ON_MACOS_PROPERTY =
            "gollek.litert.allow_raw_litertlm_gpu_on_macos";
    static final String ENABLE_RAW_LITERTLM_AUX_GPU_ON_MACOS_PROPERTY =
            "gollek.litert.enable_raw_litertlm_aux_gpu_on_macos";
    static final String ENABLE_STATEFUL_KV_BUFFERS_PROPERTY =
            "gollek.litert.enable_stateful_kv_buffers";
    private static final int MIN_RAW_PREFILL_TOKENS = 4;
    // Batched prefill is much faster than token-by-token prompt ingestion for
    // this raw LiteRT-LM export. Keeping no decode tail preserved the Jakarta
    // smoke answer after the KV buffers were made stateful.
    private static final int RAW_PREFILL_DECODE_TAIL_TOKENS = 0;

    private final LiteRTNativeBindings bindings;
    private final Path modelPath;
    private final LiteRTTokenizer tokenizer;
    private final boolean useGpu;

    private final Arena arena = Arena.ofConfined();

    private MemorySegment environment;
    private MemorySegment options;
    private final List<MemorySegment> compilationOptions = new ArrayList<>();

    private CompiledSegment mainSegment;
    private CompiledSegment utilitySegment;
    private CompiledSegment embedderSegment;
    private CompiledSegment perLayerSegment;

    private NativeGraphLayout graphLayout = NativeGraphLayout.HELPER_SEGMENTS;
    private SignatureSpec decodeSignature;
    private SignatureSpec prefill128Signature;
    private SignatureSpec prefill1024Signature;
    private SignatureSpec verifySignature;
    private SignatureSpec utilityDecodeRopeSignature;
    private SignatureSpec embedderDecodeSignature;
    private SignatureSpec perLayerDecodeSignature;
    private TensorSpec logitsOutputSpec;
    private byte[] zeroParamTensor;

    private final Map<String, byte[]> kvCaches = new LinkedHashMap<>();
    private final Map<String, TensorSpec> kvCacheSpecs = new HashMap<>();
    private int maxSeqLen = 4096;
    private boolean initialized;
    private boolean statefulKvLogEmitted;

    public LiteRTGemmaNativeRunner(LiteRTNativeBindings bindings, Path modelPath,
                                   LiteRTTokenizer tokenizer, boolean useGpu) {
        this.bindings = Objects.requireNonNull(bindings, "bindings");
        this.modelPath = Objects.requireNonNull(modelPath, "modelPath");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.useGpu = useGpu;
    }

    public void initialize() {
        if (initialized) {
            return;
        }

        try {
            LiteRTContainerParser.ContainerInfo info = LiteRTContainerParser.parse(modelPath);
            if (info.format() != LiteRTContainerParser.ContainerFormat.LITERTLM) {
                throw new IllegalArgumentException("Expected .litertlm model, got: " + info.format());
            }
            if (info.tfliteOffset() <= 0 || info.tfliteSize() <= 0) {
                throw new IllegalArgumentException("Missing primary LiteRT-LM TFLite segment");
            }

            List<SegmentCandidate> candidates = collectSegmentCandidates(info);
            SegmentCandidate rawMainCandidate = null;
            SegmentCandidate rawEmbedderCandidate = null;
            SegmentCandidate rawPerLayerCandidate = null;
            SegmentCandidate helperMainCandidate = null;
            SegmentCandidate utilityCandidate = null;
            SegmentCandidate helperEmbedderCandidate = null;
            SegmentCandidate helperPerLayerCandidate = null;

            for (SegmentCandidate candidate : candidates) {
                if (candidate.signatureKeys.contains("decode")) {
                    if (candidate.signatureKeys.contains("prefill_128")
                            || candidate.signatureKeys.contains("prefill_1024")
                            || candidate.signatureKeys.contains("verify")) {
                        rawMainCandidate = preferLargerCandidate(rawMainCandidate, candidate);
                    } else {
                        helperMainCandidate = preferLargerCandidate(helperMainCandidate, candidate);
                    }
                }
                if (candidate.signatureKeys.contains("decode_mask")
                        && candidate.signatureKeys.contains("decode_rope")) {
                    utilityCandidate = candidate;
                }
                if (candidate.signatureKeys.contains("decode_embedder")) {
                    helperEmbedderCandidate = preferLargerCandidate(helperEmbedderCandidate, candidate);
                }
                if (candidate.signatureKeys.contains("decode_per_layer_embedder")) {
                    helperPerLayerCandidate = preferLargerCandidate(helperPerLayerCandidate, candidate);
                }
                if (candidate.signatureKeys.contains("embedder")) {
                    rawEmbedderCandidate = preferLargerCandidate(rawEmbedderCandidate, candidate);
                }
                if (candidate.signatureKeys.contains("per_layer_embedder")) {
                    rawPerLayerCandidate = preferLargerCandidate(rawPerLayerCandidate, candidate);
                }
            }

            boolean hasRawGraph = rawMainCandidate != null
                    && rawEmbedderCandidate != null
                    && rawPerLayerCandidate != null;
            boolean hasHelperGraph = helperMainCandidate != null
                    && utilityCandidate != null
                    && helperEmbedderCandidate != null
                    && helperPerLayerCandidate != null;
            if (!hasRawGraph && !hasHelperGraph) {
                throw new IllegalStateException("Could not find a supported LiteRT-LM signature layout");
            }
            if (hasRawGraph && rawLiteRtLmDisabled()) {
                throw new UnsupportedOperationException("Raw Gemma LiteRT-LM signatures were disabled by -D"
                        + DISABLE_RAW_LITERTLM_PROPERTY + "=true.");
            }

            boolean allowGpuForRawGraph = useGpu
                    && !(hasRawGraph && isMacOs() && !rawLiteRtLmGpuOnMacOsAllowed());
            if (useGpu && hasRawGraph && isMacOs() && !allowGpuForRawGraph) {
                log.warn("Disabling LiteRT GPU acceleration for the oversized raw Gemma LiteRT-LM decode graph on macOS; CPU/XNNPACK fallback is faster and correct for this export. Set -D{}=true to probe smaller auxiliary graphs on Metal, or -D{}=true to force raw decode Metal diagnostics.",
                        ENABLE_RAW_LITERTLM_AUX_GPU_ON_MACOS_PROPERTY,
                        ALLOW_RAW_LITERTLM_GPU_ON_MACOS_PROPERTY);
            }
            int cpuAccelerators = LiteRTNativeBindings.kLiteRtHwAcceleratorCpu;
            int rawAccelerators = cpuAccelerators
                    | (allowGpuForRawGraph ? LiteRTNativeBindings.kLiteRtHwAcceleratorGpu : 0);
            boolean hybridRawMetal = hasRawGraph && useGpu && isMacOs()
                    && rawLiteRtLmAuxGpuOnMacOsEnabled();
            int auxiliaryAccelerators = cpuAccelerators
                    | (hybridRawMetal ? LiteRTNativeBindings.kLiteRtHwAcceleratorGpu : 0);
            log.info("Native LiteRT-LM selected accelerators: {}",
                    hybridRawMetal ? "CPU raw decode + GPU auxiliary segments" :
                            (allowGpuForRawGraph ? "CPU+GPU" : "CPU"));
            try {
                if (hasRawGraph) {
                    initializeCompiledState(rawMainCandidate, null, rawEmbedderCandidate, rawPerLayerCandidate,
                            hybridRawMetal ? cpuAccelerators : rawAccelerators,
                            hybridRawMetal ? auxiliaryAccelerators : rawAccelerators,
                            NativeGraphLayout.RAW_SELF_CONTAINED);
                } else {
                    initializeCompiledState(helperMainCandidate, utilityCandidate,
                            helperEmbedderCandidate, helperPerLayerCandidate,
                            rawAccelerators, rawAccelerators, NativeGraphLayout.HELPER_SEGMENTS);
                }
            } catch (Exception firstFailure) {
                if (!allowGpuForRawGraph && !hybridRawMetal) {
                    throw firstFailure;
                }
                log.warn("Native LiteRT-LM GPU compile failed, retrying on CPU only: {}", firstFailure.getMessage());
                destroyCompiledState();
                if (hasRawGraph) {
                    initializeCompiledState(rawMainCandidate, null, rawEmbedderCandidate, rawPerLayerCandidate,
                            LiteRTNativeBindings.kLiteRtHwAcceleratorCpu,
                            LiteRTNativeBindings.kLiteRtHwAcceleratorCpu,
                            NativeGraphLayout.RAW_SELF_CONTAINED);
                } else {
                    initializeCompiledState(helperMainCandidate, utilityCandidate,
                            helperEmbedderCandidate, helperPerLayerCandidate,
                            LiteRTNativeBindings.kLiteRtHwAcceleratorCpu,
                            LiteRTNativeBindings.kLiteRtHwAcceleratorCpu,
                            NativeGraphLayout.HELPER_SEGMENTS);
                }
            }

            prepareCaches();
            initialized = true;
            log.info("Initialized native LiteRT-LM Gemma runner (layout={}, maxSeqLen={}, caches={})",
                    graphLayout, maxSeqLen, kvCaches.size());
        } catch (Exception e) {
            destroyCompiledState();
            throw new RuntimeException("Failed to initialize native LiteRT-LM runner: " + e.getMessage(), e);
        }
    }

    public void generate(InferenceRequest request, Consumer<String> tokenCallback) {
        boolean plainPrompt = request.getMessages() == null || request.getMessages().isEmpty();
        String prompt = request.getPrompt() == null ? "" : request.getPrompt();
        if (!plainPrompt && request.getMessages().size() == 1
                && request.getMessages().get(0).getRole() == Message.Role.USER) {
            plainPrompt = true;
            prompt = request.getMessages().get(0).getContent() == null
                    ? ""
                    : request.getMessages().get(0).getContent();
        }
        int[] inputIds = plainPrompt
                ? tokenizer.encodeWithBos(formatPlainPrompt(prompt))
                : tokenizer.encodeChatPrompt(request.getMessages());

        generate(inputIds,
                Math.max(1, request.getMaxTokens()),
                request.getTemperature(),
                request.getTopK(),
                request.getTopP(),
                request.getRepeatPenalty(),
                plainPrompt ? suppressPlainPromptEcho(prompt, tokenCallback) : tokenCallback);
    }

    public void generate(String prompt, Consumer<String> tokenCallback) {
        generate(tokenizer.encodeWithBos(formatPlainPrompt(prompt)),
                512,
                0.2d,
                40,
                0.9d,
                1.1d,
                suppressPlainPromptEcho(prompt, tokenCallback));
    }

    static String formatPlainPrompt(String prompt) {
        String trimmed = prompt == null ? "" : prompt.trim();
        if (trimmed.isEmpty()
                || trimmed.contains("\n")
                || trimmed.toLowerCase(java.util.Locale.ROOT).contains("answer:")) {
            return prompt == null ? "" : prompt;
        }
        Matcher whereMatcher = SIMPLE_WHERE_LOCATION_PROMPT.matcher(trimmed);
        if (whereMatcher.matches()) {
            String subject = titleCaseIfLower(whereMatcher.group(1));
            trimmed = "What country, region, or place is " + subject + " located in?";
        }
        // This LiteRT-LM export was trained/packaged to continue the escaped QA cue
        // more reliably than the chat template or a real newline cue.
        return "Question: " + trimmed + "\\nAnswer:";
    }

    private static String titleCaseIfLower(String text) {
        String normalized = text == null ? "" : text.trim();
        if (normalized.isEmpty() || !normalized.equals(normalized.toLowerCase(java.util.Locale.ROOT))) {
            return normalized;
        }
        String[] parts = normalized.split("\\s+");
        for (int i = 0; i < parts.length; i++) {
            if (!parts[i].isEmpty()) {
                parts[i] = parts[i].substring(0, 1).toUpperCase(java.util.Locale.ROOT) + parts[i].substring(1);
            }
        }
        return String.join(" ", parts);
    }

    private Consumer<String> suppressPlainPromptEcho(String prompt, Consumer<String> downstream) {
        String trimmedPrompt = prompt == null ? "" : prompt.trim();
        if (trimmedPrompt.isEmpty()) {
            return downstream;
        }
        return new Consumer<>() {
            private final StringBuilder pending = new StringBuilder();
            private boolean released;

            @Override
            public void accept(String delta) {
                if (delta == null || delta.isEmpty()) {
                    return;
                }
                if (released) {
                    downstream.accept(delta);
                    return;
                }
                pending.append(delta);
                String text = pending.toString();
                int answerIndex = text.toLowerCase(java.util.Locale.ROOT).lastIndexOf("answer:");
                if (answerIndex >= 0) {
                    released = true;
                    String answer = text.substring(answerIndex + "answer:".length());
                    if (!answer.isBlank()) {
                        downstream.accept(answer.stripLeading());
                    }
                    return;
                }
                String lower = text.toLowerCase(java.util.Locale.ROOT);
                String lowerPrompt = trimmedPrompt.toLowerCase(java.util.Locale.ROOT);
                if (!lower.startsWith("question:")
                        && !lowerPrompt.startsWith(lower.trim())
                        && !lower.contains(lowerPrompt)) {
                    released = true;
                    downstream.accept(text);
                }
            }
        };
    }

    public void generate(int[] inputIds,
                         int maxNewTokens,
                         double temperature,
                         int topK,
                         double topP,
                         double repeatPenalty,
                         Consumer<String> tokenCallback) {
        if (!initialized) {
            throw new IllegalStateException("Native LiteRT-LM runner is not initialized");
        }
        if (inputIds == null || inputIds.length == 0) {
            inputIds = tokenizer.encodeChatPrompt("");
        }
        if (inputIds.length >= maxSeqLen) {
            throw new IllegalArgumentException("Prompt is too long for LiteRT-LM KV cache (" + maxSeqLen + ")");
        }

        resetCaches();
        Deque<Integer> recentTokens = new ArrayDeque<>();
        List<Integer> generatedTokens = new ArrayList<>();
        String emittedText = "";
        int repeatedTokenStreak = 0;
        int lastToken = Integer.MIN_VALUE;

        try {
            PromptPreparation preparedPrompt = preparePrompt(inputIds);
            byte[] logits = preparedPrompt.logits;
            int nextToken = selectNextToken(logits, recentTokens, temperature, topK, topP, repeatPenalty);
            int position = preparedPrompt.nextPosition;

            for (int i = 0; i < maxNewTokens; i++) {
                if (tokenizer.isTerminalToken(nextToken)) {
                    break;
                }

                generatedTokens.add(nextToken);
                String decodedText = tokenizer.decode(toIntArray(generatedTokens));
                if (decodedText.startsWith(emittedText)) {
                    String delta = decodedText.substring(emittedText.length());
                    if (!delta.isEmpty()) {
                        tokenCallback.accept(delta);
                    }
                    emittedText = decodedText;
                } else {
                    String tokenText = tokenizer.decodeToken(nextToken);
                    if (!tokenText.isEmpty()) {
                        tokenCallback.accept(tokenText);
                    }
                    emittedText = decodedText;
                }

                if (nextToken == lastToken) {
                    repeatedTokenStreak++;
                } else {
                    repeatedTokenStreak = 1;
                    lastToken = nextToken;
                }
                if (repeatedTokenStreak >= MAX_REPEAT_STREAK) {
                    log.warn("Stopping native LiteRT-LM decode after {} repeated tokens ({})",
                            repeatedTokenStreak, nextToken);
                    break;
                }

                recentTokens.addLast(nextToken);
                while (recentTokens.size() > MAX_RECENT_TOKENS) {
                    recentTokens.removeFirst();
                }

                if (position >= maxSeqLen) {
                    log.warn("Stopping native LiteRT-LM decode at max sequence length {}", maxSeqLen);
                    break;
                }

                logits = runDecodeStep(nextToken, position);
                nextToken = selectNextToken(logits, recentTokens, temperature, topK, topP, repeatPenalty);
                position++;
            }
        } catch (Exception e) {
            throw new RuntimeException("Native LiteRT-LM generation failed: " + e.getMessage(), e);
        }
    }

    private PromptPreparation preparePrompt(int[] inputIds) {
        int prefetchedPrefixTokens = prefillPromptPrefix(inputIds);
        for (int i = prefetchedPrefixTokens; i < inputIds.length - 1; i++) {
            runDecodeStep(inputIds[i], i);
        }
        byte[] logits = runDecodeStep(inputIds[inputIds.length - 1], inputIds.length - 1);
        return new PromptPreparation(logits, inputIds.length);
    }

    private void initializeCompiledState(SegmentCandidate mainCandidate,
                                         SegmentCandidate utilityCandidate,
                                         SegmentCandidate embedderCandidate,
                                         SegmentCandidate perLayerCandidate,
                                         int mainAccelerators,
                                         int auxiliaryAccelerators,
                                         NativeGraphLayout graphLayout) throws IOException {
        this.environment = bindings.createEnvironment(arena);
        MemorySegment mainOptions = createCompilationOptions(mainAccelerators);
        MemorySegment auxiliaryOptions = auxiliaryAccelerators == mainAccelerators
                ? mainOptions
                : createCompilationOptions(auxiliaryAccelerators);

        this.graphLayout = graphLayout;
        this.mainSegment = loadCompiledSegment("main", mainCandidate.offset, mainCandidate.size, mainOptions);
        this.utilitySegment = utilityCandidate == null
                ? null
                : loadCompiledSegment("utility", utilityCandidate.offset, utilityCandidate.size, auxiliaryOptions);
        this.embedderSegment = loadCompiledSegment("embedder", embedderCandidate.offset, embedderCandidate.size,
                auxiliaryOptions);
        this.perLayerSegment = loadCompiledSegment("per-layer", perLayerCandidate.offset, perLayerCandidate.size,
                auxiliaryOptions);

        this.decodeSignature = requireSignature(mainSegment, "decode");
        this.prefill128Signature = mainSegment.signatures.get("prefill_128");
        this.prefill1024Signature = mainSegment.signatures.get("prefill_1024");
        this.verifySignature = mainSegment.signatures.get("verify");
        if (graphLayout == NativeGraphLayout.RAW_SELF_CONTAINED) {
            this.utilityDecodeRopeSignature = null;
            this.embedderDecodeSignature = requireSignature(embedderSegment, "embedder");
            this.perLayerDecodeSignature = requireSignature(perLayerSegment, "per_layer_embedder");
        } else {
            this.utilityDecodeRopeSignature = requireSignature(utilitySegment, "decode_rope");
            this.embedderDecodeSignature = requireSignature(embedderSegment, "decode_embedder");
            this.perLayerDecodeSignature = requireSignature(perLayerSegment, "decode_per_layer_embedder");
        }
        this.logitsOutputSpec = requireOutputSpec(decodeSignature, "logits");
    }

    private MemorySegment createCompilationOptions(int accelerators) {
        MemorySegment newOptions = bindings.createOptions(arena);
        bindings.setOptionsHardwareAccelerators(newOptions, accelerators);
        compilationOptions.add(newOptions);
        if (options == null) {
            options = newOptions;
        }
        return newOptions;
    }

    private List<SegmentCandidate> collectSegmentCandidates(LiteRTContainerParser.ContainerInfo info) throws IOException {
        List<SegmentCandidate> candidates = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (LiteRTContainerParser.SubModelEntry entry : info.subModels()) {
            String key = entry.offset() + ":" + entry.size();
            if (!seen.add(key)) {
                continue;
            }
            candidates.add(inspectSegment(entry.modelType(), entry.offset(), entry.size()));
        }
        if (info.tfliteOffset() > 0 && info.tfliteSize() > 0) {
            String primaryKey = info.tfliteOffset() + ":" + info.tfliteSize();
            if (seen.add(primaryKey)) {
                candidates.add(inspectSegment("primary", info.tfliteOffset(), info.tfliteSize()));
            }
        }
        return candidates;
    }

    private SegmentCandidate preferLargerCandidate(SegmentCandidate current, SegmentCandidate candidate) {
        if (current == null || candidate.size > current.size) {
            return candidate;
        }
        return current;
    }

    private boolean isMacOs() {
        return System.getProperty("os.name", "").toLowerCase().contains("mac");
    }

    private SegmentCandidate inspectSegment(String label, long offset, long size) throws IOException {
        try (Arena inspectArena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, offset, size);
            MemorySegment model = bindings.createModelFromBuffer(MemorySegment.ofBuffer(mapped), size, inspectArena);
            try {
                int signatureCount = bindings.getNumModelSignatures(model, inspectArena);
                List<String> signatureKeys = new ArrayList<>(signatureCount);
                for (int i = 0; i < signatureCount; i++) {
                    MemorySegment signature = bindings.getModelSignature(model, i, inspectArena);
                    signatureKeys.add(bindings.getSignatureKey(signature, inspectArena));
                }
                return new SegmentCandidate(label, offset, size, signatureKeys);
            } finally {
                bindings.destroyModel(model);
            }
        }
    }

    private CompiledSegment loadCompiledSegment(String label, long offset, long size) throws IOException {
        return loadCompiledSegment(label, offset, size, options);
    }

    private CompiledSegment loadCompiledSegment(String label, long offset, long size,
                                                MemorySegment compileOptions) throws IOException {
        try (FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, offset, size);
            MemorySegment model = bindings.createModelFromBuffer(MemorySegment.ofBuffer(mapped), size, arena);
            MemorySegment compiledModel = bindings.createCompiledModel(environment, model, compileOptions, arena);
            log.info("LiteRT-LM segment '{}' fully accelerated: {}",
                    label, bindings.isFullyAccelerated(compiledModel, arena));
            Map<String, SignatureSpec> signatures = inspectCompiledSignatures(model, compiledModel);
            return new CompiledSegment(label, mapped, model, compiledModel, signatures);
        }
    }

    private Map<String, SignatureSpec> inspectCompiledSignatures(MemorySegment model, MemorySegment compiledModel) {
        Map<String, SignatureSpec> signatures = new LinkedHashMap<>();
        int signatureCount = bindings.getNumModelSignatures(model, arena);
        for (int sigIndex = 0; sigIndex < signatureCount; sigIndex++) {
            MemorySegment signature = bindings.getModelSignature(model, sigIndex, arena);
            String signatureKey = bindings.getSignatureKey(signature, arena);
            List<TensorSpec> inputs = new ArrayList<>();
            List<TensorSpec> outputs = new ArrayList<>();

            int inputCount = bindings.getNumSignatureInputs(signature, arena);
            for (int i = 0; i < inputCount; i++) {
                String name = bindings.getSignatureInputName(signature, i, arena);
                MemorySegment tensor = bindings.getSignatureInputTensorByIndex(signature, i, arena);
                MemorySegment ranked = bindings.getRankedTensorType(tensor, arena);
                long reqBytes = bindings.getTensorBufferRequirementsBufferSize(
                        bindings.getCompiledModelInputBufferRequirements(compiledModel, sigIndex, i, arena), arena);
                inputs.add(new TensorSpec(name, parseTypeId(ranked), parseDims(ranked), reqBytes,
                        ranked.asSlice(0, 128).toArray(JAVA_BYTE)));
            }

            int outputCount = bindings.getNumSignatureOutputs(signature, arena);
            for (int i = 0; i < outputCount; i++) {
                String name = bindings.getSignatureOutputName(signature, i, arena);
                MemorySegment tensor = bindings.getSignatureOutputTensorByIndex(signature, i, arena);
                MemorySegment ranked = bindings.getRankedTensorType(tensor, arena);
                long reqBytes = bindings.getTensorBufferRequirementsBufferSize(
                        bindings.getCompiledModelOutputBufferRequirements(compiledModel, sigIndex, i, arena), arena);
                outputs.add(new TensorSpec(name, parseTypeId(ranked), parseDims(ranked), reqBytes,
                        ranked.asSlice(0, 128).toArray(JAVA_BYTE)));
            }

            signatures.put(signatureKey, new SignatureSpec(signatureKey, sigIndex, inputs, outputs));
        }
        return signatures;
    }

    private void prepareCaches() {
        kvCaches.clear();
        kvCacheSpecs.clear();
        zeroParamTensor = null;
        maxSeqLen = 0;

        for (TensorSpec input : decodeSignature.inputs) {
            if (!input.name.startsWith("kv_cache_")) {
                if (input.name.equals("param_tensor")) {
                    zeroParamTensor = new byte[input.reqBytes];
                }
                continue;
            }
            kvCaches.put(input.name, new byte[(int) input.reqBytes]);
            kvCacheSpecs.put(input.name, input);
            if (input.dims.length == 4) {
                if (input.name.contains("_k_")) {
                    maxSeqLen = Math.max(maxSeqLen, input.dims[3]);
                } else if (input.name.contains("_v_")) {
                    maxSeqLen = Math.max(maxSeqLen, input.dims[2]);
                }
            }
        }

        if (maxSeqLen <= 0) {
            maxSeqLen = 4096;
        }
    }

    private void resetCaches() {
        for (byte[] cache : kvCaches.values()) {
            Arrays.fill(cache, (byte) 0);
        }
        if (mainSegment != null && decodeSignature != null) {
            mainSegment.resetStatefulBuffers(bindings, decodeSignature, kvCaches.keySet());
        }
    }

    private byte[] runDecodeStep(int tokenId, int position) {
        if (graphLayout == NativeGraphLayout.RAW_SELF_CONTAINED) {
            return runRawDecodeStep(tokenId, position);
        }
        return runHelperDecodeStep(tokenId, position);
    }

    private int prefillPromptPrefix(int[] inputIds) {
        return prefillPromptPrefix(inputIds, Math.max(0, inputIds.length - 1));
    }

    private int prefillPromptPrefix(int[] inputIds, int prefixTokens) {
        if (graphLayout != NativeGraphLayout.RAW_SELF_CONTAINED) {
            return 0;
        }
        if (!shouldUseRawPrefill(prefixTokens)) {
            return 0;
        }

        int maxPrefillTokens = Math.max(0, prefixTokens - rawPrefillDecodeTailTokens());
        if (maxPrefillTokens < 4) {
            return 0;
        }

        int consumed = 0;
        while (consumed < maxPrefillTokens) {
            SignatureSpec signature = selectPrefillSignature(maxPrefillTokens - consumed);
            if (signature == null) {
                break;
            }
            int capacity = prefillTokenCapacity(signature);
            int chunkTokens = Math.min(maxPrefillTokens - consumed, capacity);
            if (chunkTokens < 4) {
                break;
            }
            runRawPrefillChunk(inputIds, consumed, chunkTokens, signature);
            consumed += chunkTokens;
        }
        return consumed;
    }

    static boolean shouldUseRawPrefill(int prefixTokens) {
        return prefixTokens >= rawPrefillMinTokens();
    }

    static int defaultRawPrefillMinTokens() {
        return MIN_RAW_PREFILL_TOKENS;
    }

    static int defaultRawPrefillDecodeTailTokens() {
        return RAW_PREFILL_DECODE_TAIL_TOKENS;
    }

    static boolean experimentalRawLiteRtLmEnabled() {
        return Boolean.getBoolean(ENABLE_EXPERIMENTAL_RAW_LITERTLM_PROPERTY);
    }

    static boolean rawLiteRtLmDisabled() {
        return Boolean.getBoolean(DISABLE_RAW_LITERTLM_PROPERTY);
    }

    static boolean rawLiteRtLmGpuOnMacOsAllowed() {
        return Boolean.getBoolean(ALLOW_RAW_LITERTLM_GPU_ON_MACOS_PROPERTY);
    }

    static boolean rawLiteRtLmAuxGpuOnMacOsEnabled() {
        return Boolean.getBoolean(ENABLE_RAW_LITERTLM_AUX_GPU_ON_MACOS_PROPERTY);
    }

    static boolean statefulKvBuffersEnabled() {
        return !"false".equalsIgnoreCase(System.getProperty(ENABLE_STATEFUL_KV_BUFFERS_PROPERTY, "true"));
    }

    static int rawPrefillMinTokens() {
        String configured = System.getProperty(RAW_PREFILL_MIN_TOKENS_PROPERTY);
        if (configured == null || configured.isBlank()) {
            return MIN_RAW_PREFILL_TOKENS;
        }
        try {
            int parsed = Integer.parseInt(configured.trim());
            if (parsed < 0) {
                log.warn("Ignoring negative {} value '{}'; using default {}",
                        RAW_PREFILL_MIN_TOKENS_PROPERTY, configured, MIN_RAW_PREFILL_TOKENS);
                return MIN_RAW_PREFILL_TOKENS;
            }
            return parsed;
        } catch (NumberFormatException e) {
            log.warn("Ignoring invalid {} value '{}'; using default {}",
                    RAW_PREFILL_MIN_TOKENS_PROPERTY, configured, MIN_RAW_PREFILL_TOKENS);
            return MIN_RAW_PREFILL_TOKENS;
        }
    }

    static int rawPrefillDecodeTailTokens() {
        String configured = System.getProperty(RAW_PREFILL_DECODE_TAIL_TOKENS_PROPERTY);
        if (configured == null || configured.isBlank()) {
            return RAW_PREFILL_DECODE_TAIL_TOKENS;
        }
        try {
            int parsed = Integer.parseInt(configured.trim());
            if (parsed < 0) {
                log.warn("Ignoring negative {} value '{}'; using default {}",
                        RAW_PREFILL_DECODE_TAIL_TOKENS_PROPERTY, configured, RAW_PREFILL_DECODE_TAIL_TOKENS);
                return RAW_PREFILL_DECODE_TAIL_TOKENS;
            }
            return parsed;
        } catch (NumberFormatException e) {
            log.warn("Ignoring invalid {} value '{}'; using default {}",
                    RAW_PREFILL_DECODE_TAIL_TOKENS_PROPERTY, configured, RAW_PREFILL_DECODE_TAIL_TOKENS);
            return RAW_PREFILL_DECODE_TAIL_TOKENS;
        }
    }

    private SignatureSpec selectPrefillSignature(int remainingTokens) {
        int prefill128Capacity = prefillTokenCapacity(prefill128Signature);
        if (prefill1024Signature != null && remainingTokens > prefill128Capacity) {
            int prefill1024Capacity = prefillTokenCapacity(prefill1024Signature);
            if (prefill1024Capacity >= 4) {
                return prefill1024Signature;
            }
        }
        if (prefill128Capacity >= 4) {
            return prefill128Signature;
        }
        if (prefill1024Signature != null && prefillTokenCapacity(prefill1024Signature) >= 4) {
            return prefill1024Signature;
        }
        return null;
    }

    private int prefillTokenCapacity(SignatureSpec signature) {
        if (signature == null) {
            return 0;
        }
        TensorSpec inputPosSpec = requireInputSpec(signature, "input_pos");
        return inputPosSpec.reqBytes / Integer.BYTES;
    }

    private void runRawPrefillChunk(int[] inputIds, int startOffset, int tokenCount, SignatureSpec signature) {
        TensorSpec embeddingsSpec = requireInputSpec(signature, "embeddings");
        TensorSpec perLayerEmbeddingsSpec = requireInputSpec(signature, "per_layer_embeddings");
        TensorSpec inputPosSpec = requireInputSpec(signature, "input_pos");
        TensorSpec maskSpec = requireInputSpec(signature, "mask");

        Map<String, byte[]> decodeInputs = new HashMap<>(kvCaches);
        packRawPrefillEmbeddings(decodeInputs, inputIds, startOffset, tokenCount, embeddingsSpec, perLayerEmbeddingsSpec);
        decodeInputs.put("input_pos", encodeInt32Range(inputPosSpec.reqBytes / Integer.BYTES, tokenCount, startOffset));
        decodeInputs.put("mask", buildRawPrefillMask(maskSpec, tokenCount, startOffset));

        TensorSpec paramSpec = findOptionalInputSpec(signature, "param_tensor");
        if (paramSpec != null) {
            decodeInputs.put("param_tensor", buildZeroTensor(paramSpec));
        }

        Map<String, byte[]> decodeOutputs = mainSegment.run(bindings, environment, decodeInputs, signature);
        updateCachesFromOutputs(decodeOutputs);
    }

    private byte[] runHelperDecodeStep(int tokenId, int position) {
        byte[] tokenIds = encodeInt32Array(tokenId);
        byte[] scalarPos = encodeInt32Array(position);

        Map<String, byte[]> embeddings = embedderSegment.run(bindings, environment, tokenIds, embedderDecodeSignature);
        Map<String, byte[]> perLayerEmbeddings = perLayerSegment.run(bindings, environment, tokenIds, perLayerDecodeSignature);
        Map<String, byte[]> ropes = utilitySegment.run(bindings, environment,
                Map.of("input_pos", scalarPos),
                utilityDecodeRopeSignature);

        Map<String, byte[]> decodeInputs = new HashMap<>();
        decodeInputs.putAll(kvCaches);
        decodeInputs.put("embeddings", requireOutput(embeddings, "embeddings"));
        decodeInputs.put("per_layer_embeddings", requireOutput(perLayerEmbeddings, "embeddings"));
        decodeInputs.put("mask_local", buildDecodeMask(
                requireInputSpec(decodeSignature, "mask_local"), position, true));
        decodeInputs.put("mask_global", buildDecodeMask(
                requireInputSpec(decodeSignature, "mask_global"), position, false));
        decodeInputs.put("pos_emb_cos", requireOutput(ropes, "pos_emb_cos"));
        decodeInputs.put("pos_emb_sin", requireOutput(ropes, "pos_emb_sin"));
        decodeInputs.put("pos_emb_local_cos", requireOutput(ropes, "pos_emb_local_cos"));
        decodeInputs.put("pos_emb_local_sin", requireOutput(ropes, "pos_emb_local_sin"));

        Map<String, byte[]> decodeOutputs = mainSegment.run(bindings, environment, decodeInputs, decodeSignature);
        updateCachesFromSlices(decodeOutputs, position);
        return requireOutput(decodeOutputs, "logits");
    }

    private byte[] runRawDecodeStep(int tokenId, int position) {
        byte[] tokenIds = encodeInt32Array(tokenId);
        byte[] scalarPos = encodeInt32Array(position);

        Map<String, byte[]> embeddings = embedderSegment.run(bindings, environment, tokenIds, embedderDecodeSignature);
        Map<String, byte[]> perLayerEmbeddings = perLayerSegment.run(bindings, environment, tokenIds, perLayerDecodeSignature);

        Map<String, byte[]> decodeInputs = new HashMap<>();
        decodeInputs.put("embeddings", requireOutput(embeddings, "embeddings"));
        decodeInputs.put("per_layer_embeddings", requireOutput(perLayerEmbeddings, "embeddings"));
        decodeInputs.put("input_pos", scalarPos);
        decodeInputs.put("mask", buildRawDecodeMask(requireInputSpec(decodeSignature, "mask"), position));
        if (zeroParamTensor != null) {
            decodeInputs.put("param_tensor", zeroParamTensor);
        }

        boolean statefulCache = statefulKvBuffersEnabled()
                && mainSegment.canUseStatefulBuffers(decodeSignature, kvCaches.keySet());
        if (statefulCache && !statefulKvLogEmitted) {
            log.info("Using in-place LiteRT KV cache buffers for raw decode fallback");
            statefulKvLogEmitted = true;
        }
        Map<String, byte[]> decodeOutputs = statefulCache
                ? mainSegment.runWithStatefulBuffers(bindings, environment, decodeInputs, kvCaches, decodeSignature)
                : mainSegment.run(bindings, environment, withKvCaches(decodeInputs), decodeSignature);
        if (!statefulCache) {
            updateCachesFromOutputs(decodeOutputs);
        }
        return requireOutput(decodeOutputs, "logits");
    }

    private Map<String, byte[]> withKvCaches(Map<String, byte[]> inputs) {
        Map<String, byte[]> withCaches = new HashMap<>(kvCaches);
        withCaches.putAll(inputs);
        return withCaches;
    }

    private void packRawPrefillEmbeddings(Map<String, byte[]> decodeInputs,
                                          int[] inputIds,
                                          int startOffset,
                                          int tokenCount,
                                          TensorSpec embeddingsSpec,
                                          TensorSpec perLayerEmbeddingsSpec) {
        byte[] packedEmbeddings = new byte[embeddingsSpec.reqBytes];
        byte[] packedPerLayerEmbeddings = new byte[perLayerEmbeddingsSpec.reqBytes];
        int embeddingOffset = 0;
        int perLayerOffset = 0;

        for (int i = 0; i < tokenCount; i++) {
            int tokenId = inputIds[startOffset + i];
            byte[] tokenIds = encodeInt32Array(tokenId);

            Map<String, byte[]> embeddings = embedderSegment.run(bindings, environment, tokenIds, embedderDecodeSignature);
            Map<String, byte[]> perLayerEmbeddings = perLayerSegment.run(bindings, environment, tokenIds, perLayerDecodeSignature);

            byte[] embedding = requireOutput(embeddings, "embeddings");
            byte[] perLayerEmbedding = requireOutput(perLayerEmbeddings, "embeddings");
            if (embeddingOffset + embedding.length > packedEmbeddings.length) {
                throw new IllegalArgumentException("Prefill embeddings overflow at token " + i
                        + " for signature " + embeddingsSpec.name);
            }
            if (perLayerOffset + perLayerEmbedding.length > packedPerLayerEmbeddings.length) {
                throw new IllegalArgumentException("Prefill per-layer embeddings overflow at token " + i
                        + " for signature " + perLayerEmbeddingsSpec.name);
            }

            System.arraycopy(embedding, 0, packedEmbeddings, embeddingOffset, embedding.length);
            System.arraycopy(perLayerEmbedding, 0, packedPerLayerEmbeddings, perLayerOffset, perLayerEmbedding.length);
            embeddingOffset += embedding.length;
            perLayerOffset += perLayerEmbedding.length;
        }

        decodeInputs.put("embeddings", packedEmbeddings);
        decodeInputs.put("per_layer_embeddings", packedPerLayerEmbeddings);
    }

    private void updateCachesFromSlices(Map<String, byte[]> decodeOutputs, int position) {
        for (Map.Entry<String, byte[]> entry : decodeOutputs.entrySet()) {
            String sliceName = entry.getKey();
            if (!sliceName.startsWith("kv_slice_")) {
                continue;
            }

            String cacheName = sliceName.replace("kv_slice_", "kv_cache_");
            byte[] cacheBytes = kvCaches.get(cacheName);
            TensorSpec cacheSpec = kvCacheSpecs.get(cacheName);
            TensorSpec sliceSpec = decodeSignature.outputByName.get(sliceName);
            if (cacheBytes == null || cacheSpec == null || sliceSpec == null) {
                continue;
            }

            if (cacheName.contains("_k_")) {
                copyKeySlice(cacheBytes, cacheSpec.dims, entry.getValue(), sliceSpec.dims, position);
            } else if (cacheName.contains("_v_")) {
                copyValueSlice(cacheBytes, cacheSpec.dims, entry.getValue(), sliceSpec.dims, position);
            }
        }
    }

    private void updateCachesFromOutputs(Map<String, byte[]> decodeOutputs) {
        for (Map.Entry<String, byte[]> entry : decodeOutputs.entrySet()) {
            byte[] cacheBytes = kvCaches.get(entry.getKey());
            if (cacheBytes == null) {
                continue;
            }
            byte[] updated = entry.getValue();
            if (updated.length != cacheBytes.length) {
                throw new IllegalArgumentException("KV cache output '" + entry.getKey() + "' expected "
                        + cacheBytes.length + " bytes but got " + updated.length);
            }
            System.arraycopy(updated, 0, cacheBytes, 0, cacheBytes.length);
        }
    }

    private void copyKeySlice(byte[] cacheBytes, int[] cacheDims, byte[] sliceBytes, int[] sliceDims, int position) {
        if (cacheDims.length != 4 || sliceDims.length != 4) {
            return;
        }
        int depth = cacheDims[2];
        int seqLen = cacheDims[3];
        int stepCount = sliceDims[3];
        if (position + stepCount > seqLen) {
            throw new IllegalArgumentException("Key cache slice exceeds sequence length");
        }

        for (int d = 0; d < depth; d++) {
            int srcOffset = d * stepCount;
            int dstOffset = d * seqLen + position;
            System.arraycopy(sliceBytes, srcOffset, cacheBytes, dstOffset, stepCount);
        }
    }

    private void copyValueSlice(byte[] cacheBytes, int[] cacheDims, byte[] sliceBytes, int[] sliceDims, int position) {
        if (cacheDims.length != 4 || sliceDims.length != 4) {
            return;
        }
        int seqLen = cacheDims[2];
        int depth = cacheDims[3];
        int stepCount = sliceDims[2];
        if (position + stepCount > seqLen) {
            throw new IllegalArgumentException("Value cache slice exceeds sequence length");
        }

        for (int t = 0; t < stepCount; t++) {
            int srcOffset = t * depth;
            int dstOffset = (position + t) * depth;
            System.arraycopy(sliceBytes, srcOffset, cacheBytes, dstOffset, depth);
        }
    }

    private int selectNextToken(byte[] logitsBytes,
                                Deque<Integer> recentTokens,
                                double temperature,
                                int topK,
                                double topP,
                                double repeatPenalty) {
        if (logitsBytes == null || logitsBytes.length == 0) {
            return LiteRTTokenizer.EOS_TOKEN_ID;
        }

        Set<Integer> repeated = recentTokens.isEmpty() ? Set.of() : new HashSet<>(recentTokens);
        if (temperature <= 1.0e-4d || topK <= 1) {
            return greedyToken(logitsBytes, repeated, repeatPenalty, logitsOutputSpec);
        }
        return sampleToken(logitsBytes, repeated, temperature, topK, topP, repeatPenalty, logitsOutputSpec);
    }

    private int greedyToken(byte[] logitsBytes, Set<Integer> repeated, double repeatPenalty, TensorSpec logitsSpec) {
        int bestToken = 0;
        float bestScore = Float.NEGATIVE_INFINITY;
        int vocabSize = logitsElementCount(logitsBytes, logitsSpec);
        for (int i = 0; i < vocabSize; i++) {
            float score = adjustedScore(readLogit(logitsBytes, logitsSpec, i), repeated.contains(i), repeatPenalty);
            if (score > bestScore) {
                bestScore = score;
                bestToken = i;
            }
        }
        return bestToken;
    }

    private int sampleToken(byte[] logitsBytes, Set<Integer> repeated,
                            double temperature, int topK, double topP, double repeatPenalty,
                            TensorSpec logitsSpec) {
        int vocabSize = logitsElementCount(logitsBytes, logitsSpec);
        int candidateCount = Math.max(1, Math.min(topK, vocabSize));
        float[] scores = new float[candidateCount];
        int[] ids = new int[candidateCount];
        Arrays.fill(scores, Float.NEGATIVE_INFINITY);
        Arrays.fill(ids, -1);

        for (int i = 0; i < vocabSize; i++) {
            float score = adjustedScore(readLogit(logitsBytes, logitsSpec, i), repeated.contains(i), repeatPenalty);
            int minIndex = 0;
            for (int j = 1; j < candidateCount; j++) {
                if (scores[j] < scores[minIndex]) {
                    minIndex = j;
                }
            }
            if (score > scores[minIndex]) {
                scores[minIndex] = score;
                ids[minIndex] = i;
            }
        }

        for (int i = 0; i < candidateCount - 1; i++) {
            int best = i;
            for (int j = i + 1; j < candidateCount; j++) {
                if (scores[j] > scores[best]) {
                    best = j;
                }
            }
            if (best != i) {
                float tmpScore = scores[i];
                scores[i] = scores[best];
                scores[best] = tmpScore;
                int tmpId = ids[i];
                ids[i] = ids[best];
                ids[best] = tmpId;
            }
        }

        float max = Float.NEGATIVE_INFINITY;
        int valid = 0;
        for (int i = 0; i < candidateCount; i++) {
            if (ids[i] < 0) {
                continue;
            }
            scores[i] = (float) (scores[i] / Math.max(temperature, 1.0e-4d));
            max = Math.max(max, scores[i]);
            valid++;
        }
        if (valid == 0) {
            return LiteRTTokenizer.EOS_TOKEN_ID;
        }

        double total = 0.0d;
        double[] probs = new double[valid];
        int[] probIds = new int[valid];
        for (int i = 0, out = 0; i < candidateCount; i++) {
            if (ids[i] < 0) {
                continue;
            }
            double p = Math.exp(scores[i] - max);
            probs[out] = p;
            probIds[out] = ids[i];
            total += p;
            out++;
        }

        for (int i = 0; i < valid; i++) {
            probs[i] /= total;
        }

        double cumulative = 0.0d;
        int limited = valid;
        if (topP > 0.0d && topP < 1.0d) {
            for (int i = 0; i < valid; i++) {
                cumulative += probs[i];
                if (cumulative >= topP) {
                    limited = i + 1;
                    break;
                }
            }
        }

        double renorm = 0.0d;
        for (int i = 0; i < limited; i++) {
            renorm += probs[i];
        }
        double sample = ThreadLocalRandom.current().nextDouble();
        double running = 0.0d;
        for (int i = 0; i < limited; i++) {
            running += probs[i] / renorm;
            if (sample <= running) {
                return probIds[i];
            }
        }
        return probIds[limited - 1];
    }

    private float adjustedScore(byte rawLogit, boolean repeated, double repeatPenalty) {
        float score = rawLogit;
        if (repeated && repeatPenalty > 1.0d) {
            float penalty = (float) repeatPenalty;
            score = score >= 0.0f ? score / penalty : score * penalty;
        }
        return score;
    }

    private float adjustedScore(float rawLogit, boolean repeated, double repeatPenalty) {
        float score = rawLogit;
        if (repeated && repeatPenalty > 1.0d) {
            float penalty = (float) repeatPenalty;
            score = score >= 0.0f ? score / penalty : score * penalty;
        }
        return score;
    }

    private byte[] buildDecodeMask(TensorSpec spec, int position, boolean localWindow) {
        if (spec.dims.length != 4 || spec.dims[0] != 1 || spec.dims[1] != 1 || spec.dims[2] != 1) {
            throw new IllegalArgumentException("Unexpected decode mask shape for " + spec.name);
        }
        int totalPositions = spec.dims[3];
        int cacheSlots = totalPositions - 1;
        byte[] mask = new byte[spec.reqBytes];
        int currentSlot = totalPositions - 1;

        int start = 0;
        if (localWindow) {
            int slidingWindow = cacheSlots;
            start = Math.max(0, position - slidingWindow + 1);
        }
        int endExclusive = Math.min(position, cacheSlots);
        for (int i = start; i < endExclusive; i++) {
            mask[i] = 1;
        }
        mask[currentSlot] = 1;
        return mask;
    }

    private byte[] buildRawDecodeMask(TensorSpec spec, int position) {
        if (spec.dims.length != 4 || spec.dims[0] != 1 || spec.dims[1] != 1 || spec.dims[2] != 1) {
            throw new IllegalArgumentException("Unexpected raw decode mask shape for " + spec.name);
        }
        byte[] mask = new byte[spec.reqBytes];
        int currentSlot = spec.dims[3] - 1;
        int endExclusive = Math.min(position, currentSlot);
        for (int i = 0; i < endExclusive; i++) {
            mask[i] = 1;
        }
        mask[currentSlot] = 1;
        return mask;
    }

    private byte[] buildRawPrefillMask(TensorSpec spec, int tokenCount, int positionOffset) {
        if (spec.dims.length != 4 || spec.dims[0] != 1 || spec.dims[1] != 1) {
            throw new IllegalArgumentException("Unexpected raw prefill mask shape for " + spec.name);
        }
        int rows = spec.dims[2];
        int cols = spec.dims[3];
        byte[] mask = new byte[spec.reqBytes];
        int activeRows = Math.min(tokenCount, rows);
        for (int row = 0; row < activeRows; row++) {
            int visibleCols = Math.min(cols, positionOffset + row + 1);
            for (int col = 0; col < visibleCols; col++) {
                mask[row * cols + col] = 1;
            }
        }
        return mask;
    }

    private byte[] buildZeroTensor(TensorSpec spec) {
        if (zeroParamTensor != null && zeroParamTensor.length == spec.reqBytes) {
            return zeroParamTensor;
        }
        return new byte[spec.reqBytes];
    }

    private TensorSpec findOptionalInputSpec(SignatureSpec signature, String name) {
        for (TensorSpec input : signature.inputs) {
            if (input.name.equals(name)) {
                return input;
            }
        }
        return null;
    }

    private int logitsElementCount(byte[] logitsBytes, TensorSpec logitsSpec) {
        LiteRTNativeBindings.LitertType type = logitsSpec == null
                ? LiteRTNativeBindings.LitertType.INT8
                : LiteRTNativeBindings.LitertType.fromInt(logitsSpec.typeId);
        return switch (type) {
            case FLOAT32 -> logitsBytes.length / Float.BYTES;
            default -> logitsBytes.length;
        };
    }

    private float readLogit(byte[] logitsBytes, TensorSpec logitsSpec, int index) {
        LiteRTNativeBindings.LitertType type = logitsSpec == null
                ? LiteRTNativeBindings.LitertType.INT8
                : LiteRTNativeBindings.LitertType.fromInt(logitsSpec.typeId);
        return switch (type) {
            case FLOAT32 -> ByteBuffer.wrap(logitsBytes, index * Float.BYTES, Float.BYTES)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .getFloat();
            case UINT8 -> Byte.toUnsignedInt(logitsBytes[index]);
            default -> logitsBytes[index];
        };
    }

    private static byte[] requireOutput(Map<String, byte[]> outputs, String name) {
        byte[] value = outputs.get(name);
        if (value == null) {
            throw new IllegalArgumentException("Missing LiteRT-LM output: " + name);
        }
        return value;
    }

    private static TensorSpec requireInputSpec(SignatureSpec signature, String name) {
        for (TensorSpec input : signature.inputs) {
            if (input.name.equals(name)) {
                return input;
            }
        }
        throw new IllegalArgumentException("Missing LiteRT-LM input spec: " + name);
    }

    private static TensorSpec requireOutputSpec(SignatureSpec signature, String name) {
        for (TensorSpec output : signature.outputs) {
            if (output.name.equals(name)) {
                return output;
            }
        }
        throw new IllegalArgumentException("Missing LiteRT-LM output spec: " + name);
    }

    private static SignatureSpec requireSignature(CompiledSegment segment, String key) {
        SignatureSpec signature = segment.signatures.get(key);
        if (signature == null) {
            throw new IllegalArgumentException("Missing LiteRT-LM signature '" + key + "' in " + segment.label);
        }
        return signature;
    }

    private static int parseTypeId(MemorySegment rankedTensorType) {
        return rankedTensorType.get(JAVA_INT, 0);
    }

    private static int[] parseDims(MemorySegment rankedTensorType) {
        int rank = rankedTensorType.get(JAVA_INT, Integer.BYTES);
        int[] dims = new int[Math.max(rank, 0)];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = rankedTensorType.get(JAVA_INT, (long) (i + 2) * Integer.BYTES);
        }
        return dims;
    }

    private static byte[] encodeInt32Array(int... values) {
        byte[] bytes = new byte[values.length * Integer.BYTES];
        for (int i = 0; i < values.length; i++) {
            int value = values[i];
            int offset = i * Integer.BYTES;
            bytes[offset] = (byte) (value);
            bytes[offset + 1] = (byte) (value >>> 8);
            bytes[offset + 2] = (byte) (value >>> 16);
            bytes[offset + 3] = (byte) (value >>> 24);
        }
        return bytes;
    }

    private static byte[] encodeInt32Range(int length, int activeCount, int start) {
        byte[] bytes = new byte[length * Integer.BYTES];
        for (int i = 0; i < length; i++) {
            int value = i < activeCount ? start + i : 0;
            int offset = i * Integer.BYTES;
            bytes[offset] = (byte) (value);
            bytes[offset + 1] = (byte) (value >>> 8);
            bytes[offset + 2] = (byte) (value >>> 16);
            bytes[offset + 3] = (byte) (value >>> 24);
        }
        return bytes;
    }

    private static int[] toIntArray(List<Integer> values) {
        int[] ids = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            ids[i] = values.get(i);
        }
        return ids;
    }

    private void destroyCompiledState() {
        try {
            if (mainSegment != null) {
                mainSegment.close(bindings);
                bindings.destroyCompiledModel(mainSegment.compiledModel);
                bindings.destroyModel(mainSegment.model);
                mainSegment = null;
            }
            if (utilitySegment != null) {
                utilitySegment.close(bindings);
                bindings.destroyCompiledModel(utilitySegment.compiledModel);
                bindings.destroyModel(utilitySegment.model);
                utilitySegment = null;
            }
            if (embedderSegment != null) {
                embedderSegment.close(bindings);
                bindings.destroyCompiledModel(embedderSegment.compiledModel);
                bindings.destroyModel(embedderSegment.model);
                embedderSegment = null;
            }
            if (perLayerSegment != null) {
                perLayerSegment.close(bindings);
                bindings.destroyCompiledModel(perLayerSegment.compiledModel);
                bindings.destroyModel(perLayerSegment.model);
                perLayerSegment = null;
            }
            for (MemorySegment compiledOptions : compilationOptions) {
                bindings.destroyOptions(compiledOptions);
            }
            compilationOptions.clear();
            options = null;
            if (environment != null) {
                bindings.destroyEnvironment(environment);
                environment = null;
            }
        } catch (Exception e) {
            log.warn("Failed cleaning up native LiteRT-LM state: {}", e.getMessage());
        }
    }

    @Override
    public void close() {
        destroyCompiledState();
        arena.close();
    }

    private static final class CompiledSegment {
        final String label;
        final MappedByteBuffer mapped;
        final MemorySegment model;
        final MemorySegment compiledModel;
        final Map<String, SignatureSpec> signatures;
        final Arena bufferArena = Arena.ofConfined();
        final Map<Integer, SignatureBuffers> reusableBuffers = new HashMap<>();
        final Map<Integer, SignatureBuffers> reusableStatefulBuffers = new HashMap<>();

        private CompiledSegment(String label, MappedByteBuffer mapped, MemorySegment model,
                                MemorySegment compiledModel, Map<String, SignatureSpec> signatures) {
            this.label = label;
            this.mapped = mapped;
            this.model = model;
            this.compiledModel = compiledModel;
            this.signatures = signatures;
        }

        Map<String, byte[]> run(LiteRTNativeBindings bindings, MemorySegment environment,
                                byte[] singleInput, SignatureSpec signature) {
            String inputName = signature.inputs.get(0).name;
            return run(bindings, environment, Map.of(inputName, singleInput), signature);
        }

        synchronized Map<String, byte[]> run(LiteRTNativeBindings bindings, MemorySegment environment,
                                Map<String, byte[]> inputs, SignatureSpec signature) {
            SignatureBuffers signatureBuffers = reusableBuffers.computeIfAbsent(signature.index,
                    ignored -> SignatureBuffers.create(bindings, environment, compiledModel,
                            signature, bufferArena, Set.of()));
            try (Arena inferArena = Arena.ofConfined()) {
                try {
                    for (int i = 0; i < signature.inputs.size(); i++) {
                        TensorSpec spec = signature.inputs.get(i);
                        byte[] bytes = inputs.get(spec.name);
                        if (bytes == null) {
                            throw new IllegalArgumentException("Missing input '" + spec.name + "' for " + signature.key);
                        }
                        if (bytes.length != spec.reqBytes) {
                            throw new IllegalArgumentException("Input '" + spec.name + "' for " + signature.key
                                    + " expected " + spec.reqBytes + " bytes but got " + bytes.length);
                        }
                        MemorySegment tensorBuffer = signatureBuffers.inputBuffers.get(i);
                        MemorySegment locked = bindings.lockTensorBuffer(
                                tensorBuffer, LiteRTNativeBindings.kLiteRtTensorBufferLockModeWrite, inferArena);
                        MemorySegment.copy(MemorySegment.ofArray(bytes), 0, locked.reinterpret(bytes.length), 0, bytes.length);
                        bindings.unlockTensorBuffer(tensorBuffer);
                    }

                    bindings.runCompiledModel(compiledModel, signature.index,
                            signatureBuffers.inputArray, signature.inputs.size(),
                            signatureBuffers.outputArray, signature.outputs.size());

                    Map<String, byte[]> outputs = new LinkedHashMap<>();
                    for (int i = 0; i < signature.outputs.size(); i++) {
                        TensorSpec spec = signature.outputs.get(i);
                        MemorySegment tensorBuffer = signatureBuffers.outputBuffers.get(i);
                        long outputSize = bindings.getTensorBufferSize(tensorBuffer, inferArena);
                        MemorySegment locked = bindings.lockTensorBuffer(
                                tensorBuffer, LiteRTNativeBindings.kLiteRtTensorBufferLockModeRead, inferArena);
                        byte[] bytes = locked.reinterpret(outputSize).toArray(JAVA_BYTE);
                        bindings.unlockTensorBuffer(tensorBuffer);
                        outputs.put(spec.name, bytes);
                    }
                    return outputs;
                } catch (RuntimeException e) {
                    throw new RuntimeException("LiteRT-LM segment '" + label + "' signature '"
                            + signature.key + "' failed: " + e.getMessage(), e);
                }
            }
        }

        boolean canUseStatefulBuffers(SignatureSpec signature, Set<String> statefulNames) {
            if (statefulNames.isEmpty()) {
                return false;
            }
            for (String name : statefulNames) {
                TensorSpec input = signature.inputByName.get(name);
                TensorSpec output = signature.outputByName.get(name);
                if (input == null || output == null || input.reqBytes != output.reqBytes) {
                    return false;
                }
            }
            return true;
        }

        synchronized Map<String, byte[]> runWithStatefulBuffers(LiteRTNativeBindings bindings,
                                                               MemorySegment environment,
                                                               Map<String, byte[]> inputs,
                                                               Map<String, byte[]> statefulInputs,
                                                               SignatureSpec signature) {
            SignatureBuffers signatureBuffers = reusableStatefulBuffers.computeIfAbsent(signature.index,
                    ignored -> SignatureBuffers.create(bindings, environment, compiledModel,
                            signature, bufferArena, statefulInputs.keySet()));
            try (Arena inferArena = Arena.ofConfined()) {
                try {
                    for (int i = 0; i < signature.inputs.size(); i++) {
                        TensorSpec spec = signature.inputs.get(i);
                        byte[] bytes = inputs.get(spec.name);
                        boolean stateful = false;
                        if (bytes == null) {
                            bytes = statefulInputs.get(spec.name);
                            stateful = bytes != null;
                        }
                        if (bytes == null) {
                            throw new IllegalArgumentException("Missing input '" + spec.name + "' for " + signature.key);
                        }
                        if (bytes.length != spec.reqBytes) {
                            throw new IllegalArgumentException("Input '" + spec.name + "' for " + signature.key
                                    + " expected " + spec.reqBytes + " bytes but got " + bytes.length);
                        }
                        if (stateful && signatureBuffers.isStatefulInitialized(spec.name)) {
                            continue;
                        }
                        MemorySegment tensorBuffer = signatureBuffers.inputBuffers.get(i);
                        MemorySegment locked = bindings.lockTensorBuffer(
                                tensorBuffer, LiteRTNativeBindings.kLiteRtTensorBufferLockModeWrite, inferArena);
                        MemorySegment.copy(MemorySegment.ofArray(bytes), 0, locked.reinterpret(bytes.length), 0, bytes.length);
                        bindings.unlockTensorBuffer(tensorBuffer);
                        if (stateful) {
                            signatureBuffers.markStatefulInitialized(spec.name);
                        }
                    }

                    bindings.runCompiledModel(compiledModel, signature.index,
                            signatureBuffers.inputArray, signature.inputs.size(),
                            signatureBuffers.outputArray, signature.outputs.size());

                    Map<String, byte[]> outputs = new LinkedHashMap<>();
                    for (int i = 0; i < signature.outputs.size(); i++) {
                        TensorSpec spec = signature.outputs.get(i);
                        if (statefulInputs.containsKey(spec.name)) {
                            continue;
                        }
                        MemorySegment tensorBuffer = signatureBuffers.outputBuffers.get(i);
                        long outputSize = bindings.getTensorBufferSize(tensorBuffer, inferArena);
                        MemorySegment locked = bindings.lockTensorBuffer(
                                tensorBuffer, LiteRTNativeBindings.kLiteRtTensorBufferLockModeRead, inferArena);
                        byte[] bytes = locked.reinterpret(outputSize).toArray(JAVA_BYTE);
                        bindings.unlockTensorBuffer(tensorBuffer);
                        outputs.put(spec.name, bytes);
                    }
                    signatureBuffers.copyStatefulOutputsToInputs(bindings, inferArena, statefulInputs.keySet());
                    return outputs;
                } catch (RuntimeException e) {
                    throw new RuntimeException("LiteRT-LM segment '" + label + "' signature '"
                            + signature.key + "' failed: " + e.getMessage(), e);
                }
            }
        }

        synchronized void resetStatefulBuffers(LiteRTNativeBindings bindings,
                                               SignatureSpec signature,
                                               Set<String> statefulNames) {
            SignatureBuffers signatureBuffers = reusableBuffers.get(signature.index);
            if (signatureBuffers != null) {
                signatureBuffers.resetStatefulBuffers(bindings, signature, statefulNames);
            }
            signatureBuffers = reusableStatefulBuffers.get(signature.index);
            if (signatureBuffers != null) {
                signatureBuffers.resetStatefulBuffers(bindings, signature, statefulNames);
            }
        }

        void close(LiteRTNativeBindings bindings) {
            for (SignatureBuffers buffers : reusableBuffers.values()) {
                buffers.close(bindings);
            }
            for (SignatureBuffers buffers : reusableStatefulBuffers.values()) {
                buffers.close(bindings);
            }
            reusableBuffers.clear();
            reusableStatefulBuffers.clear();
            bufferArena.close();
        }
    }

    private static final class SignatureBuffers {
        final MemorySegment inputArray;
        final MemorySegment outputArray;
        final List<MemorySegment> inputBuffers;
        final List<MemorySegment> outputBuffers;
        final Map<String, Integer> inputIndexByName;
        final Map<String, Integer> outputIndexByName;
        final Set<Integer> aliasedOutputIndexes;
        final Set<String> aliasedStatefulNames;
        final Set<String> initializedStatefulBuffers = new HashSet<>();

        private SignatureBuffers(MemorySegment inputArray, MemorySegment outputArray,
                                 List<MemorySegment> inputBuffers, List<MemorySegment> outputBuffers,
                                 Map<String, Integer> inputIndexByName,
                                 Map<String, Integer> outputIndexByName,
                                 Set<Integer> aliasedOutputIndexes,
                                 Set<String> aliasedStatefulNames) {
            this.inputArray = inputArray;
            this.outputArray = outputArray;
            this.inputBuffers = inputBuffers;
            this.outputBuffers = outputBuffers;
            this.inputIndexByName = inputIndexByName;
            this.outputIndexByName = outputIndexByName;
            this.aliasedOutputIndexes = Set.copyOf(aliasedOutputIndexes);
            this.aliasedStatefulNames = Set.copyOf(aliasedStatefulNames);
        }

        static SignatureBuffers create(LiteRTNativeBindings bindings,
                                       MemorySegment environment,
                                       MemorySegment compiledModel,
                                       SignatureSpec signature,
                                       Arena arena,
                                       Set<String> aliasedOutputNames) {
            MemorySegment inputArray = arena.allocate(ADDRESS, signature.inputs.size());
            MemorySegment outputArray = arena.allocate(ADDRESS, signature.outputs.size());
            List<MemorySegment> inputBuffers = new ArrayList<>(signature.inputs.size());
            List<MemorySegment> outputBuffers = new ArrayList<>(signature.outputs.size());
            Map<String, Integer> inputIndexByName = new HashMap<>();
            Map<String, Integer> outputIndexByName = new HashMap<>();
            Set<Integer> aliasedOutputIndexes = new HashSet<>();
            Set<String> aliasedStatefulNames = new HashSet<>();

            for (int i = 0; i < signature.inputs.size(); i++) {
                TensorSpec spec = signature.inputs.get(i);
                inputIndexByName.put(spec.name, i);
                MemorySegment bufReq = bindings.getCompiledModelInputBufferRequirements(
                        compiledModel, signature.index, i, arena);
                MemorySegment tensorType = arena.allocate(spec.rankedTensorType.length, 8);
                tensorType.copyFrom(MemorySegment.ofArray(spec.rankedTensorType));
                MemorySegment tensorBuffer = bindings.createManagedTensorBufferFromRequirements(
                        environment, tensorType, bufReq, arena);
                inputBuffers.add(tensorBuffer);
                inputArray.setAtIndex(ADDRESS, i, tensorBuffer);
            }

            for (int i = 0; i < signature.outputs.size(); i++) {
                TensorSpec spec = signature.outputs.get(i);
                outputIndexByName.put(spec.name, i);
                Integer aliasedInputIndex = inputIndexByName.get(spec.name);
                if (aliasedInputIndex != null && aliasedOutputNames.contains(spec.name)) {
                    TensorSpec inputSpec = signature.inputs.get(aliasedInputIndex);
                    if (inputSpec.typeId == spec.typeId
                            && inputSpec.reqBytes == spec.reqBytes
                            && Arrays.equals(inputSpec.dims, spec.dims)) {
                        MemorySegment tensorBuffer = inputBuffers.get(aliasedInputIndex);
                        outputBuffers.add(tensorBuffer);
                        outputArray.setAtIndex(ADDRESS, i, tensorBuffer);
                        aliasedOutputIndexes.add(i);
                        aliasedStatefulNames.add(spec.name);
                        continue;
                    }
                }
                MemorySegment bufReq = bindings.getCompiledModelOutputBufferRequirements(
                        compiledModel, signature.index, i, arena);
                MemorySegment tensorType = arena.allocate(spec.rankedTensorType.length, 8);
                tensorType.copyFrom(MemorySegment.ofArray(spec.rankedTensorType));
                MemorySegment tensorBuffer = bindings.createManagedTensorBufferFromRequirements(
                        environment, tensorType, bufReq, arena);
                outputBuffers.add(tensorBuffer);
                outputArray.setAtIndex(ADDRESS, i, tensorBuffer);
            }

            return new SignatureBuffers(inputArray, outputArray,
                    inputBuffers, outputBuffers,
                    Map.copyOf(inputIndexByName), Map.copyOf(outputIndexByName),
                    aliasedOutputIndexes, aliasedStatefulNames);
        }

        boolean isStatefulInitialized(String name) {
            return initializedStatefulBuffers.contains(name);
        }

        void markStatefulInitialized(String name) {
            initializedStatefulBuffers.add(name);
        }

        void copyStatefulOutputsToInputs(LiteRTNativeBindings bindings,
                                         Arena copyArena,
                                         Set<String> statefulNames) {
            for (String name : statefulNames) {
                if (aliasedStatefulNames.contains(name)) {
                    markStatefulInitialized(name);
                    continue;
                }
                Integer inputIndex = inputIndexByName.get(name);
                Integer outputIndex = outputIndexByName.get(name);
                if (inputIndex == null || outputIndex == null) {
                    continue;
                }
                MemorySegment inputBuffer = inputBuffers.get(inputIndex);
                MemorySegment outputBuffer = outputBuffers.get(outputIndex);
                long inputSize = bindings.getTensorBufferSize(inputBuffer, copyArena);
                long outputSize = bindings.getTensorBufferSize(outputBuffer, copyArena);
                if (inputSize != outputSize) {
                    throw new IllegalArgumentException("Stateful tensor '" + name + "' input/output buffer size mismatch");
                }
                MemorySegment outputLocked = bindings.lockTensorBuffer(
                        outputBuffer, LiteRTNativeBindings.kLiteRtTensorBufferLockModeRead, copyArena);
                MemorySegment inputLocked = bindings.lockTensorBuffer(
                        inputBuffer, LiteRTNativeBindings.kLiteRtTensorBufferLockModeWrite, copyArena);
                MemorySegment.copy(outputLocked.reinterpret(outputSize), 0,
                        inputLocked.reinterpret(inputSize), 0, inputSize);
                bindings.unlockTensorBuffer(inputBuffer);
                bindings.unlockTensorBuffer(outputBuffer);
                markStatefulInitialized(name);
            }
        }

        void resetStatefulBuffers(LiteRTNativeBindings bindings,
                                  SignatureSpec signature,
                                  Set<String> statefulNames) {
            try (Arena resetArena = Arena.ofConfined()) {
                for (String name : statefulNames) {
                    zeroBuffer(bindings, resetArena, inputIndexByName.get(name), inputBuffers);
                    if (!aliasedStatefulNames.contains(name)) {
                        zeroBuffer(bindings, resetArena, outputIndexByName.get(name), outputBuffers);
                    }
                }
                initializedStatefulBuffers.clear();
            }
        }

        private void zeroBuffer(LiteRTNativeBindings bindings,
                                Arena resetArena,
                                Integer index,
                                List<MemorySegment> buffers) {
            if (index == null) {
                return;
            }
            MemorySegment tensorBuffer = buffers.get(index);
            long size = bindings.getTensorBufferSize(tensorBuffer, resetArena);
            MemorySegment locked = bindings.lockTensorBuffer(
                    tensorBuffer, LiteRTNativeBindings.kLiteRtTensorBufferLockModeWrite, resetArena);
            locked.reinterpret(size).fill((byte) 0);
            bindings.unlockTensorBuffer(tensorBuffer);
        }

        void close(LiteRTNativeBindings bindings) {
            for (MemorySegment buffer : inputBuffers) {
                bindings.destroyTensorBuffer(buffer);
            }
            for (int i = 0; i < outputBuffers.size(); i++) {
                if (aliasedOutputIndexes.contains(i)) {
                    continue;
                }
                MemorySegment buffer = outputBuffers.get(i);
                bindings.destroyTensorBuffer(buffer);
            }
        }
    }

    private static final class SignatureSpec {
        final String key;
        final int index;
        final List<TensorSpec> inputs;
        final List<TensorSpec> outputs;
        final Map<String, TensorSpec> inputByName;
        final Map<String, TensorSpec> outputByName;

        private SignatureSpec(String key, int index, List<TensorSpec> inputs, List<TensorSpec> outputs) {
            this.key = key;
            this.index = index;
            this.inputs = List.copyOf(inputs);
            this.outputs = List.copyOf(outputs);
            Map<String, TensorSpec> namedInputs = new HashMap<>();
            for (TensorSpec input : inputs) {
                namedInputs.put(input.name, input);
            }
            Map<String, TensorSpec> namedOutputs = new HashMap<>();
            for (TensorSpec output : outputs) {
                namedOutputs.put(output.name, output);
            }
            this.inputByName = Map.copyOf(namedInputs);
            this.outputByName = Map.copyOf(namedOutputs);
        }
    }

    private static final class TensorSpec {
        final String name;
        final int typeId;
        final int[] dims;
        final int reqBytes;
        final byte[] rankedTensorType;

        private TensorSpec(String name, int typeId, int[] dims, long reqBytes, byte[] rankedTensorType) {
            this.name = name;
            this.typeId = typeId;
            this.dims = dims;
            this.reqBytes = Math.toIntExact(reqBytes);
            this.rankedTensorType = rankedTensorType;
        }
    }

    private static final class SegmentCandidate {
        final String label;
        final long offset;
        final long size;
        final Set<String> signatureKeys;

        private SegmentCandidate(String label, long offset, long size, List<String> signatureKeys) {
            this.label = label;
            this.offset = offset;
            this.size = size;
            this.signatureKeys = Set.copyOf(signatureKeys);
        }
    }

    private record PromptPreparation(byte[] logits, int nextPosition) {}

    private enum NativeGraphLayout {
        HELPER_SEGMENTS,
        RAW_SELF_CONTAINED
    }
}
