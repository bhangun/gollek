/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.plugin.runner.onnx;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.subscription.MultiEmitter;
import org.jboss.logging.Logger;
import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerSession;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.StreamingDecoder;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * ONNX Runtime runner session implementation.
 * 
 * <p>Wraps an OrtSession to execute inference using FFM-based ONNX Runtime bindings.</p>
 * 
 * @since 2.1.0
 */
public class OnnxRunnerSession implements RunnerSession {

    private static final Logger LOG = Logger.getLogger(OnnxRunnerSession.class);

    private final String sessionId;
    private final String modelPath;
    private final Map<String, Object> config;
    private final RunnerPlugin runner;
    private final AtomicBoolean active = new AtomicBoolean(true);
    private final ModelInfo modelInfo;

    private final OnnxRuntimeBinding ort;
    private final MemorySegment ortEnv;
    private final MemorySegment ortSession;
    private final MemorySegment memInfo;
    private final String[] inputNames;
    private final String[] outputNames;
    private final Tokenizer tokenizer;

    public OnnxRunnerSession(String modelPath, Map<String, Object> config,
            String executionProvider, int intraOpThreads, int interOpThreads) {
        this.sessionId = UUID.randomUUID().toString();
        this.modelPath = modelPath;
        this.config = config;
        this.runner = new OnnxRunnerPlugin();
        
        this.ort = OnnxRuntimeBinding.getInstance();
        this.tokenizer = loadTokenizer(modelPath);
        
        try {
            this.ortEnv = ort.createEnv("gollek-session-" + sessionId);
            
            MemorySegment opts = ort.createSessionOptions();
            ort.setIntraOpNumThreads(opts, intraOpThreads);
            ort.setInterOpNumThreads(opts, interOpThreads);
            
            // Attach execution provider if not CPU
            if (!"cpu".equalsIgnoreCase(executionProvider)) {
                if ("cuda".equalsIgnoreCase(executionProvider)) {
                    ort.appendCudaProvider(opts, 0);
                } else if ("rocm".equalsIgnoreCase(executionProvider)) {
                    ort.appendRocmProvider(opts, 0);
                } else if ("coreml".equalsIgnoreCase(executionProvider)) {
                    ort.appendCoreMlProvider(opts, 0);
                }
            }
            
            this.ortSession = ort.createSession(ortEnv, modelPath, opts);
            ort.releaseSessionOptions(opts);
            
            int numInputs = (int) ort.getInputCount(ortSession);
            int numOutputs = (int) ort.getOutputCount(ortSession);
            this.inputNames = new String[numInputs];
            this.outputNames = new String[numOutputs];
            
            for (int i = 0; i < numInputs; i++) {
                inputNames[i] = ort.getInputName(ortSession, i);
            }
            for (int i = 0; i < numOutputs; i++) {
                outputNames[i] = ort.getOutputName(ortSession, i);
            }
            
            this.memInfo = ort.createCpuMemoryInfo();
            this.modelInfo = loadModelInfo(modelPath);

            LOG.infof("Created ONNX session %s for model: %s (inputs: %d, outputs: %d, vocab: %d)", 
                sessionId, modelPath, numInputs, numOutputs, tokenizer.vocabSize());
            
        } catch (Exception e) {
            LOG.errorf("Failed to initialize ONNX session: %s", e.getMessage());
            throw new RuntimeException("Failed to initialize ONNX session", e);
        }
    }

    @Override
    public String getSessionId() {
        return sessionId;
    }

    @Override
    public String getModelPath() {
        return modelPath;
    }

    @Override
    public RunnerPlugin getRunner() {
        return runner;
    }

    @Override
    public Uni<InferenceResponse> infer(InferenceRequest request) {
        if (!active.get()) {
            return Uni.createFrom().failure(new IllegalStateException("Session is closed"));
        }

        return Uni.createFrom().item(() -> executeInference(request))
                .onFailure().invoke(e -> LOG.errorf("ONNX inference failed: %s", e.getMessage()));
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        if (!active.get()) {
            return Multi.createFrom().failure(new IllegalStateException("Session is closed"));
        }

        return Multi.createFrom().emitter(emitter -> executeStreamingInference(request, emitter));
    }

    @Override
    public Map<String, Object> getConfig() {
        return config;
    }

    @Override
    public boolean isActive() {
        return active.get();
    }

    @Override
    public void close() {
        if (active.compareAndSet(true, false)) {
            LOG.infof("Closing ONNX session %s", sessionId);
            releaseModel();
        }
    }

    @Override
    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    private ModelInfo loadModelInfo(String modelPath) {
        Path path = Paths.get(modelPath);
        String modelName = path.getFileName().toString();
        return new ModelInfo(modelName, "onnx", 0, 4096, 768, 
            Map.of("format", "ONNX", "session_id", sessionId));
    }

    private Tokenizer loadTokenizer(String modelPath) {
        Path path = Paths.get(modelPath);
        Path dir = path.getParent();
        if (dir != null) {
            try {
                return TokenizerFactory.load(dir, null);
            } catch (IOException e) {
                LOG.errorf(e, "OnnxRunnerSession: failed to load tokenizer for [%s]", modelPath);
            }
        }
        throw new RuntimeException("Tokenizer loading failed for " + modelPath);
    }

    private String getPrompt(InferenceRequest request) {
        if (request.getMessages() == null || request.getMessages().isEmpty()) {
            return "";
        }
        // Return content of the last user message as prompt
        return request.getMessages().stream()
                .filter(m -> m.getRole() == tech.kayys.gollek.spi.Message.Role.USER)
                .reduce((first, second) -> second)
                .map(tech.kayys.gollek.spi.Message::getContent)
                .orElse(request.getMessages().get(request.getMessages().size() - 1).getContent());
    }

    private InferenceResponse executeInference(InferenceRequest request) {
        long t0 = System.currentTimeMillis();
        String prompt = getPrompt(request);
        long[] inputIds = tokenizer.encode(prompt, EncodeOptions.defaultOptions());
        int maxNewTokens = extractInt(request.getParameters(), "max_tokens", 128);
        
        try (Arena arena = Arena.ofShared()) {
            StringBuilder sb = new StringBuilder();
            List<Long> currentTokens = new ArrayList<>();
            for (long id : inputIds) currentTokens.add(id);

            for (int step = 0; step < maxNewTokens; step++) {
                if (!active.get()) break;

                long seqLen = currentTokens.size();
                
                // Build input tensors
                MemorySegment ids64 = arena.allocate(seqLen * 8L, 8);
                for (int i = 0; i < seqLen; i++) {
                    ids64.setAtIndex(ValueLayout.JAVA_LONG, i, currentTokens.get(i));
                }
                
                MemorySegment mask64 = arena.allocate(seqLen * 8L, 8);
                for (int i = 0; i < seqLen; i++) {
                    mask64.setAtIndex(ValueLayout.JAVA_LONG, i, 1L);
                }

                MemorySegment idsValue = ort.createTensorWithData(memInfo, ids64, new long[]{1, seqLen}, OnnxRuntimeBinding.ONNX_TENSOR_INT64);
                MemorySegment maskValue = ort.createTensorWithData(memInfo, mask64, new long[]{1, seqLen}, OnnxRuntimeBinding.ONNX_TENSOR_INT64);

                // Run inference - assuming standard LLM input names
                MemorySegment[] outputs = ort.run(ortSession, MemorySegment.NULL, 
                    new String[]{"input_ids", "attention_mask"}, 
                    new MemorySegment[]{idsValue, maskValue}, 
                    new String[]{"logits"});

                // Extract logits and sample (greedy)
                float[] logits = ort.getTensorDataFloat(outputs[0], tokenizer.vocabSize());
                int nextTokenId = sampleGreedy(logits);

                // Release iteration values
                ort.releaseValue(idsValue);
                ort.releaseValue(maskValue);
                for (MemorySegment out : outputs) ort.releaseValue(out);

                if (nextTokenId == tokenizer.eosTokenId() || nextTokenId < 0) break;

                String piece = tokenizer.decode(new long[]{nextTokenId}, DecodeOptions.defaultOptions());
                sb.append(piece);
                currentTokens.add((long) nextTokenId);
            }

            return InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .content(sb.toString())
                    .model(modelPath)
                    .inputTokens(inputIds.length)
                    .outputTokens(currentTokens.size() - inputIds.length)
                    .durationMs(System.currentTimeMillis() - t0)
                    .build();
        }
    }

    private void executeStreamingInference(InferenceRequest request, MultiEmitter<? super StreamingInferenceChunk> emitter) {
        long t0 = System.currentTimeMillis();
        String prompt = getPrompt(request);
        long[] inputIds = tokenizer.encode(prompt, EncodeOptions.defaultOptions());
        int maxNewTokens = extractInt(request.getParameters(), "max_tokens", 128);
        
        try (Arena arena = Arena.ofShared()) {
            List<Long> currentTokens = new ArrayList<>();
            for (long id : inputIds) currentTokens.add(id);

            for (int step = 0; step < maxNewTokens; step++) {
                if (!active.get()) break;

                long seqLen = currentTokens.size();
                MemorySegment ids64 = arena.allocate(seqLen * 8L, 8);
                for (int i = 0; i < seqLen; i++) ids64.setAtIndex(ValueLayout.JAVA_LONG, i, currentTokens.get(i));
                
                MemorySegment mask64 = arena.allocate(seqLen * 8L, 8);
                for (int i = 0; i < seqLen; i++) mask64.setAtIndex(ValueLayout.JAVA_LONG, i, 1L);

                MemorySegment idsValue = ort.createTensorWithData(memInfo, ids64, new long[]{1, seqLen}, OnnxRuntimeBinding.ONNX_TENSOR_INT64);
                MemorySegment maskValue = ort.createTensorWithData(memInfo, mask64, new long[]{1, seqLen}, OnnxRuntimeBinding.ONNX_TENSOR_INT64);

                MemorySegment[] outputs = ort.run(ortSession, MemorySegment.NULL, 
                    new String[]{"input_ids", "attention_mask"}, 
                    new MemorySegment[]{idsValue, maskValue}, 
                    new String[]{"logits"});

                float[] logits = ort.getTensorDataFloat(outputs[0], tokenizer.vocabSize());
                int nextTokenId = sampleGreedy(logits);

                ort.releaseValue(idsValue);
                ort.releaseValue(maskValue);
                for (MemorySegment out : outputs) ort.releaseValue(out);

                if (nextTokenId == tokenizer.eosTokenId() || nextTokenId < 0) break;

                String piece = tokenizer.decode(new long[]{nextTokenId}, DecodeOptions.defaultOptions());
                currentTokens.add((long) nextTokenId);

                emitter.emit(new StreamingInferenceChunk(
                    request.getRequestId(),
                    step,
                    ModalityType.TEXT,
                    piece,
                    null,
                    false,
                    null,
                    null,
                    Instant.now(),
                    null
                ));
            }
            
            // Final chunk
            emitter.emit(new StreamingInferenceChunk(
                request.getRequestId(),
                currentTokens.size() - inputIds.length,
                ModalityType.TEXT,
                "",
                null,
                true,
                "stop",
                new StreamingInferenceChunk.ChunkUsage(
                    inputIds.length,
                    currentTokens.size() - inputIds.length,
                    System.currentTimeMillis() - t0
                ),
                Instant.now(),
                null
            ));
            emitter.complete();
        } catch (Exception e) {
            emitter.fail(e);
        }
    }

    private int sampleGreedy(float[] logits) {
        int bestId = -1;
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                bestId = i;
            }
        }
        return bestId;
    }

    private int extractInt(Map<String, Object> params, String key, int defaultValue) {
        if (params == null) return defaultValue;
        Object val = params.get(key);
        if (val instanceof Number n) return n.intValue();
        if (val != null) {
            try { return Integer.parseInt(val.toString()); } catch (Exception ignored) {}
        }
        return defaultValue;
    }

    private void releaseModel() {
        if (ort != null && ort.isNativeAvailable()) {
            if (memInfo != null && !memInfo.equals(MemorySegment.NULL)) ort.releaseMemoryInfo(memInfo);
            if (ortSession != null && !ortSession.equals(MemorySegment.NULL)) ort.releaseSession(ortSession);
            if (ortEnv != null && !ortEnv.equals(MemorySegment.NULL)) ort.releaseEnv(ortEnv);
        }
        LOG.debugf("Released ONNX session resources for %s", sessionId);
    }
}
