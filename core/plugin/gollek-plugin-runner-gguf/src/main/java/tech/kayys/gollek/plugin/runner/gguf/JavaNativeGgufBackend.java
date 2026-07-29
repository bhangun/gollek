package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFParser;
import tech.kayys.gollek.gguf.loader.GGUFReader;
import tech.kayys.gollek.gguf.runtime.GgufBudget;
import tech.kayys.gollek.gguf.runtime.GgufRuntimeProfile;
import tech.kayys.gollek.gguf.runtime.GgufRuntimeProbe;
import tech.kayys.gollek.gguf.runtime.GgufTensorOps;
import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;

import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.TensorFactory;
import tech.kayys.gollek.gguf.model.alkhawarizm.LlamaModel;
import tech.kayys.gollek.gguf.runner.AljabrWeightAdapter;
import tech.kayys.gollek.gguf.model.ModelConfig;

/**
 * Java GGUF backend used for the actively working native-loader path.
 *
 * <p>
 * It intentionally refuses generation for now. That keeps the production
 * runner from silently taking the slow/wrong path while still exposing the
 * Java loader readiness profile for diagnostics and future engine work.
 * </p>
 */
public final class JavaNativeGgufBackend implements GgufBackend {
    private final GGUFModel model;
    private final GgufRuntimeProfile profile;
    private final GgufRuntimeProbe.PreparedMatrixCacheDecision preparedCacheDecision;

    public JavaNativeGgufBackend(Path modelPath) throws Exception {
        long startNanos = System.nanoTime();
        this.model = loadModel(modelPath);
        long loadMillis = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
        this.profile = GgufRuntimeProfile.fromModel(model, Files.size(modelPath), loadMillis);
        this.preparedCacheDecision = prepareDecoderMatrixCaches(model);
    }

    JavaNativeGgufBackend(GGUFModel model, long modelBytes, long loadMillis) {
        this.model = model;
        this.profile = GgufRuntimeProfile.fromModel(model, modelBytes, loadMillis);
        this.preparedCacheDecision = prepareDecoderMatrixCaches(model);
    }

    @Override
    public String name() {
        return "java";
    }

    @Override
    public Map<String, Object> metadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("backend", name());
        metadata.put("javaStatus", profile.javaStatus());
        metadata.put("architecture", profile.architecture());
        metadata.put("ggufVersion", profile.ggufVersion());
        metadata.put("tensorCount", profile.tensorCount());
        metadata.put("decoderTensorRatio", profile.decoderTensorRatio());
        metadata.put("missingDecoderTensorCount", profile.missingDecoderTensorCount());
        metadata.put("malformedDecoderTensorCount", profile.malformedDecoderTensorCount());
        metadata.put("missingDecoderTensorExamples", profile.missingDecoderTensorExamples());
        metadata.put("malformedDecoderTensorExamples", profile.malformedDecoderTensorExamples());
        metadata.put("knownTensorTypeRatio", profile.knownTensorTypeRatio());
        metadata.put("preparedMatrixCachePlan", preparedCacheDecision.plan().compactSummary());
        metadata.put("preparedMatrixCache", preparedCacheDecision.compactSummary());
        metadata.put("preparedMatrixCacheMode", preparedCacheDecision.mode());
        metadata.put("preparedMatrixCacheEstimatedBytes", preparedCacheDecision.plan().estimatedPreparedBytes());
        metadata.put("preparedMatrixCacheBytes", preparedCacheDecision.stats().cacheBytes());
        metadata.put("preparedMatrixCacheEntries", preparedCacheDecision.stats().cacheEntries());
        metadata.put("loaderReady", profile.knownTensorTypeRatio() > 0.0d);
        metadata.put("decoderTensorsReady", profile.decoderTensorSetComplete());
        metadata.put("rowDotReady", profile.rowDotPrimitivesReady());
        metadata.put("generationReady", false);
        if (profile.modelConfig() != null) {
            metadata.put("modelType", profile.modelConfig().getModelType());
            metadata.put("layers", profile.modelConfig().getNumHiddenLayers());
            metadata.put("hiddenSize", profile.modelConfig().getHiddenSize());
            metadata.put("attentionHeads", profile.modelConfig().getNumAttentionHeads());
            metadata.put("kvHeads", profile.modelConfig().getResolvedNumKvHeads());
            metadata.put("headDim", profile.modelConfig().getResolvedHeadDim());
            metadata.put("contextLength", profile.modelConfig().getMaxPositionEmbeddings());
            metadata.put("vocabSize", profile.modelConfig().getVocabSize());
        }
        return Map.copyOf(metadata);
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request) {
        try {
            System.out.println("Initializing Aljabr Java-native GGUF Engine...");
            java.util.Map<String, tech.kayys.gollek.gguf.core.GGUFTensorInfo> tensorMap = model.tensors().stream()
                    .map(t -> {
                        java.lang.foreign.MemorySegment data = model.segment().asSlice(model.dataStart() + t.offset(),
                                t.sizeInBytes());
                        return new tech.kayys.gollek.gguf.core.GGUFTensorInfo(t.name(), t.shape(),
                                tech.kayys.gollek.gguf.core.GgmlType.fromId(t.typeId()), t.offset(), data);
                    })
                    .collect(java.util.stream.Collectors.toMap(tech.kayys.gollek.gguf.core.GGUFTensorInfo::name,
                            t -> t));
            tech.kayys.gollek.gguf.loader.gguf.GGUFFile file = new tech.kayys.gollek.gguf.loader.gguf.GGUFFile(
                    model.version(), model.metadata(), tensorMap);

            AljabrWeightAdapter weights = new AljabrWeightAdapter(file);
            ModelConfig config = ModelConfig.fromGGUF(file);
            LlamaModel textModel = null;
            try {
                textModel = new LlamaModel(config, weights);
            } catch (NullPointerException e) {
                System.err.println("NPE during LlamaModel initialization. Available tensor keys:");
                tensorMap.keySet().stream().sorted().forEach(System.err::println);
                e.printStackTrace();
                throw e;
            }

            System.out.println("Starting Java-native generation loop...");
            Tensor promptIds = TensorFactory.of(new float[] { 1, 2, 3 }, 1, 3);

            for (int i = 0; i < 5; i++) {
                long startMs = System.currentTimeMillis();
                Tensor logits = textModel.forward(promptIds);
                long elapsed = System.currentTimeMillis() - startMs;
                System.out.println("Step " + (i + 1) + ": textModel.forward() took " + elapsed + "ms. Logits shape: "
                        + logits.shape());
                promptIds = TensorFactory.of(new float[] { 4 }, 1, 1);
            }

            return (RunnerResult<T>) RunnerResult
                    .success("Aljabr native engine successfully executed generation loop with "
                            + textModel.parameterCount() + " parameters.");
        } catch (Exception e) {
            System.err.println("Runtime generation error:");
            e.printStackTrace();
            return RunnerResult.failed("Aljabr native engine failed: " + e.getMessage());
        }
    }

    @Override
    public void close() {
        GgufTensorOps.clearPreparedMatrixCaches(model);
        model.close();
    }

    GgufTensorOps.PreparedMatrixCachePlan preparedCachePlan() {
        return preparedCacheDecision.plan();
    }

    GgufTensorOps.PreparedMatrixCacheStats preparedCacheStats() {
        return preparedCacheDecision.stats();
    }

    String preparedCacheSummary() {
        return preparedCacheDecision.compactSummary();
    }

    private static GgufRuntimeProbe.PreparedMatrixCacheDecision prepareDecoderMatrixCaches(GGUFModel model) {
        int explicitMinRows = Math.max(0, Integer.getInteger("gollek.gguf.java_native.prepare_min_rows", 0));
        if (explicitMinRows > 0) {
            return GgufRuntimeProbe.prepareDecoderMatrixCaches(
                    model,
                    GgufRuntimeProbe.selectDecoderPreparedMatrixCache(
                            model,
                            explicitMinRows,
                            false,
                            1,
                            0L));
        }

        int autoMinRows = Math.max(1, Integer.getInteger("gollek.gguf.java_native.auto_prepare_min_rows", 32));
        long budgetBytes = autoPrepareBudgetBytes();
        return GgufRuntimeProbe.prepareDecoderMatrixCaches(
                model,
                GgufRuntimeProbe.selectDecoderPreparedMatrixCache(
                        model,
                        0,
                        Boolean.parseBoolean(System.getProperty("gollek.gguf.java_native.auto_prepare", "true")),
                        autoMinRows,
                        budgetBytes));
    }

    private static GGUFModel loadModel(Path modelPath) throws Exception {
        Arena arena = Arena.ofShared();
        try (GGUFReader reader = new GGUFReader(modelPath, arena)) {
            return new GGUFParser().parse(reader.segment(), arena);
        } catch (Exception exception) {
            arena.close();
            throw exception;
        } catch (Error error) {
            arena.close();
            throw error;
        }
    }

    private static long autoPrepareBudgetBytes() {
        return GgufBudget.byteSizeProperty(
                "gollek.gguf.java_native.auto_prepare_budget_bytes",
                GgufBudget.defaultAutoPrepareBytes());
    }

}
