package tech.kayys.gollek.inference.llamacpp;

import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ModelFormat;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.Map;

/**
 * Handles GGUF model and context initialization with GPU fallback logic.
 * Responsible for loading models, configuring context parameters, and managing
 * hardware acceleration settings.
 */
public class LlamaCppModelInitializer {

    private static final Logger log = Logger.getLogger(LlamaCppModelInitializer.class);
    private static final long LARGE_MODEL_THRESHOLD = 4L * 1024L * 1024L * 1024L; // 4 GiB

    private final LlamaCppBinding binding;
    private final LlamaCppProviderConfig providerConfig;

    public LlamaCppModelInitializer(LlamaCppBinding binding, LlamaCppProviderConfig providerConfig) {
        this.binding = binding;
        this.providerConfig = providerConfig;
    }

    /**
     * Model initialization result containing all initialized resources and
     * metadata.
     */
    public static class InitializationResult {
        public final MemorySegment model;
        public final MemorySegment context;
        public final int contextSize;
        public final int vocabSize;
        public final int eosToken;
        public final int bosToken;
        public final String chatTemplate;
        public final int runtimeBatchSize;
        public final int activeGpuLayers;

        public InitializationResult(MemorySegment model, MemorySegment context, int contextSize,
                int vocabSize, int eosToken, int bosToken, String chatTemplate,
                int runtimeBatchSize, int activeGpuLayers) {
            this.model = model;
            this.context = context;
            this.contextSize = contextSize;
            this.vocabSize = vocabSize;
            this.eosToken = eosToken;
            this.bosToken = bosToken;
            this.chatTemplate = chatTemplate;
            this.runtimeBatchSize = runtimeBatchSize;
            this.activeGpuLayers = activeGpuLayers;
        }
    }

    /**
     * Initialize the GGUF model and context from manifest.
     */
    public InitializationResult initialize(ModelManifest manifest, Map<String, Object> runnerConfig) {
        log.infof("Initializing GGUF runner for model %s (tenant: %s)",
                manifest.modelId(), manifest.requestId());

        try {
            Path modelPath = resolveModelPath(manifest);
            validateModelFile(modelPath);

            suppressNativeLogsIfNeeded();

            ModelConfig config = buildModelConfig(runnerConfig, modelPath);

            log.infof("Loading GGUF model from: %s", modelPath.toAbsolutePath());
            MemorySegment model = loadModel(modelPath, config);
            MemorySegment context = createContext(modelPath, model, config);

            return buildInitializationResult(model, context, config);
        } catch (Exception e) {
            log.errorf("GGUF initialization error: %s", e.getMessage());
            throw new RuntimeException("Failed to initialize GGUF runner: " + e.getMessage(), e);
        }
    }

    private Path resolveModelPath(ModelManifest manifest) {
        var artifact = manifest.artifacts().get(ModelFormat.GGUF);
        if (artifact == null) {
            throw new RuntimeException("No GGUF artifact found in manifest for model " + manifest.modelId());
        }
        return Path.of(artifact.uri());
    }

    private void validateModelFile(Path modelPath) {
        if (!Files.exists(modelPath)) {
            throw new RuntimeException("Model file not found: " + modelPath);
        }
        if (!hasGgufHeader(modelPath)) {
            throw new RuntimeException(
                    "Model file is not a valid GGUF binary (missing GGUF header): " + modelPath +
                            ". Re-pull or re-convert the model.");
        }
    }

    private void suppressNativeLogsIfNeeded() {
        if (!providerConfig.verboseLogging()) {
            try {
                binding.suppressNativeLogs();
            } catch (Throwable t) {
                log.debugf("Unable to suppress native logs: %s", t.getMessage());
            }
        }
    }

    private ModelConfig buildModelConfig(Map<String, Object> runnerConfig, Path modelPath) {
        int configuredGpuLayers = getIntConfig(runnerConfig, "nGpuLayers",
                providerConfig.gpuEnabled() ? providerConfig.gpuLayers() : 0);
        int configuredThreads = Math.max(1,
                getIntConfig(runnerConfig, "nThreads", providerConfig.threads()));
        int configuredCtx = Math.max(512,
                getIntConfig(runnerConfig, "nCtx", providerConfig.maxContextTokens()));
        int configuredBatch = Math.max(1,
                getIntConfig(runnerConfig, "nBatch", providerConfig.batchSize()));
        boolean useMmap = getBooleanConfig(runnerConfig, "useMmap", providerConfig.mmapEnabled());
        boolean useMlock = getBooleanConfig(runnerConfig, "useMlock", providerConfig.mlockEnabled());

        long modelSizeBytes = safeFileSize(modelPath);
        int activeGpuLayers = adjustGpuLayersForLargeModel(configuredGpuLayers, modelSizeBytes);

        int effectiveBatch = Math.min(configuredBatch, 128);

        if (activeGpuLayers != configuredGpuLayers) {
            log.warnf(
                    "Large GGUF model detected (%.2f GiB). Capping initial GPU layers from %d to %d " +
                            "for faster and safer startup. Set GOLLEK_GGUF_FORCE_GPU_FOR_LARGE_MODEL=true " +
                            "to keep requested layers.",
                    modelSizeBytes / (1024.0 * 1024.0 * 1024.0),
                    configuredGpuLayers,
                    activeGpuLayers);
        }

        return new ModelConfig(
                activeGpuLayers,
                configuredThreads,
                configuredCtx,
                effectiveBatch,
                useMmap,
                useMlock);
    }

    private int adjustGpuLayersForLargeModel(int configuredGpuLayers, long modelSizeBytes) {
        boolean forceGpuForLargeModel = Boolean.parseBoolean(
                System.getProperty(
                        "gollek.gguf.force.gpu.large-model",
                        System.getenv().getOrDefault("GOLLEK_GGUF_FORCE_GPU_FOR_LARGE_MODEL", "false")));

        if (configuredGpuLayers < -1) {
            configuredGpuLayers = -1;
        }

        if (configuredGpuLayers != 0 && modelSizeBytes >= LARGE_MODEL_THRESHOLD && !forceGpuForLargeModel) {
            return configuredGpuLayers == -1 ? 8 : Math.min(configuredGpuLayers, 8);
        }
        return configuredGpuLayers;
    }

    private MemorySegment loadModel(Path modelPath, ModelConfig config) {
        MemorySegment modelParams = binding.getDefaultModelParams();
        binding.setModelParam(modelParams, "n_gpu_layers", config.gpuLayers);
        binding.setModelParam(modelParams, "main_gpu", providerConfig.gpuDeviceId());
        binding.setModelParam(modelParams, "use_mmap", config.useMmap);
        binding.setModelParam(modelParams, "use_direct_io", false);
        binding.setModelParam(modelParams, "use_mlock", config.useMlock);
        binding.setModelParam(modelParams, "check_tensors", false);
        binding.setModelParam(modelParams, "use_extra_bufts", false);
        binding.setModelParam(modelParams, "no_host", false);
        binding.setModelParam(modelParams, "no_alloc", false);

        return binding.loadModel(modelPath.toString(), modelParams);
    }

    private MemorySegment createContext(Path modelPath, MemorySegment model, ModelConfig config) {
        MemorySegment contextParams = binding.getDefaultContextParams();
        binding.setContextParam(contextParams, "n_ctx", config.contextSize);
        binding.setContextParam(contextParams, "n_batch", config.batchSize);
        binding.setContextParam(contextParams, "n_ubatch", config.batchSize);
        binding.setContextParam(contextParams, "n_seq_max", Math.max(1, providerConfig.coalesceSeqMax()));
        binding.setContextParam(contextParams, "n_threads", config.threads);
        binding.setContextParam(contextParams, "n_threads_batch", config.threads);
        binding.setContextParam(contextParams, "offload_kqv", config.gpuLayers != 0);
        binding.setContextParam(contextParams, "flash_attn_type", 0);
        binding.setContextParam(contextParams, "samplers", MemorySegment.NULL);
        binding.setContextParam(contextParams, "n_samplers", 0L);

        try {
            return binding.createContext(model, contextParams);
        } catch (RuntimeException gpuContextError) {
            if (config.gpuLayers == 0) {
                throw gpuContextError;
            }

            log.warnf("GPU context initialization failed (n_gpu_layers=%d). Retrying on CPU. Cause: %s",
                    config.gpuLayers, gpuContextError.getMessage());
            return createContextOnCpu(modelPath, model, config, gpuContextError);
        }
    }

    private MemorySegment createContextOnCpu(Path modelPath, MemorySegment model, ModelConfig config,
            RuntimeException gpuError) {
        binding.freeModel(model);

        MemorySegment cpuModelParams = binding.getDefaultModelParams();
        binding.setModelParam(cpuModelParams, "n_gpu_layers", 0);
        binding.setModelParam(cpuModelParams, "main_gpu", providerConfig.gpuDeviceId());
        binding.setModelParam(cpuModelParams, "use_mmap", config.useMmap);
        binding.setModelParam(cpuModelParams, "use_direct_io", false);
        binding.setModelParam(cpuModelParams, "use_mlock", config.useMlock);
        binding.setModelParam(cpuModelParams, "check_tensors", false);
        binding.setModelParam(cpuModelParams, "use_extra_bufts", false);
        binding.setModelParam(cpuModelParams, "no_host", false);
        binding.setModelParam(cpuModelParams, "no_alloc", false);

        MemorySegment cpuModel = binding.loadModel(
                modelPath.toAbsolutePath().toString(),
                cpuModelParams);

        MemorySegment contextParams = binding.getDefaultContextParams();
        binding.setContextParam(contextParams, "n_ctx", config.contextSize);
        binding.setContextParam(contextParams, "n_batch", config.batchSize);
        binding.setContextParam(contextParams, "n_ubatch", config.batchSize);
        binding.setContextParam(contextParams, "n_seq_max", Math.max(1, providerConfig.coalesceSeqMax()));
        binding.setContextParam(contextParams, "n_threads", config.threads);
        binding.setContextParam(contextParams, "n_threads_batch", config.threads);
        binding.setContextParam(contextParams, "offload_kqv", false);
        binding.setContextParam(contextParams, "flash_attn_type", 0);
        binding.setContextParam(contextParams, "samplers", MemorySegment.NULL);
        binding.setContextParam(contextParams, "n_samplers", 0L);

        return binding.createContext(cpuModel, contextParams);
    }

    private InitializationResult buildInitializationResult(MemorySegment model, MemorySegment context,
            ModelConfig config) {
        int contextSize = binding.getContextSize(context);
        int vocabSize = binding.getVocabSize(model);
        int eosToken = binding.getEosToken(model);
        int bosToken = binding.getBosToken(model);
        String chatTemplate = binding.getModelMetadata(model, "tokenizer.chat_template");

        log.debugf("Loaded chat template: %s", chatTemplate != null ? "Yes" : "No");
        log.debugf("Model initialized: ctx=%d vocab=%d eos=%d bos=%d", contextSize, vocabSize, eosToken, bosToken);
        log.infof("GGUF runtime config: gpu_layers=%d, n_ctx=%d, n_batch=%d, threads=%d",
                config.gpuLayers, config.contextSize, config.batchSize, config.threads);

        return new InitializationResult(
                model,
                context,
                contextSize,
                vocabSize,
                eosToken,
                bosToken,
                chatTemplate,
                config.batchSize,
                config.gpuLayers);
    }

    private boolean hasGgufHeader(Path path) {
        try (var in = Files.newInputStream(path)) {
            byte[] magic = in.readNBytes(4);
            return magic.length == 4
                    && magic[0] == 'G'
                    && magic[1] == 'G'
                    && magic[2] == 'U'
                    && magic[3] == 'F';
        } catch (Exception ignored) {
            return false;
        }
    }

    private long safeFileSize(Path path) {
        try {
            return Files.size(path);
        } catch (Exception ignored) {
            return -1;
        }
    }

    private int getIntConfig(Map<String, Object> config, String key, int defaultValue) {
        if (config == null || !config.containsKey(key)) {
            return defaultValue;
        }
        Object value = config.get(key);
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String s) {
            try {
                return Integer.parseInt(s.trim());
            } catch (NumberFormatException ignored) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    private boolean getBooleanConfig(Map<String, Object> config, String key, boolean defaultValue) {
        if (config == null || !config.containsKey(key)) {
            return defaultValue;
        }
        Object value = config.get(key);
        if (value instanceof Boolean b) {
            return b;
        }
        if (value instanceof String s) {
            return Boolean.parseBoolean(s.trim());
        }
        return defaultValue;
    }

    private static class ModelConfig {
        final int gpuLayers;
        final int threads;
        final int contextSize;
        final int batchSize;
        final boolean useMmap;
        final boolean useMlock;

        ModelConfig(int gpuLayers, int threads, int contextSize, int batchSize,
                boolean useMmap, boolean useMlock) {
            this.gpuLayers = gpuLayers;
            this.threads = threads;
            this.contextSize = contextSize;
            this.batchSize = batchSize;
            this.useMmap = useMmap;
            this.useMlock = useMlock;
        }
    }
}
