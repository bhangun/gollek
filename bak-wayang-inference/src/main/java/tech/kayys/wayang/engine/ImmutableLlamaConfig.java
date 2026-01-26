package tech.kayys.golek.engine;



import java.nio.file.Path;
import java.util.List;

public record ImmutableLlamaConfig(
    String libraryPath,
    String modelPath,
    int gpuLayers,
    boolean useMmap,
    boolean useMlock,
    boolean vocabOnly,
    int contextSize,
    int batchSize,
    int uBatchSize,
    int threads,
    int threadsBatch,
    float ropeFreqBase,
    float ropeFreqScale,
    int ropeScalingType,
    int nSeqMax,
    int poolingType,
    int attentionType,
    boolean embeddings,
    boolean flashAttention,
    boolean offloadKqv,
    float defragThreshold,
    float yarnExtFactor,
    float yarnAttnFactor,
    float yarnBetaFast,
    float yarnBetaSlow,
    int yarnOrigCtx,
    int seed,
    String tensorSplit,
    int mainGpu,
    int splitMode,
    int serverPort,
    String serverHost,
    int requestTimeoutMs,
    int maxConcurrentRequests,
    int maxRequestSize,
    boolean warmupModel,
    int warmupTokens,
    boolean enablePromptCache,
    int promptCacheSize,
    String logLevel,
    boolean enableMetrics,
    boolean enableTracing,
    int maxTokensLimit,
    int maxStopSequences,
    boolean validateInput,
    String pluginsDirectory,
    String dataDirectory,
    List<String> enabledPlugins,
    LlamaConfig.SamplingConfig sampling,
  
    int shutdownTimeoutMs
) implements LlamaConfig {
    
    public ImmutableLlamaConfig {
        // Validate all parameters
        validate();
    }
    
    @Override
    public Path getModelPath() {
        return Path.of(modelPath);
    }
    
    @Override
    public Path getLibraryPath() {
        return Path.of(libraryPath);
    }
    
    @Override
    public boolean isGpuAccelerated() {
        return gpuLayers > 0;
    }
    
    @Override
    public int getEffectiveSeed() {
        return seed == -1 ? (int) System.currentTimeMillis() : seed;
    }
}