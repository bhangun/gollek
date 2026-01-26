package tech.kayys.golek.engine;

import java.nio.file.Path;

public interface LlamaConfig {
    // === CORE CONFIGURATION ===
    // Required paths
    String libraryPath();
    String modelPath();
    
    // Model loading configuration
    int gpuLayers();
    boolean useMmap();
    boolean useMlock();
    boolean vocabOnly();
    
    // Context configuration
    int contextSize();
    int batchSize();
    int uBatchSize();
    
    // Thread configuration
    int threads();
    int threadsBatch();
    
    // Rope configuration
    float ropeFreqBase();
    float ropeFreqScale();
    int ropeScalingType();
    
    // Advanced context parameters
    int nSeqMax();
    int poolingType();
    int attentionType();
    
    // Memory and performance
    boolean embeddings();
    boolean flashAttention();
    boolean offloadKqv();
    float defragThreshold();
    
    // YARN extrapolation
    float yarnExtFactor();
    float yarnAttnFactor();
    float yarnBetaFast();
    float yarnBetaSlow();
    int yarnOrigCtx();
    
    // Randomness
    int seed();
    
    // Tensor split for multi-GPU
    String tensorSplit();
    int mainGpu();
    int splitMode();
    
    // Core sampling configuration
    SamplingConfig sampling();
    
    // === CORE VALIDATION ===
    default void validate() {
        if (libraryPath() == null || libraryPath().trim().isEmpty()) {
            throw new IllegalArgumentException("Library path cannot be null or empty");
        }
        
        if (modelPath() == null || modelPath().trim().isEmpty()) {
            throw new IllegalArgumentException("Model path cannot be null or empty");
        }
        
        if (contextSize() <= 0) {
            throw new IllegalArgumentException("Context size must be positive");
        }
        
        if (batchSize() <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        
        if (uBatchSize() <= 0) {
            throw new IllegalArgumentException("UBatch size must be positive");
        }
        
        if (uBatchSize() > batchSize()) {
            throw new IllegalArgumentException("UBatch size cannot be larger than batch size");
        }
        
        if (threads() <= 0) {
            throw new IllegalArgumentException("Thread count must be positive");
        }
        
        // Validate sampling config
        sampling().validate();
    }
    
    // === CORE HELPER METHODS ===
    default Path getModelPath() {
        return Path.of(modelPath());
    }
    
    default Path getLibraryPath() {
        return Path.of(libraryPath());
    }
    
    default boolean isGpuAccelerated() {
        return gpuLayers() > 0;
    }
    
    default int getEffectiveSeed() {
        return seed() == -1 ? (int) System.currentTimeMillis() : seed();
    }
    
    // === CORE SAMPLING CONFIG ===
    // === CORE SAMPLING CONFIG ===
interface SamplingConfig {
    float temperature();
    int topK();
    float topP();
    float minP();
    float repeatPenalty();
    int repeatLastN();
    float presencePenalty();
    float frequencyPenalty();
    float typicalP();
    float tfsZ();
    float mirostat();
    float mirostatTau();
    float mirostatEta();
    String grammar();
    boolean grammarAcceptToken();
    
    void validate();
    
    // Add builder method
    static SamplingConfigBuilder builder() {
        return new SamplingConfigBuilder();
    }
    
    // Builder class for SamplingConfig
    class SamplingConfigBuilder {
        private float temperature = 1.0f;
        private int topK = 40;
        private float topP = 0.9f;
        private float minP = 0.05f;
        private float repeatPenalty = 1.1f;
        private int repeatLastN = 64;
        private float presencePenalty = 0.0f;
        private float frequencyPenalty = 0.0f;
        private float typicalP = 1.0f;
        private float tfsZ = 1.0f;
        private float mirostat = 0.0f;
        private float mirostatTau = 5.0f;
        private float mirostatEta = 0.1f;
        private String grammar = "";
        private boolean grammarAcceptToken = false;
        
        public SamplingConfigBuilder temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }
        
        public SamplingConfigBuilder topK(int topK) {
            this.topK = topK;
            return this;
        }
        
        public SamplingConfigBuilder topP(float topP) {
            this.topP = topP;
            return this;
        }
        
        public SamplingConfigBuilder minP(float minP) {
            this.minP = minP;
            return this;
        }
        
        public SamplingConfigBuilder repeatPenalty(float repeatPenalty) {
            this.repeatPenalty = repeatPenalty;
            return this;
        }
        
        public SamplingConfigBuilder repeatLastN(int repeatLastN) {
            this.repeatLastN = repeatLastN;
            return this;
        }
        
        public SamplingConfigBuilder presencePenalty(float presencePenalty) {
            this.presencePenalty = presencePenalty;
            return this;
        }
        
        public SamplingConfigBuilder frequencyPenalty(float frequencyPenalty) {
            this.frequencyPenalty = frequencyPenalty;
            return this;
        }
        
        public SamplingConfigBuilder typicalP(float typicalP) {
            this.typicalP = typicalP;
            return this;
        }
        
        public SamplingConfigBuilder tfsZ(float tfsZ) {
            this.tfsZ = tfsZ;
            return this;
        }
        
        public SamplingConfigBuilder mirostat(float mirostat) {
            this.mirostat = mirostat;
            return this;
        }
        
        public SamplingConfigBuilder mirostatTau(float mirostatTau) {
            this.mirostatTau = mirostatTau;
            return this;
        }
        
        public SamplingConfigBuilder mirostatEta(float mirostatEta) {
            this.mirostatEta = mirostatEta;
            return this;
        }
        
        public SamplingConfigBuilder grammar(String grammar) {
            this.grammar = grammar;
            return this;
        }
        
        public SamplingConfigBuilder grammarAcceptToken(boolean grammarAcceptToken) {
            this.grammarAcceptToken = grammarAcceptToken;
            return this;
        }
        
        public SamplingConfig build() {
            return new DefaultSamplingConfig(
                temperature, topK, topP, minP, repeatPenalty, repeatLastN,
                presencePenalty, frequencyPenalty, typicalP, tfsZ, mirostat,
                mirostatTau, mirostatEta, grammar, grammarAcceptToken
            );
        }
    }
    
    // Record implementation for SamplingConfig
    record DefaultSamplingConfig(
        float temperature,
        int topK,
        float topP,
        float minP,
        float repeatPenalty,
        int repeatLastN,
        float presencePenalty,
        float frequencyPenalty,
        float typicalP,
        float tfsZ,
        float mirostat,
        float mirostatTau,
        float mirostatEta,
        String grammar,
        boolean grammarAcceptToken
    ) implements SamplingConfig {
        
        public DefaultSamplingConfig {
            // Validate on construction
            validate();
        }
        
        @Override
        public void validate() {
            if (temperature < 0.0f || temperature > 2.0f) {
                throw new IllegalArgumentException("Temperature must be between 0.0 and 2.0");
            }
            if (topK < 0 && topK != -1) {
                throw new IllegalArgumentException("TopK must be positive or -1");
            }
            if (topP < 0.0f || topP > 1.0f) {
                throw new IllegalArgumentException("TopP must be between 0.0 and 1.0");
            }
            if (minP < 0.0f || minP > 1.0f) {
                throw new IllegalArgumentException("MinP must be between 0.0 and 1.0");
            }
            if (repeatPenalty < 0.0f) {
                throw new IllegalArgumentException("Repeat penalty must be non-negative");
            }
            if (typicalP < 0.0f || typicalP > 1.0f) {
                throw new IllegalArgumentException("TypicalP must be between 0.0 and 1.0");
            }
            if (tfsZ < 0.0f || tfsZ > 1.0f) {
                throw new IllegalArgumentException("TFS-Z must be between 0.0 and 1.0");
            }
            if (mirostat < 0.0f || mirostat > 2.0f) {
                throw new IllegalArgumentException("Mirostat must be 0, 1, or 2");
            }
        }
    }
}
    
    // === CORE BUILDER ===
    static CoreBuilder builder() {
        return new CoreBuilder();
    }
    
    class CoreBuilder {
        // Core fields only
        protected String libraryPath;
        protected String modelPath;
        protected int gpuLayers = 0;
        protected boolean useMmap = true;
        protected boolean useMlock = false;
        protected boolean vocabOnly = false;
        protected int contextSize = 4096;
        protected int batchSize = 512;
        protected int uBatchSize = 512;
        protected int threads = 8;
        protected int threadsBatch = 8;
        protected float ropeFreqBase = 10000.0f;
        protected float ropeFreqScale = 1.0f;
        protected int ropeScalingType = -1;
        protected int nSeqMax = 16;
        protected int poolingType = 0;
        protected int attentionType = 1;
        protected boolean embeddings = false;
        protected boolean flashAttention = false;
        protected boolean offloadKqv = false;
        protected float defragThreshold = -1.0f;
        protected float yarnExtFactor = -1.0f;
        protected float yarnAttnFactor = 1.0f;
        protected float yarnBetaFast = 32.0f;
        protected float yarnBetaSlow = 1.0f;
        protected int yarnOrigCtx = 0;
        protected int seed = -1;
        protected String tensorSplit = "";
        protected int mainGpu = 0;
        protected int splitMode = 0;
        protected SamplingConfig sampling = new DefaultSamplingConfig();
        
        public CoreBuilder libraryPath(String path) { 
            this.libraryPath = path; 
            return this; 
        }
        
        public CoreBuilder modelPath(String path) { 
            this.modelPath = path; 
            return this; 
        }
        
        public CoreBuilder gpuLayers(int layers) { 
            this.gpuLayers = layers; 
            return this; 
        }
        
        public CoreBuilder useMmap(boolean use) { 
            this.useMmap = use; 
            return this; 
        }
        
        public CoreBuilder useMlock(boolean use) { 
            this.useMlock = use; 
            return this; 
        }
        
        public CoreBuilder vocabOnly(boolean vocabOnly) { 
            this.vocabOnly = vocabOnly; 
            return this; 
        }
        
        public CoreBuilder contextSize(int size) { 
            this.contextSize = size; 
            return this; 
        }
        
        public CoreBuilder batchSize(int size) { 
            this.batchSize = size; 
            return this; 
        }
        
        public CoreBuilder uBatchSize(int size) { 
            this.uBatchSize = size; 
            return this; 
        }
        
        public CoreBuilder threads(int threads) { 
            this.threads = threads; 
            return this; 
        }
        
        public CoreBuilder threadsBatch(int threadsBatch) { 
            this.threadsBatch = threadsBatch; 
            return this; 
        }
        
        public CoreBuilder ropeFreqBase(float base) { 
            this.ropeFreqBase = base; 
            return this; 
        }
        
        public CoreBuilder ropeFreqScale(float scale) { 
            this.ropeFreqScale = scale; 
            return this; 
        }
        
        public CoreBuilder ropeScalingType(int type) { 
            this.ropeScalingType = type; 
            return this; 
        }
        
        public CoreBuilder nSeqMax(int nSeqMax) { 
            this.nSeqMax = nSeqMax; 
            return this; 
        }
        
        public CoreBuilder poolingType(int type) { 
            this.poolingType = type; 
            return this; 
        }
        
        public CoreBuilder attentionType(int type) { 
            this.attentionType = type; 
            return this; 
        }
        
        public CoreBuilder embeddings(boolean embeddings) { 
            this.embeddings = embeddings; 
            return this; 
        }
        
        public CoreBuilder flashAttention(boolean flashAttention) { 
            this.flashAttention = flashAttention; 
            return this; 
        }
        
        public CoreBuilder offloadKqv(boolean offloadKqv) { 
            this.offloadKqv = offloadKqv; 
            return this; 
        }
        
        public CoreBuilder defragThreshold(float threshold) { 
            this.defragThreshold = threshold; 
            return this; 
        }
        
        public CoreBuilder yarnExtFactor(float factor) { 
            this.yarnExtFactor = factor; 
            return this; 
        }
        
        public CoreBuilder yarnAttnFactor(float factor) { 
            this.yarnAttnFactor = factor; 
            return this; 
        }
        
        public CoreBuilder yarnBetaFast(float beta) { 
            this.yarnBetaFast = beta; 
            return this; 
        }
        
        public CoreBuilder yarnBetaSlow(float beta) { 
            this.yarnBetaSlow = beta; 
            return this; 
        }
        
        public CoreBuilder yarnOrigCtx(int ctx) { 
            this.yarnOrigCtx = ctx; 
            return this; 
        }
        
        public CoreBuilder seed(int seed) { 
            this.seed = seed; 
            return this; 
        }
        
        public CoreBuilder tensorSplit(String split) { 
            this.tensorSplit = split; 
            return this; 
        }
        
        public CoreBuilder mainGpu(int gpu) { 
            this.mainGpu = gpu; 
            return this; 
        }
        
        public CoreBuilder splitMode(int mode) { 
            this.splitMode = mode; 
            return this; 
        }
        
        public CoreBuilder sampling(SamplingConfig sampling) { 
            this.sampling = sampling; 
            return this; 
        }
        
        public LlamaConfig build() {
            return new CoreLlamaConfig(
                libraryPath, modelPath, gpuLayers, useMmap, useMlock, vocabOnly,
                contextSize, batchSize, uBatchSize, threads, threadsBatch,
                ropeFreqBase, ropeFreqScale, ropeScalingType, nSeqMax, poolingType,
                attentionType, embeddings, flashAttention, offloadKqv, defragThreshold,
                yarnExtFactor, yarnAttnFactor, yarnBetaFast, yarnBetaSlow, yarnOrigCtx,
                seed, tensorSplit, mainGpu, splitMode, sampling
            );
        }
    }
    
    // === CORE IMPLEMENTATION ===
    record CoreLlamaConfig(
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
        SamplingConfig sampling
    ) implements LlamaConfig {
        
        public CoreLlamaConfig {
            // Validate on construction
            validate();
        }
    }
    
    // === CORE DEFAULT IMPLEMENTATIONS ===
    class DefaultSamplingConfig implements SamplingConfig {
        @Override public float temperature() { return 1.0f; }
        @Override public int topK() { return 40; }
        @Override public float topP() { return 0.9f; }
        @Override public float minP() { return 0.05f; }
        @Override public float repeatPenalty() { return 1.1f; }
        @Override public int repeatLastN() { return 64; }
        @Override public float presencePenalty() { return 0.0f; }
        @Override public float frequencyPenalty() { return 0.0f; }
        @Override public float typicalP() { return 1.0f; }
        @Override public float tfsZ() { return 1.0f; }
        @Override public float mirostat() { return 0.0f; }
        @Override public float mirostatTau() { return 5.0f; }
        @Override public float mirostatEta() { return 0.1f; }
        @Override public String grammar() { return ""; }
        @Override public boolean grammarAcceptToken() { return false; }
        
        @Override
        public void validate() {
            if (temperature() < 0.0f || temperature() > 2.0f) {
                throw new IllegalArgumentException("Temperature must be between 0.0 and 2.0");
            }
            if (topK() < 0 && topK() != -1) {
                throw new IllegalArgumentException("TopK must be positive or -1");
            }
            if (topP() < 0.0f || topP() > 1.0f) {
                throw new IllegalArgumentException("TopP must be between 0.0 and 1.0");
            }
            if (minP() < 0.0f || minP() > 1.0f) {
                throw new IllegalArgumentException("MinP must be between 0.0 and 1.0");
            }
            if (repeatPenalty() < 0.0f) {
                throw new IllegalArgumentException("Repeat penalty must be non-negative");
            }
            if (typicalP() < 0.0f || typicalP() > 1.0f) {
                throw new IllegalArgumentException("TypicalP must be between 0.0 and 1.0");
            }
            if (tfsZ() < 0.0f || tfsZ() > 1.0f) {
                throw new IllegalArgumentException("TFS-Z must be between 0.0 and 1.0");
            }
            if (mirostat() < 0.0f || mirostat() > 2.0f) {
                throw new IllegalArgumentException("Mirostat must be 0, 1, or 2");
            }
        }
    }
}
