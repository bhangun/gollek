package tech.kayys.golek.provider.onnx;

import java.util.Map;
import java.util.Optional;

/**
 * Configuration for ONNX Runtime adapter
 */
public class OnnxConfig {
    
    private final String executionProvider;
    private final Integer interOpThreads;
    private final Integer intraOpThreads;
    private final boolean enableMemoryPatternOptimization;
    private final boolean enableMemArena;
    private final String optimizationLevel;
    private final boolean enableProfiling;
    private final String profileOutputPath;

    public OnnxConfig(String executionProvider, Integer interOpThreads, Integer intraOpThreads,
                      boolean enableMemoryPatternOptimization, boolean enableMemArena,
                      String optimizationLevel, boolean enableProfiling, String profileOutputPath) {
        this.executionProvider = executionProvider;
        this.interOpThreads = interOpThreads;
        this.intraOpThreads = intraOpThreads;
        this.enableMemoryPatternOptimization = enableMemoryPatternOptimization;
        this.enableMemArena = enableMemArena;
        this.optimizationLevel = optimizationLevel;
        this.enableProfiling = enableProfiling;
        this.profileOutputPath = profileOutputPath;
    }

    public String getExecutionProvider() {
        return executionProvider;
    }

    public Optional<Integer> getInterOpThreads() {
        return Optional.ofNullable(interOpThreads);
    }

    public Optional<Integer> getIntraOpThreads() {
        return Optional.ofNullable(intraOpThreads);
    }

    public boolean isEnableMemoryPatternOptimization() {
        return enableMemoryPatternOptimization;
    }

    public boolean isEnableMemArena() {
        return enableMemArena;
    }

    public String getOptimizationLevel() {
        return optimizationLevel;
    }

    public boolean isEnableProfiling() {
        return enableProfiling;
    }

    public Optional<String> getProfileOutputPath() {
        return Optional.ofNullable(profileOutputPath);
    }

    public static OnnxConfigBuilder builder() {
        return new OnnxConfigBuilder();
    }

    public static class OnnxConfigBuilder {
        private String executionProvider;
        private Integer interOpThreads;
        private Integer intraOpThreads;
        private boolean enableMemoryPatternOptimization = true;
        private boolean enableMemArena = true;
        private String optimizationLevel = "ALL_OPT";
        private boolean enableProfiling = false;
        private String profileOutputPath;

        public OnnxConfigBuilder executionProvider(String executionProvider) {
            this.executionProvider = executionProvider;
            return this;
        }

        public OnnxConfigBuilder interOpThreads(Integer interOpThreads) {
            this.interOpThreads = interOpThreads;
            return this;
        }

        public OnnxConfigBuilder intraOpThreads(Integer intraOpThreads) {
            this.intraOpThreads = intraOpThreads;
            return this;
        }

        public OnnxConfigBuilder enableMemoryPatternOptimization(boolean enableMemoryPatternOptimization) {
            this.enableMemoryPatternOptimization = enableMemoryPatternOptimization;
            return this;
        }

        public OnnxConfigBuilder enableMemArena(boolean enableMemArena) {
            this.enableMemArena = enableMemArena;
            return this;
        }

        public OnnxConfigBuilder optimizationLevel(String optimizationLevel) {
            this.optimizationLevel = optimizationLevel;
            return this;
        }

        public OnnxConfigBuilder enableProfiling(boolean enableProfiling) {
            this.enableProfiling = enableProfiling;
            return this;
        }

        public OnnxConfigBuilder profileOutputPath(String profileOutputPath) {
            this.profileOutputPath = profileOutputPath;
            return this;
        }

        public OnnxConfig build() {
            return new OnnxConfig(executionProvider, interOpThreads, intraOpThreads,
                    enableMemoryPatternOptimization, enableMemArena, optimizationLevel,
                    enableProfiling, profileOutputPath);
        }
    }
}