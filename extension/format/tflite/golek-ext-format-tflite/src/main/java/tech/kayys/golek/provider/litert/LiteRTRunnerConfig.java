package tech.kayys.golek.provider.litert;

public record LiteRTRunnerConfig(
        int numThreads,
        boolean useGpu,
        boolean useNpu,
        String gpuBackend,
        String npuType) {

    public static LiteRTRunnerConfig defaults() {
        return new LiteRTRunnerConfig(4, false, false, "auto", "auto");
    }
}
