package tech.kayys.golek.engine;

public record SamplingParams(
        float temperature,
        int topK,
        float topP,
        float minP,
        float repeatPenalty,
        int repeatLastN,
        float presencePenalty,
        float frequencyPenalty,
        int seed) {
    public static SamplingParams fromConfig(LlamaConfig.SamplingConfig config, int seed) {
        return new SamplingParams(
                config.temperature(),
                config.topK(),
                config.topP(),
                config.minP(),
                config.repeatPenalty(),
                config.repeatLastN(),
                config.presencePenalty(),
                config.frequencyPenalty(),
                seed);
    }
}