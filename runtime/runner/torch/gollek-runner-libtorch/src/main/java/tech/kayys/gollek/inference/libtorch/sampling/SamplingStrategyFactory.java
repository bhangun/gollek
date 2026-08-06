package tech.kayys.gollek.inference.libtorch.sampling;

/**
 * Factory for creating sampling strategies from configuration parameters.
 * <p>
 * Usage:
 * 
 * <pre>{@code
 * SamplingStrategy strategy = SamplingStrategyFactory.create("top_p", 0.9, 0.8, 50);
 * long nextToken = strategy.sample(logits);
 * }</pre>
 */
public final class SamplingStrategyFactory {

    private SamplingStrategyFactory() {
    }

    /**
     * Create a sampling strategy from configuration.
     *
     * @param type        strategy type: "greedy", "temperature", "top_k", "top_p"
     * @param temperature temperature parameter (ignored for greedy)
     * @param topP        nucleus probability threshold (only for top_p)
     * @param topK        number of top tokens (only for top_k)
     * @return configured sampling strategy
     */
    public static SamplingStrategy create(String type, double temperature, double topP, int topK) {
        return switch (type.toLowerCase().trim()) {
            case "greedy", "argmax" -> new GreedySampler();
            case "temperature" -> new TemperatureSampler(temperature);
            case "top_k", "topk" -> new TopKSampler(topK, temperature);
            case "top_p", "topp", "nucleus" -> new TopPSampler(topP, temperature);
            default -> throw new IllegalArgumentException(
                    "Unknown sampling strategy: '" + type
                            + "'. Supported: greedy, temperature, top_k, top_p");
        };
    }

    /**
     * Create a default sampling strategy (greedy).
     */
    public static SamplingStrategy defaultStrategy() {
        return new GreedySampler();
    }
}
