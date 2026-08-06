package tech.kayys.gollek.sdk.core;

public record ChatParams(
    double temperature,
    int maxTokens,
    double topP,
    double repeatPenalty,
    String preferredProvider
) {
    public static ChatParams defaults() {
        return new ChatParams(0.7, 4096, 0.9, 1.1, null);
    }
    
    public static ChatParams of(double temperature, int maxTokens) {
        return new ChatParams(temperature, maxTokens, 0.9, 1.1, null);
    }

    public static ChatParams of(double temperature, int maxTokens, double topP, double repeatPenalty) {
        return new ChatParams(temperature, maxTokens, topP, repeatPenalty, null);
    }
}
