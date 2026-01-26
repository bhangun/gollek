package tech.kayys.golek.engine.model;

public record RunnerCandidate(
        String runnerName,
        String providerType,
        int score,
        boolean available
) {
}