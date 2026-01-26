package tech.kayys.golek.model.core;

import tech.kayys.golek.api.model.RunnerMetadata;

/**
 * Candidate runner for selection with score
 */
public record RunnerCandidate(
                String name,
                int score,
                RunnerMetadata metadata) {
}
