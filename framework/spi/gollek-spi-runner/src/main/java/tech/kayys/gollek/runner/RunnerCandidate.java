package tech.kayys.gollek.runner;

import tech.kayys.alkhawarizm.spi.model.RunnerMetadata;

/**
 * Candidate runner for selection with score
 */
public record RunnerCandidate(
                String name,
                int score,
                RunnerMetadata metadata) {
}
