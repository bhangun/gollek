package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;
import tech.kayys.wayang.memory.model.MemoryContext;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Implements memory reinforcement and adaptive learning
 * Based on spaced repetition and forgetting curves
 */
@ApplicationScoped
public class MemoryReinforcementService {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryReinforcementService.class);

    /**
     * Calculate memory strength based on recency and access patterns
     */
    public Uni<Map<String, MemoryStrength>> calculateMemoryStrengths(
            String sessionId,
            List<ConversationMemory> memories,
            Map<String, List<Instant>> accessHistory) {

        return Uni.createFrom().item(() -> {
            Map<String, MemoryStrength> strengths = new HashMap<>();

            for (ConversationMemory memory : memories) {
                List<Instant> accesses = accessHistory.getOrDefault(
                    memory.getId(),
                    List.of(memory.getTimestamp())
                );

                double strength = calculateStrength(
                    memory.getTimestamp(),
                    accesses,
                    memory.getRelevanceScore()
                );

                double decayRate = calculateDecayRate(accesses);
                Instant nextReview = calculateNextReviewTime(strength, decayRate);

                strengths.put(memory.getId(), new MemoryStrength(
                    memory.getId(),
                    strength,
                    decayRate,
                    accesses.size(),
                    nextReview
                ));
            }

            return strengths;
        });
    }

    /**
     * Identify memories that need reinforcement
     */
    public Uni<List<ConversationMemory>> identifyMemoriesForReinforcement(
            List<ConversationMemory> memories,
            Map<String, MemoryStrength> strengths) {

        return Uni.createFrom().item(() -> {
            Instant now = Instant.now();

            return memories.stream()
                .filter(m -> {
                    MemoryStrength strength = strengths.get(m.getId());
                    return strength != null &&
                           strength.getNextReviewTime().isBefore(now) &&
                           strength.getStrength() < 0.7;
                })
                .sorted(Comparator.comparing(m -> strengths.get(m.getId()).getStrength()))
                .limit(10)
                .collect(Collectors.toList());
        });
    }

    /**
     * Apply reinforcement to memories through retrieval practice
     */
    public Uni<Void> reinforceMemory(
            String memoryId,
            Map<String, List<Instant>> accessHistory) {

        return Uni.createFrom().item(() -> {
            accessHistory.computeIfAbsent(memoryId, k -> new ArrayList<>())
                        .add(Instant.now());

            LOG.debug("Reinforced memory: {}, total accesses: {}",
                     memoryId, accessHistory.get(memoryId).size());
            return null;
        }).replaceWithVoid();
    }

    /**
     * Calculate memory strength using spaced repetition algorithm
     */
    private double calculateStrength(
            Instant creationTime,
            List<Instant> accessHistory,
            Double baseRelevance) {

        if (accessHistory.isEmpty()) {
            return baseRelevance != null ? baseRelevance : 0.5;
        }

        double strength = baseRelevance != null ? baseRelevance : 0.5;
        Instant lastAccess = accessHistory.get(accessHistory.size() - 1);

        // Apply forgetting curve
        long daysSinceLastAccess = Duration.between(lastAccess, Instant.now()).toDays();
        double forgetting = Math.exp(-daysSinceLastAccess / 7.0); // 7-day half-life

        // Apply learning from repetitions
        double learning = 1.0 - Math.exp(-accessHistory.size() / 5.0);

        strength = (strength * forgetting + learning) / 2.0;

        return Math.max(0.1, Math.min(1.0, strength));
    }

    /**
     * Calculate decay rate based on access pattern
     */
    private double calculateDecayRate(List<Instant> accessHistory) {
        if (accessHistory.size() < 2) {
            return 0.5; // Default decay rate
        }

        // Calculate average interval between accesses
        long totalInterval = 0;
        for (int i = 1; i < accessHistory.size(); i++) {
            totalInterval += Duration.between(
                accessHistory.get(i - 1),
                accessHistory.get(i)
            ).toDays();
        }

        double avgInterval = (double) totalInterval / (accessHistory.size() - 1);

        // Faster decay for infrequently accessed memories
        return 1.0 / (1.0 + avgInterval);
    }

    /**
     * Calculate next optimal review time using SM-2 algorithm
     */
    private Instant calculateNextReviewTime(double strength, double decayRate) {
        // More spaced intervals for stronger memories
        double daysUntilReview = Math.pow(2, strength * 5) * (1 - decayRate);

        return Instant.now().plus(Duration.ofDays((long) daysUntilReview));
    }

    /**
     * Adaptive forgetting: Remove or compress low-value memories
     */
    public Uni<List<ConversationMemory>> applyAdaptiveForgetting(
            List<ConversationMemory> memories,
            Map<String, MemoryStrength> strengths,
            int targetCount) {

        return Uni.createFrom().item(() -> {
            return memories.stream()
                .sorted((m1, m2) -> {
                    MemoryStrength s1 = strengths.get(m1.getId());
                    MemoryStrength s2 = strengths.get(m2.getId());

                    double score1 = s1 != null ? s1.getStrength() : 0.0;
                    double score2 = s2 != null ? s2.getStrength() : 0.0;

                    return Double.compare(score2, score1);
                })
                .limit(targetCount)
                .collect(Collectors.toList());
        });
    }
}