package tech.kayys.golek.api.inference;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Comprehensive ordered phases of inference execution.
 * Each phase represents a distinct stage with specific responsibilities.
 * Plugins are bound to specific phases.
 */
public enum InferencePhase {

    /**
     * Phase 1: Pre-validation checks
     * - Request structure validation
     * - Basic sanity checks
     * - Early rejection of malformed requests
     */
    PRE_VALIDATE(1, "Pre-Validation"),

    /**
     * Phase 2: Deep validation
     * - Schema validation (JSON Schema)
     * - Content safety checks
     * - Input format verification
     * - Model compatibility checks
     */
    VALIDATE(2, "Validation"),

    /**
     * Phase 3: Authorization & access control
     * - Tenant verification
     * - Model access permissions
     * - Feature flag checks
     * - Quota verification
     */
    AUTHORIZE(3, "Authorization"),

    /**
     * Phase 4: Intelligent routing & provider selection
     * - Model-to-provider mapping
     * - Multi-factor scoring
     * - Load balancing
     * - Availability checks
     */
    ROUTE(4, "Routing"),

    /**
     * Phase 5: Request transformation & enrichment
     * - Prompt templating
     * - Context injection
     * - Parameter normalization
     * - Request mutation
     */
    PRE_PROCESSING(5, "Pre-Processing"),

    /**
     * Phase 6: Actual provider dispatch
     * - Provider invocation
     * - Streaming/batch execution
     * - Circuit breaker protection
     * - Fallback handling
     */
    PROVIDER_DISPATCH(6, "Provider Dispatch"),

    /**
     * Phase 7: Response post-processing
     * - Output validation
     * - Format normalization
     * - Metadata enrichment
     * - Content moderation
     */
    POST_PROCESSING(7, "Post-Processing"),

    /**
     * Phase 8: Audit logging
     * - Event recording
     * - Provenance tracking
     * - Compliance logging
     * - Immutable audit trail
     */
    AUDIT(8, "Audit"),

    /**
     * Phase 9: Observability & metrics
     * - Metrics emission
     * - Trace completion
     * - Performance tracking
     * - Cost accounting
     */
    OBSERVABILITY(9, "Observability"),

    /**
     * Phase 10: Resource cleanup
     * - Cache invalidation
     * - Connection release
     * - Quota release
     * - Temporary resource cleanup
     */
    CLEANUP(10, "Cleanup");

    private final int order;
    private final String displayName;

    InferencePhase(int order, String displayName) {
        this.order = order;
        this.displayName = displayName;
    }

    public int getOrder() {
        return order;
    }

    public String getDisplayName() {
        return displayName;
    }

    /**
     * Get phases in execution order
     */
    public static List<InferencePhase> ordered() {
        return Arrays.stream(values())
                .sorted(Comparator.comparing(InferencePhase::getOrder))
                .toList();
    }

    /**
     * Check if this phase is critical (execution cannot proceed if it fails)
     */
    public boolean isCritical() {
        return this == PRE_VALIDATE ||
                this == VALIDATE ||
                this == AUTHORIZE ||
                this == PROVIDER_DISPATCH;
    }

    /**
     * Check if this phase can be retried
     */
    public boolean isRetryable() {
        return this == ROUTE ||
                this == PROVIDER_DISPATCH;
    }

    /**
     * Check if this phase is idempotent
     */
    public boolean isIdempotent() {
        return this != PROVIDER_DISPATCH;
    }

    /**
     * Check if this phase should run even on error
     */
    public boolean runsOnError() {
        return this == AUDIT ||
                this == OBSERVABILITY ||
                this == CLEANUP;
    }
}