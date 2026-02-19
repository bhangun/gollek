package tech.kayys.gollek.spi.inference;

/**
 * Defines the batching strategy used by the scheduler.
 * <p>
 * Supported strategies align with production inference optimization patterns:
 * <ul>
 * <li>{@link #STATIC} - Waits for a fixed batch size before processing. simple
 * but causes head-of-line blocking.</li>
 * <li>{@link #DYNAMIC} - Uses a time window and max batch size. Better latency
 * balance.</li>
 * <li>{@link #CONTINUOUS} - Iteration-level scheduling. Finished sequences are
 * replaced immediately.
 * Required for maximum GPU utilization in production LLM serving.</li>
 * </ul>
 */
public enum BatchStrategy {
    STATIC("Static"),
    DYNAMIC("Dynamic"),
    CONTINUOUS("Continuous");

    private final String displayName;

    BatchStrategy(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}
