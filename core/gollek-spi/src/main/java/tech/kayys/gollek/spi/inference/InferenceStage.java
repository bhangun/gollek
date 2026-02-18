package tech.kayys.gollek.spi.inference;

/**
 * Represents the compute stage of an inference request within the engine.
 * <p>
 * In a disaggregated Prefill/Decode architecture, a single user request
 * is split into two distinct compute phases:
 * <ul>
 * <li><b>PREFILL</b> — Compute-bound. Processes the entire input prompt in one
 * forward pass to generate the initial KV-Cache. Runs on high-FLOP hardware
 * (e.g., H100). This is where the prompt "attention" is calculated.</li>
 * <li><b>DECODE</b> — Memory-bandwidth bound. Generates tokens one at a time
 * autoregressively. Runs on high-bandwidth hardware (e.g., A100/L40S).
 * Uses the KV-Cache from the prefill phase.</li>
 * <li><b>COMBINED</b> — Both phases run on the same node (single-machine mode).
 * This is the default for local/standalone deployments.</li>
 * </ul>
 * <p>
 * This is separate from {@link InferencePhase} which represents the plugin
 * pipeline stages. {@code InferenceStage} represents the hardware-level
 * compute phase for routing decisions.
 *
 * @see InferencePhase
 */
public enum InferenceStage {

    /**
     * Prefill phase: process the entire input prompt.
     * Compute-bound, optimized for high TFLOPS.
     */
    PREFILL("Prefill", true, false),

    /**
     * Decode phase: generate tokens one at a time.
     * Memory-bandwidth bound, optimized for high memory throughput.
     */
    DECODE("Decode", false, true),

    /**
     * Combined mode: both prefill and decode on the same node.
     * Default for single-machine / standalone deployments.
     */
    COMBINED("Combined", true, true);

    private final String displayName;
    private final boolean handlesPrefill;
    private final boolean handlesDecode;

    InferenceStage(String displayName, boolean handlesPrefill, boolean handlesDecode) {
        this.displayName = displayName;
        this.handlesPrefill = handlesPrefill;
        this.handlesDecode = handlesDecode;
    }

    public String getDisplayName() {
        return displayName;
    }

    /**
     * Whether this stage handles the prefill (prompt processing) phase.
     */
    public boolean handlesPrefill() {
        return handlesPrefill;
    }

    /**
     * Whether this stage handles the decode (token generation) phase.
     */
    public boolean handlesDecode() {
        return handlesDecode;
    }

    /**
     * Whether this stage is disaggregated (not combined).
     */
    public boolean isDisaggregated() {
        return this != COMBINED;
    }

    /**
     * Determine the appropriate stage based on prompt size and configuration.
     *
     * @param promptTokens         number of tokens in the prompt
     * @param disaggregated        whether the cluster is in disaggregated mode
     * @param smallPromptThreshold token count below which disaggregation is skipped
     * @return the appropriate inference stage
     */
    public static InferenceStage forRequest(int promptTokens, boolean disaggregated,
            int smallPromptThreshold) {
        if (!disaggregated) {
            return COMBINED;
        }
        // Small prompts don't benefit from disaggregation
        if (promptTokens < smallPromptThreshold) {
            return COMBINED;
        }
        // Large prompts start with PREFILL, then transition to DECODE
        return PREFILL;
    }
}
