package tech.kayys.golek.memory;

/**
 * Types of memory in the cognitive architecture.
 * Based on human memory systems and modern agent frameworks.
 */
public enum MemoryType {

    /**
     * Episodic memory - specific events and experiences
     * Time-ordered, contextual, autobiographical
     */
    EPISODIC("episodic", true, true),

    /**
     * Semantic memory - facts and general knowledge
     * Conceptual, context-independent, generalized
     */
    SEMANTIC("semantic", false, true),

    /**
     * Procedural memory - skills and procedures
     * How-to knowledge, patterns, learned behaviors
     */
    PROCEDURAL("procedural", false, false),

    /**
     * Working memory - temporary active information
     * Short-lived, high-priority, conversation context
     */
    WORKING("working", true, false);

    private final String id;
    private final boolean temporal; // Has timestamp ordering
    private final boolean embedable; // Can be embedded for similarity search

    MemoryType(String id, boolean temporal, boolean embedable) {
        this.id = id;
        this.temporal = temporal;
        this.embedable = embedable;
    }

    public String getId() {
        return id;
    }

    public boolean isTemporal() {
        return temporal;
    }

    public boolean isEmbedable() {
        return embedable;
    }
}
