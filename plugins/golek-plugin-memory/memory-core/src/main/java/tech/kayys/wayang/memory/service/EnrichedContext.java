package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.memory.context.*;
import tech.kayys.wayang.memory.model.ConversationMemory;
import tech.kayys.wayang.memory.model.MemoryContext;
import java.time.Instant;
import java.util.List;

/**
 * Enriched context containing all memory layers
 */
public class EnrichedContext {
    private final MemoryContext baseContext;
    private final EpisodicMemory episodic;
    private final SemanticMemory semantic;
    private final ProceduralMemory procedural;
    private final WorkingMemory working;
    private final ContextWindow contextWindow;
    private final List<ConversationMemory> relevantMemories;
    private final Instant createdAt;

    public EnrichedContext(MemoryContext baseContext, EpisodicMemory episodic,
                          SemanticMemory semantic, ProceduralMemory procedural,
                          WorkingMemory working, ContextWindow contextWindow,
                          List<ConversationMemory> relevantMemories, Instant createdAt) {
        this.baseContext = baseContext;
        this.episodic = episodic;
        this.semantic = semantic;
        this.procedural = procedural;
        this.working = working;
        this.contextWindow = contextWindow;
        this.relevantMemories = relevantMemories;
        this.createdAt = createdAt;
    }

    public MemoryContext getBaseContext() { return baseContext; }
    public EpisodicMemory getEpisodic() { return episodic; }
    public SemanticMemory getSemantic() { return semantic; }
    public ProceduralMemory getProcedural() { return procedural; }
    public WorkingMemory getWorking() { return working; }
    public ContextWindow getContextWindow() { return contextWindow; }
    public List<ConversationMemory> getRelevantMemories() { return relevantMemories; }
    public Instant getCreatedAt() { return createdAt; }
}