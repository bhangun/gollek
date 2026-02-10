package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;

import java.util.ArrayList;
import java.util.List;

public class MemoryCluster {
    private final List<ConversationMemory> memories = new ArrayList<>();

    public void addMemory(ConversationMemory memory) {
        memories.add(memory);
    }

    public List<ConversationMemory> getMemories() {
        return memories;
    }
}