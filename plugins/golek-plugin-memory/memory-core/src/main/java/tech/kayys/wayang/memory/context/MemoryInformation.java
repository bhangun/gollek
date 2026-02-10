package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;

public class MemoryInformation {
    private final ConversationMemory memory;
    private final double informationContent;
    private final double redundancy;

    public MemoryInformation(ConversationMemory memory, double informationContent, double redundancy) {
        this.memory = memory;
        this.informationContent = informationContent;
        this.redundancy = redundancy;
    }

    public ConversationMemory getMemory() { return memory; }
    public double getInformationContent() { return informationContent; }
    public double getRedundancy() { return redundancy; }
    public double getNetInformation() { return informationContent - redundancy; }
}