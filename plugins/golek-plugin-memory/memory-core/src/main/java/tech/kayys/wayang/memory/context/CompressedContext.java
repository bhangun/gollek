package tech.kayys.wayang.memory.context;

public class CompressedContext {
    private final String compressedContent;
    private final int originalMemoryCount;
    private final int compressedUnitCount;
    private final double compressionRatio;
    private final CompressionStrategy strategy;
    private final double informationRetention;

    public CompressedContext(String compressedContent, int originalMemoryCount,
                           int compressedUnitCount, double compressionRatio,
                           CompressionStrategy strategy, double informationRetention) {
        this.compressedContent = compressedContent;
        this.originalMemoryCount = originalMemoryCount;
        this.compressedUnitCount = compressedUnitCount;
        this.compressionRatio = compressionRatio;
        this.strategy = strategy;
        this.informationRetention = informationRetention;
    }

    public String getCompressedContent() { return compressedContent; }
    public int getOriginalMemoryCount() { return originalMemoryCount; }
    public int getCompressedUnitCount() { return compressedUnitCount; }
    public double getCompressionRatio() { return compressionRatio; }
    public CompressionStrategy getStrategy() { return strategy; }
    public double getInformationRetention() { return informationRetention; }
}