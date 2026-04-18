package tech.kayys.gollek.quantizer.gptq;

import java.util.Map;

/**
 * Model inspection result.
 */
public record ModelInspectionResult(
        GPTQConfig config,
        int layerCount,
        long totalMemoryBytes,
        java.util.List<String> layerNames,
        Map<String, String> metadata) {
    
    public double totalMemoryMB() {
        return totalMemoryBytes / (1024.0 * 1024.0);
    }

    @Override
    public String toString() {
        return "ModelInspectionResult{config=%s, layers=%d, memory=%.2f MB}"
                .formatted(config, layerCount, totalMemoryMB());
    }
}
