package tech.kayys.gollek.plugin.optimization;

import tech.kayys.gollek.plugin.runner.RunnerSession;

/**
 * KV Cache optimization plugin that monitors VRAM usage and offloads 
 * inactive sequences to CPU/NVMe to maximize GPU memory efficiency.
 */
public class KVCacheOptimizationPlugin implements OptimizationPlugin {

    private static final double VRAM_OFFLOAD_THRESHOLD = 0.90; // Offload if >90%

    @Override
    public String id() {
        return "kv-cache-offload";
    }

    @Override
    public String name() {
        return "KV Cache Offloader";
    }

    @Override
    public String description() {
        return "Offloads KV cache to CPU/NVMe when VRAM is under high pressure.";
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public int priority() {
        return 100; // High priority to ensure memory is freed
    }

    @Override
    public boolean apply(ExecutionContext context) {
        // Since ExecutionContext does not expose RunnerSession directly in this mock implementation,
        // we retrieve it from context attributes.
        return context.getAttribute("runnerSession", RunnerSession.class)
                .map(session -> {
                    if (session.getKVCacheState() != null) {
                        double utilization = session.getKVCacheState().getVramUtilization();
                        if (utilization >= VRAM_OFFLOAD_THRESHOLD) {
                            session.offloadCache();
                            return true;
                        }
                    }
                    return false;
                })
                .orElse(false);
    }
}
