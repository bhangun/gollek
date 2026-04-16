package tech.kayys.gollek.inference.gguf;

import java.util.LinkedHashSet;
import java.util.Set;

final class GGUFOptimizationDetector {

    private GGUFOptimizationDetector() {
    }

    static Set<String> detectFeatures() {
        Set<String> features = new LinkedHashSet<>();
        if (isPresent("tech.kayys.gollek.cache.PromptCacheLookupPlugin")) {
            features.add("prompt_cache");
        }
        if (isPresent("tech.kayys.gollek.kvcache.PagedKVCacheManager")) {
            features.add("paged_kv_cache");
        }
        if (isPresent("tech.kayys.gollek.kernel.paged.PagedAttentionBinding")) {
            features.add("paged_attention");
        }
        if (isPresent("tech.kayys.gollek.prefilldecode.PrefillDecodeDisaggService")) {
            features.add("prefill_decode_disagg");
        }
        if (isPresent("tech.kayys.gollek.hybridattn.HybridAttentionGdnRunner")) {
            features.add("hybrid_attention");
        }
        if (isPresent("tech.kayys.gollek.flashattn.FlashAttention4Runner")) {
            features.add("flash_attention4");
        }
        return features;
    }

    static boolean hasOptimizationModules() {
        return !detectFeatures().isEmpty();
    }

    private static boolean isPresent(String className) {
        try {
            Class.forName(className, false, GGUFOptimizationDetector.class.getClassLoader());
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }
}
