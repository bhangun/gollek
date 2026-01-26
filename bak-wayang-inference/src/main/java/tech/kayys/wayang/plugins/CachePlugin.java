package tech.kayys.golek.plugins;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import tech.kayys.golek.engine.EnginePlugin;
import tech.kayys.golek.model.GenerationResult;
import tech.kayys.golek.plugin.PluginContext;

public class CachePlugin implements EnginePlugin {
    private PluginContext context;
    private final Map<String, CachedResponse> cache = new ConcurrentHashMap<>();
    private int maxCacheSize = 1000;
    private long ttlMs = 3600000; // 1 hour
    
    @Override
    public String getId() {
        return "cache";
    }
    
    @Override
    public String getName() {
        return "Response Cache";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "Caches generation responses";
    }
    
    @Override
    public void initialize(PluginContext context) {
        this.context = context;
        
        Integer configSize = context.getConfigValue("cache.size", Integer.class);
        if (configSize != null) {
            maxCacheSize = configSize;
        }
        
        Long configTtl = context.getConfigValue("cache.ttl", Long.class);
        if (configTtl != null) {
            ttlMs = configTtl;
        }
    }
    
    @Override
    public void start() {}
    
    @Override
    public void stop() {
        cache.clear();
    }
    
    @Override
    public String preprocessPrompt(String prompt) {
        // Check cache
        CachedResponse cached = cache.get(prompt);
        if (cached != null && !cached.isExpired()) {
            // Store in shared data for retrieval
            context.setSharedData("cache.hit." + prompt.hashCode(), cached.response);
            return "[CACHED]" + prompt;
        }
        return prompt;
    }
    
    @Override
    public void onGenerationComplete(GenerationResult result) {
        if (cache.size() >= maxCacheSize) {
            // Simple eviction: remove oldest
            cache.entrySet().stream()
                .min((e1, e2) -> Long.compare(e1.getValue().timestamp, e2.getValue().timestamp))
                .ifPresent(entry -> cache.remove(entry.getKey()));
        }
        
        // Cache the result (in real implementation, you'd need the original prompt)
        // This is simplified for demonstration
    }
    
    private record CachedResponse(String response, long timestamp, long ttl) {
        boolean isExpired() {
            return System.currentTimeMillis() - timestamp > ttl;
        }
    }
}
