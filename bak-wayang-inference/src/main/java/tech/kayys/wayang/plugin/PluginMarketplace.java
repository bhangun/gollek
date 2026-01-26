package tech.kayys.golek.plugin;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class PluginMarketplace {
    
    private static final Map<String, MarketplacePlugin> OFFICIAL_PLUGINS = new ConcurrentHashMap<>();
    
    static {
        // Register official plugins
        registerPlugin(new MarketplacePlugin(
            "logging",
            "Logging Plugin",
            "1.0.0",
            "Logs all generation requests and responses",
            "LlamaJava Team",
            "https://github.com/llamajava/plugins/logging",
            "com.llamajava.plugins:logging-plugin:1.0.0",
            List.of("monitoring", "debugging"),
            true,
            4.8
        ));
        
        registerPlugin(new MarketplacePlugin(
            "rag",
            "RAG Plugin",
            "1.0.0",
            "Retrieval Augmented Generation with vector search",
            "LlamaJava Team",
            "https://github.com/llamajava/plugins/rag",
            "com.llamajava.plugins:rag-plugin:1.0.0",
            List.of("retrieval", "search", "embeddings"),
            true,
            4.9
        ));
        
        registerPlugin(new MarketplacePlugin(
            "auth",
            "Authentication Plugin",
            "1.0.0",
            "API key authentication and authorization",
            "LlamaJava Team",
            "https://github.com/llamajava/plugins/auth",
            "com.llamajava.plugins:auth-plugin:1.0.0",
            List.of("security", "authentication"),
            true,
            4.7
        ));
        
        registerPlugin(new MarketplacePlugin(
            "cache",
            "Cache Plugin",
            "1.0.0",
            "Response caching for faster repeated queries",
            "LlamaJava Team",
            "https://github.com/llamajava/plugins/cache",
            "com.llamajava.plugins:cache-plugin:1.0.0",
            List.of("performance", "caching"),
            true,
            4.6
        ));
        
        registerPlugin(new MarketplacePlugin(
            "analytics",
            "Analytics Plugin",
            "1.0.0",
            "Usage analytics and insights",
            "LlamaJava Team",
            "https://github.com/llamajava/plugins/analytics",
            "com.llamajava.plugins:analytics-plugin:1.0.0",
            List.of("analytics", "monitoring"),
            true,
            4.5
        ));
        
        registerPlugin(new MarketplacePlugin(
            "content-filter",
            "Content Filter",
            "1.0.0",
            "Filter inappropriate content",
            "LlamaJava Team",
            "https://github.com/llamajava/plugins/content-filter",
            "com.llamajava.plugins:content-filter:1.0.0",
            List.of("safety", "moderation"),
            true,
            4.8
        ));
        
        registerPlugin(new MarketplacePlugin(
            "rate-limit",
            "Rate Limiting",
            "1.0.0",
            "Protect your API with rate limiting",
            "LlamaJava Team",
            "https://github.com/llamajava/plugins/rate-limit",
            "com.llamajava.plugins:rate-limit:1.0.0",
            List.of("security", "protection"),
            true,
            4.7
        ));
        
        registerPlugin(new MarketplacePlugin(
            "monitoring",
            "Monitoring Plugin",
            "1.0.0",
            "Real-time API monitoring and metrics",
            "LlamaJava Team",
            "https://github.com/llamajava/plugins/monitoring",
            "com.llamajava.plugins:monitoring:1.0.0",
            List.of("monitoring", "metrics", "observability"),
            true,
            4.9
        ));
    }
    
    private static void registerPlugin(MarketplacePlugin plugin) {
        OFFICIAL_PLUGINS.put(plugin.id(), plugin);
    }
    
    public static List<MarketplacePlugin> listPlugins() {
        return List.copyOf(OFFICIAL_PLUGINS.values());
    }
    
    public static List<MarketplacePlugin> searchPlugins(String query) {
        return OFFICIAL_PLUGINS.values().stream()
            .filter(p -> p.name().toLowerCase().contains(query.toLowerCase()) ||
                        p.description().toLowerCase().contains(query.toLowerCase()) ||
                        p.tags().stream().anyMatch(t -> t.toLowerCase().contains(query.toLowerCase())))
            .toList();
    }
    
    public static MarketplacePlugin getPlugin(String pluginId) {
        return OFFICIAL_PLUGINS.get(pluginId);
    }
    
    public record MarketplacePlugin(
        String id,
        String name,
        String version,
        String description,
        String author,
        String repository,
        String mavenCoordinates,
        List<String> tags,
        boolean verified,
        double rating
    ) {}
}
