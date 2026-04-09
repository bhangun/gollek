package tech.kayys.gollek.sdk.hub;

import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.model.core.ModelRepository;
import tech.kayys.gollek.model.core.ModelRepositoryProvider;
import tech.kayys.gollek.model.core.RepositoryContext;
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.gollek.model.repo.hf.HuggingFaceConfig;
import tech.kayys.gollek.model.repo.hf.HuggingFaceRepository;
import tech.kayys.gollek.model.repo.local.LocalModelRepository;

import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Factory for resolving ModelRepository instances.
 * Provides support for both CDI-managed environments and standalone Java usage.
 */
public final class ModelHubFactory {

    private static final Map<String, ModelRepository> CACHE = new ConcurrentHashMap<>();

    private ModelHubFactory() {}

    /**
     * Resolves a ModelRepository based on the scheme (e.g., "hf", "local").
     *
     * @param scheme the repository scheme
     * @param config the hub configuration
     * @return the resolved repository
     */
    public static ModelRepository getRepository(String scheme, HubConfig config) {
        String cacheKey = scheme + ":" + config.cacheDir().toString();
        return CACHE.computeIfAbsent(cacheKey, k -> createRepository(scheme, config));
    }

    private static ModelRepository createRepository(String scheme, HubConfig config) {
        // 1. Try ServiceLoader for extensibility (Kaggle, etc.)
        ServiceLoader<ModelRepositoryProvider> loader = ServiceLoader.load(ModelRepositoryProvider.class);
        for (ModelRepositoryProvider provider : loader) {
            if (provider.scheme().equalsIgnoreCase(scheme)) {
                return provider.create(toContext(config));
            }
        }

        // 2. Fallback to hardcoded defaults if providers aren't registered via SPI yet
        return switch (scheme.toLowerCase()) {
            case "hf", "huggingface" -> createStandaloneHF(config);
            case "local" -> new LocalModelRepository();
            default -> throw new IllegalArgumentException("Unsupported repository scheme: " + scheme);
        };
    }

    private static ModelRepository createStandaloneHF(HubConfig config) {
        // Manual instantiation for standalone Java usage (minimize dependencies on CDI)
        HuggingFaceConfig hfConfig = new HuggingFaceConfig() {
            @Override public String baseUrl() { return "https://huggingface.co"; }
            @Override public Optional<String> token() { return Optional.ofNullable(config.token()); }
            @Override public int timeoutSeconds() { return config.timeoutSeconds(); }
            @Override public int maxRetries() { return 3; }
            @Override public boolean parallelDownload() { return true; }
            @Override public int parallelChunks() { return 4; }
            @Override public int chunkSizeMB() { return 10; }
            @Override public String userAgent() { return "gollek-sdk/" + config.revision(); }
            @Override public boolean autoDownload() { return !config.forceDownload(); }
        };

        HuggingFaceClient client = new HuggingFaceClient();
        // Reflectively setObjectMapper since it's @Inject and we are in standalone mode
        try {
            var field = HuggingFaceClient.class.getDeclaredField("objectMapper");
            field.setAccessible(true);
            field.set(client, new ObjectMapper().registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule()));
            
            var configField = HuggingFaceClient.class.getDeclaredField("config");
            configField.setAccessible(true);
            configField.set(client, hfConfig);
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize standalone HuggingFaceClient", e);
        }

        return new HuggingFaceRepository(config.cacheDir(), client, hfConfig);
    }

    private static RepositoryContext toContext(HubConfig config) {
        return new RepositoryContext(
            config.cacheDir(),
            Duration.ofSeconds(config.timeoutSeconds()),
            Map.of("token", config.token() != null ? config.token() : "")
        );
    }
}
