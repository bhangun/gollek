package tech.kayys.gamelan.executor.memory;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for creating embedding service based on configuration
 */
@ApplicationScoped
public class EmbeddingServiceFactory {

    private static final Logger LOG = LoggerFactory.getLogger(EmbeddingServiceFactory.class);

    @ConfigProperty(name = "gamelan.embedding.provider", defaultValue = "local")
    String provider;

    @Inject
    OpenAIEmbeddingService openAIService;

    @Inject
    LocalTFIDFEmbeddingService localService;

    /**
     * Get the configured embedding service
     */
    public EmbeddingService getEmbeddingService() {
        LOG.info("Using embedding provider: {}", provider);

        return switch (provider.toLowerCase()) {
            case "openai" -> openAIService;
            case "local", "local-tfidf" -> localService;
            default -> {
                LOG.warn("Unknown provider: {}, falling back to local", provider);
                yield localService;
            }
        };
    }
}