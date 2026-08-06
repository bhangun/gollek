package tech.kayys.gollek.inference.libtorch;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.config.LibTorchProviderConfig;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages caching and loading of HuggingFace tokenizers for the Torch runner.
 */
@ApplicationScoped
public class LibTorchTokenizerManager {

    private static final Logger log = Logger.getLogger(LibTorchTokenizerManager.class);

    @Inject
    LibTorchProviderConfig config;

    private final Map<String, Tokenizer> tokenizerCache = new ConcurrentHashMap<>();

    /**
     * Get or load the tokenizer for the given model ID.
     */
    public Tokenizer getTokenizer(String modelId) {
        return tokenizerCache.computeIfAbsent(modelId, this::loadTokenizer);
    }

    private Tokenizer loadTokenizer(String modelId) {
        Path modelBasePath = Path.of(config.model().basePath());
        Path modelPath = modelBasePath.resolve(modelId);

        if (!Files.exists(modelPath)) {
            String cleanId = modelId.replaceAll("\\.(pt|pts|pth)$", "");
            modelPath = modelBasePath.resolve(cleanId);
        }

        try {
            if (Files.exists(modelPath)) {
                return TokenizerFactory.load(modelPath, null);
            } else {
                log.warnf("No model directory found for '%s', checking base path for tokenizer files", modelId);
                return TokenizerFactory.load(modelBasePath, null);
            }
        } catch (Exception e) {
            log.warnf(e, "Failed to load tokenizer for model '%s', tokenizer loading failed", modelId);
            throw new RuntimeException("Tokenizer loading failed", e);
        }
    }
}
