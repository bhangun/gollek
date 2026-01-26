package tech.kayys.golek.model.repository.hf;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

import java.util.Optional;

/**
 * HuggingFace configuration
 */
@ConfigMapping(prefix = "wayang.inference.repository.huggingface")
public interface HuggingFaceConfig {

    /**
     * HuggingFace API base URL
     */
    @WithDefault("https://huggingface.co")
    String baseUrl();

    /**
     * HuggingFace API token for private models
     */
    Optional<String> token();

    /**
     * Timeout for API calls in seconds
     */
    @WithDefault("30")
    int timeoutSeconds();

    /**
     * Max retries for failed downloads
     */
    @WithDefault("3")
    int maxRetries();

    /**
     * Enable parallel downloads
     */
    @WithDefault("true")
    boolean parallelDownload();

    /**
     * Number of concurrent download chunks
     */
    @WithDefault("4")
    int parallelChunks();

    /**
     * Chunk size for downloads in MB
     */
    @WithDefault("10")
    int chunkSizeMB();

    /**
     * User agent for requests
     */
    @WithDefault("wayang-inference/1.0")
    String userAgent();
}