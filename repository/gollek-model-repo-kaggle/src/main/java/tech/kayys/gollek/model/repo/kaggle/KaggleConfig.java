package tech.kayys.gollek.model.repo.kaggle;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

import java.util.Optional;

/**
 * Kaggle model repository configuration.
 */
@ConfigMapping(prefix = "wayang.inference.repository.kaggle")
public interface KaggleConfig {

    /**
     * Kaggle API base URL
     */
    @WithDefault("https://www.kaggle.com")
    String baseUrl();

    /**
     * Kaggle API base URL for model downloads
     */
    @WithDefault("https://www.kaggle.com/api")
    String apiBaseUrl();

    /**
     * Kaggle username for authentication
     */
    Optional<String> username();

    /**
     * Kaggle API key/token
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

    /**
     * Auto-download remote Kaggle models when not found in local cache.
     */
    @WithDefault("true")
    boolean autoDownload();
}
