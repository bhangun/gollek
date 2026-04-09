package tech.kayys.gollek.sdk.internal;

import tech.kayys.gollek.sdk.GollekClient;
import tech.kayys.gollek.engine.inference.InferenceService;

import java.time.Duration;

/**
 * Default implementation of {@link GollekClient.Builder} that auto-detects whether
 * to create a local (in-process) or remote (HTTP) client.
 *
 * <p>Selection logic:
 * <ol>
 *   <li>If {@link #local()} was called, always use the local engine.</li>
 *   <li>If no endpoint is set but {@code InferenceService} is on the classpath, use local.</li>
 *   <li>If an endpoint is set, use the remote HTTP client.</li>
 * </ol>
 */
public class GollekClientBuilderImpl implements GollekClient.Builder {

    /** Remote server endpoint URL, or {@code null} for local mode. */
    String endpoint;
    /** Default model identifier passed to the client. */
    String model;
    /** API key used for authentication. Defaults to {@code "empty"}. */
    String apiKey = "empty";
    /** Request timeout in milliseconds. Defaults to 30 seconds. */
    long timeoutMillis = 30000;
    /** When {@code true}, forces local in-process mode regardless of classpath detection. */
    boolean forceLocal = false;

    @Override
    public GollekClient.Builder endpoint(String endpoint) {
        this.endpoint = endpoint;
        return this;
    }

    @Override
    public GollekClient.Builder local() {
        this.forceLocal = true;
        return this;
    }

    @Override
    public GollekClient.Builder model(String model) {
        this.model = model;
        return this;
    }

    @Override
    public GollekClient.Builder apiKey(String apiKey) {
        this.apiKey = apiKey;
        return this;
    }

    @Override
    public GollekClient.Builder timeoutMillis(long timeout) {
        this.timeoutMillis = timeout;
        return this;
    }

    /**
     * Builds the appropriate {@link GollekClient} based on the configured options.
     *
     * @return a local or remote {@link GollekClient} instance
     * @throws IllegalStateException if neither an endpoint nor a local engine is available
     */
    @Override
    public GollekClient build() {
        // Discovery logic
        if (forceLocal || (endpoint == null && isEnginePresent())) {
            return buildLocal();
        }

        if (endpoint != null) {
            return new RemoteGollekClient(this);
        }

        throw new IllegalStateException("No Gollek configuration provided. " +
                "Either provide an .endpoint('...') for remote mode, " +
                "or ensure the gollek-engine is on the classpath for .local() mode.");
    }

    /**
     * Checks whether the Gollek inference engine is available on the classpath.
     *
     * @return {@code true} if {@code InferenceService} can be loaded
     */
    private boolean isEnginePresent() {
        try {
            Class.forName("tech.kayys.gollek.engine.inference.InferenceService");
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    /**
     * Resolves the {@link InferenceService} from the CDI container and wraps it
     * in a {@link LocalGollekClient}.
     *
     * @return a {@link LocalGollekClient} backed by the in-process engine
     * @throws RuntimeException if the CDI container is unavailable or the service is not initialized
     */
    private GollekClient buildLocal() {
        try {
            // In a Quarkus app, we'd use Arc.container().instance(InferenceService.class).get()
            Class<?> arcClass = Class.forName("io.quarkus.arc.Arc");
            Object container = arcClass.getMethod("container").invoke(null);
            Object instance = container.getClass().getMethod("instance", Class.class, java.lang.annotation.Annotation[].class)
                    .invoke(container, InferenceService.class, new java.lang.annotation.Annotation[0]);
            
            InferenceService service = (InferenceService) instance.getClass().getMethod("get").invoke(instance);
            
            if (service == null) {
                throw new IllegalStateException("InferenceService found on classpath but not initialized in CDI container. " +
                        "Local mode requires a running Quarkus/Gollek environment.");
            }
            
            return new LocalGollekClient(service, model);
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize LocalGollekClient. " +
                    "Ensure the Gollek engine is properly started in this process.", e);
        }
    }
}
