package tech.kayys.gollek.safetensor.spi;

/**
 * SPI for pluggable SafeTensor feature modules (e.g. audio, vision, RAG).
 *
 * <p>Features are discovered via CDI and automatically registered with the
 * {@code SafetensorProvider} during startup. Each feature can be conditionally
 * enabled or disabled through configuration:
 * <pre>
 * gollek.safetensor.feature.[id].enabled=false
 * </pre>
 *
 * <p>Implement this interface and annotate the class with {@code @ApplicationScoped}
 * to register a new feature.
 */
public interface SafetensorFeature {

    /**
     * Unique identifier for this feature (e.g. {@code "audio"}, {@code "vision"}).
     *
     * <p>Used as the configuration key suffix:
     * {@code gollek.safetensor.feature.[id].enabled}.
     *
     * @return feature identifier string
     */
    String id();

    /**
     * Whether this feature should be enabled when no explicit configuration is present.
     *
     * @return {@code true} by default
     */
    default boolean enabledByDefault() {
        return true;
    }

    /**
     * Initialization priority — lower values run first.
     *
     * <p>Use this to control ordering when one feature depends on another.
     *
     * @return priority value (default: {@code 100})
     */
    default int priority() {
        return 100;
    }

    /**
     * Initializes this feature's resources and registers any required services.
     *
     * <p>Called by {@code SafetensorProvider} during its own initialization phase,
     * after CDI injection is complete.
     */
    void initialize();

    /**
     * Releases any resources held by this feature.
     *
     * <p>Called during graceful shutdown. The default implementation is a no-op.
     */
    default void shutdown() {
        // Optional cleanup
    }
}
