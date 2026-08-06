package tech.kayys.gollek.sdk.core;

import tech.kayys.gollek.sdk.config.SdkConfig;
import tech.kayys.gollek.sdk.exception.SdkException;

/**
 * Service Provider Interface for Gollek SDK implementations.
 * 
 * <p>
 * Each implementation module (local, remote) registers a provider via
 * {@code META-INF/services/tech.kayys.gollek.sdk.core.GollekSdkProvider}.
 * The facade module discovers providers at runtime using
 * {@link java.util.ServiceLoader}.
 *
 * <p>
 * Implementations can also be discovered via CDI when running inside a
 * CDI-enabled container (e.g., Quarkus).
 */
public interface GollekSdkProvider {

    /**
     * The deployment mode this provider supports.
     */
    enum Mode {
        /** Local / standalone — engine runs in the same JVM. */
        LOCAL,
        /** Remote / distributed — communicates with engine via HTTP. */
        REMOTE
    }

    /**
     * Returns the deployment mode this provider supports.
     *
     * @return the mode (LOCAL or REMOTE)
     */
    Mode mode();

    /**
     * Creates a new {@link GollekSdk} instance with the given configuration.
     *
     * @param config SDK configuration (may be {@code null} for defaults)
     * @return a configured SDK instance
     * @throws SdkException if creation fails
     */
    GollekSdk create(SdkConfig config) throws SdkException;

    /**
     * Provider priority. Lower values are preferred when multiple providers
     * are available for the same mode.
     *
     * @return priority value (default 100)
     */
    default int priority() {
        return 100;
    }
}
