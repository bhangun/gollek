package tech.kayys.gollek.spi.tensor;

import java.util.Comparator;
import java.util.ServiceLoader;

/**
 * Registry that selects the best available {@link ComputeBackend} at runtime.
 *
 * <p>Discovery order:
 * <ol>
 *   <li>All {@code ComputeBackend} implementations found via {@link ServiceLoader}</li>
 *   <li>Sorted by {@link ComputeBackend#priority()} descending (GPU &gt; CPU)</li>
 *   <li>Falls back to {@link CpuBackend} if no providers are found</li>
 * </ol>
 *
 * <p>The selected backend is cached for the lifetime of the JVM. To force
 * re-discovery (e.g. after loading a native library), call {@link #refresh()}.
 *
 * <h3>Example: registering a Metal backend</h3>
 * <pre>{@code
 * // META-INF/services/tech.kayys.gollek.spi.tensor.ComputeBackend
 * tech.kayys.gollek.kernel.metal.MetalComputeBackend
 * }</pre>
 */
public final class ComputeBackendRegistry {

    private static volatile ComputeBackend active;

    private ComputeBackendRegistry() {}

    /**
     * Returns the active compute backend (highest priority discovered backend).
     */
    public static ComputeBackend get() {
        if (active == null) {
            synchronized (ComputeBackendRegistry.class) {
                if (active == null) {
                    active = discover();
                }
            }
        }
        return active;
    }

    /**
     * Forces re-discovery of backends. Useful after dynamic native library loading.
     */
    public static void refresh() {
        synchronized (ComputeBackendRegistry.class) {
            active = discover();
        }
    }

    /**
     * Explicitly sets the active backend, bypassing ServiceLoader discovery.
     * Useful for testing or manual configuration.
     */
    public static void set(ComputeBackend backend) {
        synchronized (ComputeBackendRegistry.class) {
            active = backend;
        }
    }

    private static ComputeBackend discover() {
        return ServiceLoader.load(ComputeBackend.class)
                .stream()
                .map(ServiceLoader.Provider::get)
                .max(Comparator.comparingInt(ComputeBackend::priority))
                .orElse(new CpuBackend());
    }
}
