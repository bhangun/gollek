/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.sdk.vision.backend;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Global registry for vision backend implementations.
 *
 * <p>This registry manages available vision backends and provides automatic
 * selection based on availability and priority. Backends register themselves
 * at initialization time.</p>
 *
 * <h2>Backend Selection Strategy</h2>
 * <p>Backends are selected in the following order:</p>
 * <ol>
 *   <li>Priority order (lower number = higher priority)</li>
 *   <li>Availability (backend must be available on the system)</li>
 *   <li>Fallback to CPU if no accelerated backend available</li>
 * </ol>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Get the best available backend
 * VisionBackendProvider backend = VisionBackendRegistry.getDefault();
 *
 * // Get specific backend
 * VisionBackendProvider cudaBackend = VisionBackendRegistry.get("cuda");
 *
 * // Check availability
 * if (VisionBackendRegistry.isAvailable("metal")) {
 *     VisionBackendProvider metalBackend = VisionBackendRegistry.get("metal");
 *     // Use Metal GPU backend
 * }
 * }</pre>
 *
 * @author Gollek Team
 * @since 0.2.0
 */
public final class VisionBackendRegistry {

    private static final Logger LOG = LoggerFactory.getLogger(VisionBackendRegistry.class);

    private static final Map<String, VisionBackendProvider> BACKENDS = new ConcurrentHashMap<>();
    private static volatile VisionBackendProvider DEFAULT_BACKEND;
    private static volatile boolean initialized = false;

    private VisionBackendRegistry() {
        // Prevent instantiation
    }

    /**
     * Register a vision backend provider.
     *
     * @param provider the backend provider to register
     * @throws NullPointerException if provider is null
     */
    public static synchronized void register(VisionBackendProvider provider) {
        if (provider == null) {
            throw new NullPointerException("Vision backend provider cannot be null");
        }

        String backendId = provider.getBackendId();
        BACKENDS.put(backendId, provider);

        LOG.info("Registered vision backend: {} (priority: {}, available: {})",
                backendId, provider.getPriority(), provider.isAvailable());

        // Reset default backend when new one registers
        DEFAULT_BACKEND = null;
    }

    /**
     * Unregister a vision backend provider.
     *
     * @param backendId the backend identifier
     * @return the unregistered provider, or null if not found
     */
    public static synchronized VisionBackendProvider unregister(String backendId) {
        VisionBackendProvider provider = BACKENDS.remove(backendId);
        if (provider != null) {
            LOG.info("Unregistered vision backend: {}", backendId);
            DEFAULT_BACKEND = null;
        }
        return provider;
    }

    /**
     * Get a vision backend by identifier.
     *
     * @param backendId the backend identifier
     * @return the vision backend provider
     * @throws IllegalArgumentException if backend not found
     */
    public static VisionBackendProvider get(String backendId) {
        VisionBackendProvider provider = BACKENDS.get(backendId);
        if (provider == null) {
            throw new IllegalArgumentException("Vision backend not found: " + backendId
                    + ". Available: " + BACKENDS.keySet());
        }
        return provider;
    }

    /**
     * Get the default vision backend.
     *
     * <p>Returns the best available backend based on priority and availability.
     * If multiple backends are available, returns the one with lowest priority number.</p>
     *
     * @return the default vision backend provider
     * @throws IllegalStateException if no backends are registered
     */
    public static VisionBackendProvider getDefault() {
        if (DEFAULT_BACKEND != null) {
            return DEFAULT_BACKEND;
        }

        synchronized (VisionBackendRegistry.class) {
            if (DEFAULT_BACKEND != null) {
                return DEFAULT_BACKEND;
            }

            if (BACKENDS.isEmpty()) {
                throw new IllegalStateException("No vision backends registered. " +
                        "Ensure CPU backend or hardware-specific backends are loaded.");
            }

            // Find best available backend by priority
            VisionBackendProvider best = BACKENDS.values()
                    .stream()
                    .filter(VisionBackendProvider::isAvailable)
                    .min(Comparator.comparingInt(VisionBackendProvider::getPriority))
                    .orElseThrow(() -> new IllegalStateException(
                            "No available vision backends. Registered: " + BACKENDS.keySet()));

            DEFAULT_BACKEND = best;
            LOG.info("Selected default vision backend: {}", best.getBackendId());
            return DEFAULT_BACKEND;
        }
    }

    /**
     * Check if a backend is registered and available.
     *
     * @param backendId the backend identifier
     * @return true if backend exists and is available
     */
    public static boolean isAvailable(String backendId) {
        VisionBackendProvider provider = BACKENDS.get(backendId);
        return provider != null && provider.isAvailable();
    }

    /**
     * Get all registered backend identifiers.
     *
     * @return set of backend identifiers
     */
    public static Set<String> getRegisteredBackends() {
        return new HashSet<>(BACKENDS.keySet());
    }

    /**
     * Get all available backend identifiers.
     *
     * @return set of available backend identifiers
     */
    public static Set<String> getAvailableBackends() {
        return BACKENDS.values()
                .stream()
                .filter(VisionBackendProvider::isAvailable)
                .map(VisionBackendProvider::getBackendId)
                .collect(Collectors.toSet());
    }

    /**
     * Get backend selection info for diagnostics.
     *
     * @return formatted backend information
     */
    public static String getDiagnosticsInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("Vision Backends:\n");

        if (BACKENDS.isEmpty()) {
            sb.append("  No backends registered\n");
        } else {
            List<VisionBackendProvider> sorted = BACKENDS.values()
                    .stream()
                    .sorted(Comparator.comparingInt(VisionBackendProvider::getPriority))
                    .collect(Collectors.toList());

            for (VisionBackendProvider provider : sorted) {
                String status = provider.isAvailable() ? "✓" : "✗";
                String memory = provider.getMemoryUsage() > 0
                        ? String.format(" (%.2f MB)", provider.getMemoryUsage() / 1024.0 / 1024.0)
                        : "";
                sb.append(String.format("  %s %-8s [P%d] %s%s\n",
                        status,
                        provider.getBackendId(),
                        provider.getPriority(),
                        provider.getMemoryStrategy(),
                        memory));
            }
        }

        if (DEFAULT_BACKEND != null) {
            sb.append("  Default: ").append(DEFAULT_BACKEND.getBackendId()).append("\n");
        }

        return sb.toString();
    }

    /**
     * Clear all registered backends.
     *
     * <p><strong>WARNING:</strong> This is for testing only. Clears all backends
     * and resets the default backend.</p>
     */
    public static synchronized void clear() {
        BACKENDS.clear();
        DEFAULT_BACKEND = null;
        initialized = false;
        LOG.info("Cleared all vision backends");
    }

    /**
     * Initialize default backends if not already initialized.
     *
     * <p>This method is called automatically but can be called explicitly
     * to ensure CPU backend is registered.</p>
     */
    public static synchronized void ensureInitialized() {
        if (!initialized) {
            // CPU backend will be registered via discovery mechanism
            // or explicitly by the application
            initialized = true;
            LOG.debug("Vision backend registry initialized");
        }
    }
}
