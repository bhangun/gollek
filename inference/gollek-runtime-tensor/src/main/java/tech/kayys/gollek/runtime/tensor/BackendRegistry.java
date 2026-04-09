package tech.kayys.gollek.runtime.tensor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Global registry for backend implementations in the Gollek inference runtime.
 * <p>
 * This class provides a central lookup table for available inference backends.
 * Backends register themselves at startup, and the runtime/graph executor
 * queries this registry to dispatch operations to the appropriate native
 * implementation.
 * <p>
 * <h2>Registry Pattern</h2>
 * <p>
 * The registry implements the Service Locator pattern:
 * </p>
 * <ul>
 *   <li><strong>Registration:</strong> Backends call {@link #register(Backend)} at startup</li>
 *   <li><strong>Lookup:</strong> Runtime calls {@link #get(BackendType)} to obtain backend</li>
 *   <li><strong>Availability Check:</strong> Call {@link #isAvailable(BackendType)} before lookup</li>
 * </ul>
 * <p>
 * <h2>Thread Safety</h2>
 * <p>
 * This class is fully thread-safe:
 * </p>
 * <ul>
 *   <li>Uses {@link ConcurrentHashMap} for lock-free concurrent access</li>
 *   <li>Registration and lookup can occur from multiple threads safely</li>
 *   <li>Best practice: Register all backends during initialization, before concurrent access</li>
 * </ul>
 * <p>
 * <h2>Backend Registration</h2>
 * <p>
 * Backends typically register themselves during static initialization or module
 * startup:
 * </p>
 * <pre>{@code
 * // In backend implementation
 * public class GGMLBackend implements Backend {
 *     static {
 *         BackendRegistry.register(new GGMLBackend());
 *     }
 *     
 *     {@literal @Override}
 *     public BackendType type() {
 *         return BackendType.GGML;
 *     }
 * }
 * }</pre>
 * <p>
 * <h2>Usage Examples</h2>
 * <pre>{@code
 * // Check if backend is available
 * if (BackendRegistry.isAvailable(BackendType.GGML)) {
 *     // Get the backend instance
 *     Backend backend = BackendRegistry.get(BackendType.GGML);
 *     
 *     // Create tensors through the backend
 *     Tensor tensor = backend.createTensor(shape, dtype, device, ctx);
 * }
 * 
 * // List available backends (for diagnostics)
 * System.out.println("Available: " + 
 *     BackendRegistry.getRegisteredTypes());
 * }</pre>
 * <p>
 * <h2>Testing Support</h2>
 * <p>
 * The {@link #clear()} method unregisters all backends, useful for test
 * isolation. Tests can register mock backends for controlled testing.
 * </p>
 *
 * @see Backend
 * @see BackendType
 * @since 1.0
 */
public final class BackendRegistry {

    /**
     * Thread-safe map from backend type to implementation instance.
     * Each backend type maps to exactly one registered implementation.
     */
    private static final Map<BackendType, Backend> BACKENDS = new ConcurrentHashMap<>();

    /**
     * Private constructor to prevent instantiation.
     * This class is a utility with only static methods.
     */
    private BackendRegistry() {}

    /**
     * Registers a backend implementation in the global registry.
     * <p>
     * The backend is stored under its {@link Backend#type()} identifier.
     * If a backend of the same type was previously registered, it is replaced.
     * </p>
     * <p>
     * <strong>Thread Safety:</strong> This method is thread-safe. However,
     * best practice is to register all backends during initialization before
     * concurrent access begins.
     * </p>
     *
     * @param backend the backend implementation to register
     * @throws NullPointerException if backend is null
     * @see Backend#type()
     */
    public static void register(Backend backend) {
        BACKENDS.put(backend.type(), backend);
    }

    /**
     * Retrieves a registered backend by type.
     * <p>
     * Returns the backend implementation for the specified type. If no backend
     * of that type is registered, an exception is thrown with details about
     * available backends.
     * </p>
     * <p>
     * <strong>Usage Pattern:</strong>
     * </p>
     * <pre>{@code
     * // Safe pattern - check availability first
     * if (BackendRegistry.isAvailable(type)) {
     *     Backend backend = BackendRegistry.get(type);
     *     // Use backend...
     * }
     * 
     * // Or handle exception
     * try {
     *     Backend backend = BackendRegistry.get(type);
     * } catch (IllegalStateException e) {
     *     // Fallback behavior...
     * }
     * }</pre>
     *
     * @param type the backend type to look up
     * @return the registered backend implementation
     * @throws IllegalStateException if no backend of that type is registered
     * @throws NullPointerException if type is null
     */
    public static Backend get(BackendType type) {
        Backend backend = BACKENDS.get(type);
        if (backend == null) {
            throw new IllegalStateException("Backend not registered: " + type
                + ". Available: " + BACKENDS.keySet());
        }
        return backend;
    }

    /**
     * Checks if a backend of the specified type is registered.
     * <p>
     * This method provides a safe way to check backend availability before
     * attempting to use it. Returns {@code true} if the backend is registered,
     * {@code false} otherwise.
     * </p>
     * <p>
     * <strong>Thread Safety:</strong> This method is thread-safe and lock-free.
     * </p>
     *
     * @param type the backend type to check
     * @return true if the backend is registered, false otherwise
     * @throws NullPointerException if type is null
     */
    public static boolean isAvailable(BackendType type) {
        return BACKENDS.containsKey(type);
    }

    /**
     * Unregisters all backends from the registry.
     * <p>
     * <strong>Warning:</strong> This method is intended for testing purposes only.
     * Calling this during active inference will cause subsequent {@link #get(BackendType)}
     * calls to fail.
     * </p>
     * <p>
     * After clearing, backends must re-register themselves before use.
     * </p>
     */
    public static void clear() {
        BACKENDS.clear();
    }
}
