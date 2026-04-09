package tech.kayys.gollek.ml.autograd;

/**
 * Context manager for disabling gradient computation.
 * <p>
 * When inside a {@code NoGrad} scope, new tensors created by operations
 * will not track gradients, improving performance during inference.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * try (var _ = NoGrad.enter()) {
 *     var output = model.forward(input);
 *     // no gradients tracked here
 * }
 * }</pre>
 */
public final class NoGrad implements AutoCloseable {

    private static final ThreadLocal<Boolean> ENABLED = ThreadLocal.withInitial(() -> false);

    private final boolean previousState;

    private NoGrad() {
        this.previousState = ENABLED.get();
        ENABLED.set(true);
    }

    /** Enter a no-grad context. Use with try-with-resources. */
    public static NoGrad enter() {
        return new NoGrad();
    }

    /** Check if gradient computation is currently disabled. */
    public static boolean isActive() {
        return ENABLED.get();
    }

    @Override
    public void close() {
        ENABLED.set(previousState);
    }
}
