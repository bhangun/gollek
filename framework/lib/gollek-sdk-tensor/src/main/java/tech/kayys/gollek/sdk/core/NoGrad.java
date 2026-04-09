package tech.kayys.gollek.sdk.core;

/**
 * NoGrad context manager for disabling gradient computation.
 *
 * <p>Similar to PyTorch's `torch.no_grad()`, this context manager disables
 * gradient calculation temporarily, which saves memory and speeds up inference.</p>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * try (NoGrad ctx = new NoGrad()) {
 *     Tensor output = model.forward(input);
 *     // No gradients computed
 * }
 * // Gradients re-enabled
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public final class NoGrad implements AutoCloseable {

    private static volatile boolean enabled = true;

    /**
     * Check if gradient computation is enabled.
     */
    public static boolean isEnabled() {
        return enabled;
    }

    /**
     * Create NoGrad context (disables gradients).
     */
    public NoGrad() {
        enabled = false;
    }

    @Override
    public void close() {
        enabled = true;
    }

    /**
     * Set gradient enabled state.
     *
     * @param enabled true to enable gradients
     */
    public static void setEnabled(boolean enabled) {
        NoGrad.enabled = enabled;
    }
}
