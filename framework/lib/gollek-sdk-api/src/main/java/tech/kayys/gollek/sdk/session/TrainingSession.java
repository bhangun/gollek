package tech.kayys.gollek.sdk.session;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.function.BiFunction;

/**
 * Training session — manages a complete training lifecycle with metrics,
 * checkpointing, and early stopping.
 *
 * <p>DISABLED: This class requires tech.kayys.gollek.ml.nn and related modules
 * which have been disabled due to structural compilation issues. Re-enable this
 * class once those modules are refactored and completed.
 *
 * @deprecated Requires nn, data, and metrics modules to be re-enabled
 */
@Deprecated(since = "0.1.0", forRemoval = true)
public final class TrainingSession implements AutoCloseable {

    private TrainingSession() {
        throw new UnsupportedOperationException(
            "TrainingSession requires gollek-sdk-nn, gollek-sdk-data, and gollek-sdk-metrics modules. " +
            "These modules are currently disabled due to structural issues. " +
            "Please check documentation for re-enabling instructions.");
    }

    @Override public void close() { /* no-op */ }

    public static Builder builder() {
        throw new UnsupportedOperationException(
            "TrainingSession requires gollek-sdk-nn, gollek-sdk-data, and gollek-sdk-metrics modules. " +
            "These modules are currently disabled due to structural issues.");
    }

    /**
     * Builder for {@link TrainingSession} - DISABLED.
     */
    public static final class Builder {
        @Deprecated(since = "0.1.0", forRemoval = true)
        public TrainingSession build() {
            throw new UnsupportedOperationException(
                "TrainingSession requires gollek-sdk-nn, gollek-sdk-data, and gollek-sdk-metrics modules. " +
                "These modules are currently disabled due to structural issues.");
        }
    }
}
