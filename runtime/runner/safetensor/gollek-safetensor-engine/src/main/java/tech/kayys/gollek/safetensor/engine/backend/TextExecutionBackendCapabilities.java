package tech.kayys.gollek.safetensor.engine.backend;

/**
 * Capability flags for a text execution backend.
 *
 * <p>This is the minimal contract needed to stop provider/orchestration code from
 * hard-coding execution assumptions. Backends declare what kind of inference
 * behavior they support; orchestration can then validate or normalize requests
 * without knowing backend internals.
 */
public record TextExecutionBackendCapabilities(
        boolean supportsStreaming,
        boolean supportsWeightQuantization,
        boolean supportsKvCacheQuantization,
        boolean supportsTextPrefixCaching,
        boolean supportsContinuousBatching,
        boolean supportsBackendResidentWeights,
        boolean supportsStatefulPreparedModels) {

    public static TextExecutionBackendCapabilities directReference() {
        return new TextExecutionBackendCapabilities(
                true,
                true,
                true,
                false,
                false,
                false,
                true);
    }
}
