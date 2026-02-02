package tech.kayys.golek.engine.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Describes capabilities and features of a model runner.
 * 
 * @author bhangun
 * @since 1.0.0
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RunnerCapabilities {

    /**
     * Whether runner supports streaming inference.
     */
    @Builder.Default
    private boolean supportsStreaming = false;

    /**
     * Whether runner supports batch inference.
     */
    @Builder.Default
    private boolean supportsBatching = true;

    /**
     * Whether runner supports quantization.
     */
    @Builder.Default
    private boolean supportsQuantization = false;

    /**
     * Maximum batch size supported.
     */
    @Builder.Default
    private int maxBatchSize = 1;

    /**
     * Supported tensor data types.
     */
    private String[] supportedDataTypes;
}
