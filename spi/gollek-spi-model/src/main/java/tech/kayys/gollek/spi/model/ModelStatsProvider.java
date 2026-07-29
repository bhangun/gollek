package tech.kayys.gollek.spi.model;

import tech.kayys.gollek.spi.spec.*;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.model.ModelFormat;

import io.smallrye.mutiny.Uni;

/**
 * Provider interface for model statistics.
 * This allows the registry module to get stats without depending on the engine
 * module.
 */
public interface ModelStatsProvider {

    /**
     * Get model statistics including inference counts.
     * 
     * @param requestId The request/tenant ID
     * @param modelId   The model ID
     * @return Model statistics
     */
    Uni<ModelRegistry.ModelStats> getModelStats(String requestId, String modelId);
}
