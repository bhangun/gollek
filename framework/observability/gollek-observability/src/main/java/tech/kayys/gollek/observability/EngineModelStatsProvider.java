package tech.kayys.gollek.observability;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;

import tech.kayys.alkhawarizm.spi.model.ModelRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelStatsProvider;

import java.time.LocalDateTime;

/**
 * Engine implementation of ModelStatsProvider.
 */
@ApplicationScoped
public class EngineModelStatsProvider implements ModelStatsProvider {

    @Override
    public Uni<ModelRegistry.ModelStats> getModelStats(String requestId, String modelId) {
        // In a real implementation, we would query the metrics for the specific model
        // For now, we'll return a basic structure

        return Uni.createFrom().item(() -> {
            return new ModelRegistry.ModelStats(
                    modelId,
                    "PRODUCTION", // Default stage
                    1, // Version count
                    0L, // Total inferences placeholder
                    LocalDateTime.now(),
                    LocalDateTime.now());
        });
    }
}
