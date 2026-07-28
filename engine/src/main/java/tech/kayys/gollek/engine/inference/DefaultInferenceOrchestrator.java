package tech.kayys.gollek.engine.inference;

import tech.kayys.gollek.plugin.ModelRouterService;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.VerificationRequest;
import tech.kayys.gollek.spi.inference.VerificationResponse;
import io.smallrye.mutiny.Uni;

import java.util.List;

/**
 * Orchestrator that manages speculative decoding by routing between 
 * a draft model and a target model.
 */
public class DefaultInferenceOrchestrator {

    private final ModelRouterService routerService;
    
    public DefaultInferenceOrchestrator(ModelRouterService routerService) {
        this.routerService = routerService;
            }

    /**
     * Executes inference using speculative decoding if a draft model is available.
     */
        public Uni<InferenceResponse> executeSpeculative(InferenceRequest request) {
        return Uni.createFrom().failure(new UnsupportedOperationException("Speculative loop not fully implemented in skeleton"));
    }
}
