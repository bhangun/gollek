package tech.kayys.golek.runtime;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.inference.StreamChunk;
import tech.kayys.golek.runtime.service.ModelRunnerFactory;

/**
 * Default implementation of the inference engine
 */
@ApplicationScoped
public class DefaultInferenceEngine implements InferenceEngine {

    private static final Logger LOG = Logger.getLogger(DefaultInferenceEngine.class);

    @Inject
    ModelRunnerFactory runnerFactory;

    private volatile boolean initialized = false;
    private volatile boolean healthy = true;
    private volatile long activeInferences = 0;
    private volatile long totalInferences = 0;
    private volatile long failedInferences = 0;
    private volatile double totalLatency = 0;

    @Override
    public void initialize() {
        LOG.info("Initializing inference engine...");
        
        // Initialize model runners
        runnerFactory.initialize();
        
        initialized = true;
        healthy = true;
        
        LOG.info("Inference engine initialized successfully");
    }

    @Override
    public Uni<InferenceResponse> infer(InferenceRequest request) {
        if (!initialized || !healthy) {
            return Uni.createFrom().failure(new IllegalStateException("Engine not ready"));
        }

        activeInferences++;
        long startTime = System.currentTimeMillis();

        return runnerFactory.getRunnerForModel(request.getModel())
                .flatMap(runner -> runner.infer(request))
                .onItem().invoke(response -> {
                    // Update stats
                    totalInferences++;
                    activeInferences--;
                    totalLatency += (System.currentTimeMillis() - startTime);
                })
                .onFailure().invoke(error -> {
                    // Update stats
                    failedInferences++;
                    activeInferences--;
                    LOG.errorf(error, "Inference failed for model: %s", request.getModel());
                });
    }

    @Override
    public Multi<StreamChunk> stream(InferenceRequest request) {
        if (!initialized || !healthy) {
            return Multi.createFrom().failure(new IllegalStateException("Engine not ready"));
        }

        activeInferences++;

        return runnerFactory.getRunnerForModel(request.getModel())
                .flatMapMulti(runner -> runner.stream(request))
                .onItem().invoke(chunk -> {
                    // Update stats for streaming
                    totalInferences++;
                })
                .onFailure().invoke(error -> {
                    // Update stats
                    failedInferences++;
                    activeInferences--;
                    LOG.errorf(error, "Stream failed for model: %s", request.getModel());
                })
                .onCompletion().invoke(() -> {
                    activeInferences--;
                });
    }

    @Override
    public void shutdown() {
        LOG.info("Shutting down inference engine...");
        
        // Shutdown model runners
        runnerFactory.closeAll();
        
        initialized = false;
        healthy = false;
        
        LOG.info("Inference engine shutdown complete");
    }

    @Override
    public boolean isHealthy() {
        return healthy && initialized;
    }

    @Override
    public EngineStats getStats() {
        double avgLatency = totalInferences > 0 ? totalLatency / totalInferences : 0.0;
        String status = healthy ? "HEALTHY" : "UNHEALTHY";
        
        return new EngineStats(
                activeInferences,
                totalInferences,
                failedInferences,
                avgLatency,
                status
        );
    }
}