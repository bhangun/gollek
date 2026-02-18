package tech.kayys.gollek.engine.routing;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.engine.model.ModelRegistryService;
import tech.kayys.gollek.engine.model.ModelRunner;
import tech.kayys.gollek.engine.model.ModelRunnerFactory;
import tech.kayys.gollek.engine.routing.policy.SelectionPolicy;
import tech.kayys.gollek.provider.core.exception.RetryableException;
import tech.kayys.gollek.spi.error.ErrorCode;
import tech.kayys.gollek.spi.exception.DeviceException;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.spi.exception.ModelException;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Intelligent router for selecting and executing the optimal model runner.
 *
 * <p>
 * Routing Strategy:
 * <ul>
 * <li>Check explicit runner preference in request</li>
 * <li>Apply selection policy (latency/cost/balanced)</li>
 * <li>Consider device availability and health</li>
 * <li>Implement fallback chain on failure</li>
 * <li>Track performance for future decisions</li>
 * </ul>
 *
 * @author Bhangun
 * @since 1.0.0
 */
@ApplicationScoped
public class ModelRouter {

    private static final Logger log = LoggerFactory.getLogger(ModelRouter.class);

    @Inject
    ModelRunnerFactory runnerFactory;

    @Inject
    SelectionPolicy selectionPolicy;

    @Inject
    ModelRegistryService registryService;

    @ConfigProperty(name = "inference.routing.enable-fallback", defaultValue = "true")
    boolean enableFallback;

    @ConfigProperty(name = "inference.routing.max-fallback-attempts", defaultValue = "2")
    int maxFallbackAttempts;

    @ConfigProperty(name = "inference.routing.default-runners")
    Optional<List<String>> defaultRunners;

    /**
     * Route inference request to optimal runner.
     * 
     * @param request The inference request
     * @return Inference response from selected runner
     * @throws InferenceException if all runners fail
     */
    public InferenceResponse route(InferenceRequest request) throws InferenceException {
        log.info("Routing inference request: modelId={}, requestId={}",
                request.getModel(), request.getRequestId());

        // 1. Get model manifest
        String requestId = "community";
        ModelManifest manifest = getModelManifest(requestId, request.getModel());

        // 2. Select runner candidates
        List<String> candidates = selectRunnerCandidates(request, manifest);

        if (candidates.isEmpty()) {
            throw new InferenceException(
                    ErrorCode.RUNTIME_INVALID_STATE,
                    "No suitable runners found for model: " + request.getModel());
        }

        log.debug("Runner candidates: {}", candidates);

        // 3. Try candidates in order with fallback
        Exception lastException = null;
        int attempts = 0;

        for (String runnerName : candidates) {
            if (!enableFallback && attempts > 0) {
                break; // No fallback allowed
            }

            if (attempts >= maxFallbackAttempts + 1) {
                break; // Max attempts reached
            }

            attempts++;

            try {
                log.info("Attempting runner: {} (attempt {}/{})",
                        runnerName, attempts, maxFallbackAttempts + 1);

                // Get or create runner instance
                ModelRunner runner = runnerFactory.getOrCreateRunner(runnerName, manifest);

                // Check runner health
                if (!runner.health()) {
                    log.warn("Runner {} is unhealthy, trying next", runnerName);
                    lastException = new DeviceException(
                            ErrorCode.DEVICE_NOT_AVAILABLE,
                            "Runner is unhealthy: " + runnerName,
                            runner.deviceType().getId());
                    continue;
                }

                // Execute inference
                InferenceResponse response = runner.infer(request);

                log.info("Inference successful: runner={}, latencyMs={}",
                        runnerName, response.getDurationMs());

                return response;

            } catch (RetryableException e) {
                log.warn("Retryable error on runner {}: {}", runnerName, e.getMessage());
                lastException = e;
                // Continue to next runner
            } catch (DeviceException e) {
                log.warn("Device error on runner {}: {}", runnerName, e.getMessage());
                lastException = e;
                // Continue to next runner
            } catch (Exception e) {
                log.error("Runner {} failed with non-retryable error", runnerName, e);
                lastException = e;

                // Non-retryable errors should fail fast unless it's the last runner
                if (!enableFallback || candidates.indexOf(runnerName) == candidates.size() - 1) {
                    throw mapToInferenceException(e);
                }
            }
        }

        // All runners failed
        throw new InferenceException(
                ErrorCode.ALL_RUNNERS_FAILED,
                "All " + attempts + " runner attempts failed for model: " + request.getModel(),
                lastException).addContext("attempts", attempts)
                .addContext("candidates", candidates);
    }

    /**
     * Select runner for streaming inference.
     */
    public ModelRunner selectRunner(InferenceRequest request, boolean requiresStreaming) {
        String requestId = "community";
        ModelManifest manifest = getModelManifest(requestId, request.getModel());
        List<String> candidates = selectRunnerCandidates(request, manifest);

        for (String runnerName : candidates) {
            try {
                ModelRunner runner = runnerFactory.getOrCreateRunner(runnerName, manifest);

                if (requiresStreaming && !runner.capabilities().isSupportsStreaming()) {
                    continue;
                }

                if (runner.health()) {
                    return runner;
                }
            } catch (Exception e) {
                log.warn("Failed to get runner {}: {}", runnerName, e.getMessage());
            }
        }

        throw new InferenceException(
                ErrorCode.RUNTIME_INVALID_STATE,
                "No suitable runner found for streaming");
    }

    /**
     * Select runner candidates based on request and manifest.
     */
    private List<String> selectRunnerCandidates(InferenceRequest request, ModelManifest manifest) {
        // 1. Check explicit preference
        String preferred = request.getPreferredProvider().orElse(null);
        if (preferred != null && !preferred.isEmpty()) {

            // Validate runner exists and supports model
            if (isRunnerCompatible(preferred, manifest)) {
                log.debug("Using preferred runner: {}", preferred);
                // Put preferred first, but still allow fallback
                List<String> candidates = new ArrayList<>();
                candidates.add(preferred);

                // Add other compatible runners as fallback
                if (enableFallback) {
                    candidates.addAll(
                            getCompatibleRunners(manifest).stream()
                                    .filter(r -> !r.equals(preferred))
                                    .collect(Collectors.toList()));
                }

                return candidates;
            } else {
                log.warn("Preferred runner {} is not compatible with model {}",
                        preferred, manifest.name());
            }
        }

        // 2. Apply selection policy
        List<String> compatibleRunners = getCompatibleRunners(manifest);

        if (compatibleRunners.isEmpty()) {
            return Collections.emptyList();
        }

        // 3. Use policy to rank runners
        return selectionPolicy.rankRunners(request, manifest, compatibleRunners);
    }

    // ...
    /**
     * Check if runner is compatible with model.
     */
    private boolean isRunnerCompatible(String runnerName, ModelManifest manifest) {
        try {
            // Extract framework from runner name (e.g., "litert-cpu" -> "litert")
            String framework = extractFramework(runnerName);

            // Check if manifest supports this framework
            // Assuming framework string can be matched against keys in artifacts map
            return manifest.artifacts().keySet().stream()
                    .anyMatch(format -> framework.equalsIgnoreCase(format.name()));
        } catch (Exception e) {
            log.warn("Error checking compatibility for runner {}: {}", runnerName, e.getMessage());
            return false;
        }
    }

    // ...
    /**
     * Get runners compatible with model manifest.
     */
    /**
     * Get runners compatible with model manifest.
     */
    private List<String> getCompatibleRunners(ModelManifest manifest) {
        List<String> compatible = new ArrayList<>();
        List<String> availableRunners = runnerFactory.getAvailableRunners();

        for (String runnerName : availableRunners) {
            if (isRunnerCompatible(runnerName, manifest)) {
                compatible.add(runnerName);
            }
        }

        // If no compatible runners, try default runners
        if (compatible.isEmpty() && defaultRunners.isPresent()) {
            log.warn("No compatible runners found, trying defaults");
            return new ArrayList<>(defaultRunners.get());
        }

        return compatible;
    }

    /**
     * Extract framework name from runner name.
     */
    private String extractFramework(String runnerName) {
        // Format: "framework-device" (e.g., "litert-cpu", "onnx-cuda")
        int dashIndex = runnerName.indexOf('-');
        if (dashIndex > 0) {
            return runnerName.substring(0, dashIndex);
        }
        return runnerName;
    }

    /**
     * Get model manifest from registry.
     */
    private ModelManifest getModelManifest(String requestId, String modelId) {
        try {
            // Parse model ID (format: "name:version" or just "name")
            String[] parts = modelId.split(":");
            String modelName = parts[0];
            String version = parts.length > 1 ? parts[1] : "latest";

            ModelManifest manifest = registryService.getManifest(requestId, modelName, version).await().indefinitely();

            if (manifest == null) {
                throw new ModelException(
                        ErrorCode.MODEL_NOT_FOUND,
                        "Model not found: " + modelId,
                        modelId);
            }

            return manifest;

        } catch (ModelException e) {
            throw e;
        } catch (Exception e) {
            throw new ModelException(
                    ErrorCode.MODEL_NOT_FOUND,
                    "Failed to load model manifest: " + modelId,
                    modelId,
                    e);
        }
    }

    /**
     * Map generic exception to InferenceException.
     */
    private InferenceException mapToInferenceException(Throwable t) {
        if (t instanceof InferenceException ie) {
            return ie;
        }

        return new InferenceException(
                ErrorCode.RUNTIME_INFERENCE_FAILED,
                "Inference failed: " + t.getMessage(),
                t);
    }

    /**
     * Get routing statistics for monitoring.
     */
    public RoutingStats getStats() {
        List<String> availableRunners = runnerFactory.getAvailableRunners();

        Map<String, Boolean> runnerHealth = new HashMap<>();
        for (String runnerName : availableRunners) {
            try {
                ModelRunner runner = runnerFactory.getRunner(runnerName);
                runnerHealth.put(runnerName, runner != null && runner.health());
            } catch (Exception e) {
                runnerHealth.put(runnerName, false);
            }
        }

        return new RoutingStats(
                availableRunners.size(),
                runnerHealth.values().stream().filter(h -> h).count(),
                runnerHealth);
    }

    public record RoutingStats(
            int totalRunners,
            long healthyRunners,
            Map<String, Boolean> runnerHealth) {
    }
}
