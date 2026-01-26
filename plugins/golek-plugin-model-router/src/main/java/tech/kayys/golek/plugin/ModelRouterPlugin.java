package tech.kayys.golek.plugin;

import tech.kayys.golek.inference.kernel.plugin.InferencePhasePlugin;
import tech.kayys.golek.inference.kernel.plugin.InferencePhasePlugin.PluginException;
import tech.kayys.golek.inference.kernel.engine.EngineContext;
import tech.kayys.golek.inference.kernel.execution.ExecutionContext;
import tech.kayys.golek.inference.kernel.pipeline.InferencePhase;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.util.Map;

/**
 * Intelligent model selection and provider routing.
 * Determines the optimal provider for a given model and request context.
 */
@ApplicationScoped
public class ModelRouterPlugin implements InferencePhasePlugin {

    @Inject
    ModelRouterService routerService;

    @Override
    public String id() {
        return "tech.kayys.golek.routing.model";
    }

    @Override
    public int order() {
        return 1; // Execute early in route phase
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.ROUTE; // Correct phase for routing decisions
    }

    @Override
    public void initialize(EngineContext context) {
        System.out.println("Model Router Plugin initialized");
    }

    @Override
    public void shutdown() {
        System.out.println("Model Router Plugin shut down");
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        try {
            // Get model ID from context - try multiple sources
            String modelId = extractModelId(context);
            if (modelId == null || modelId.trim().isEmpty()) {
                throw new PluginException("Model ID is required for routing");
            }

            // Get tenant context
            String tenantId = context.tenantContext().getTenantId();

            // Extract additional context for routing decisions
            Map<String, Object> requestContext = extractRequestContext(context);

            // Select best provider for this model and tenant
            String selectedProviderId = routerService.selectProvider(modelId, tenantId, requestContext);

            if (selectedProviderId == null) {
                throw new PluginException("No suitable provider found for model: " + modelId);
            }

            // Store routing decision in context for downstream phases
            context.putVariable("selectedProviderId", selectedProviderId);

            // Create detailed routing decision object
            RoutingDecision routingDecision = routerService.getRoutingDecision(modelId, tenantId, requestContext);
            context.putVariable("routingDecision", routingDecision);

            // Add audit information
            context.putMetadata("modelRouted", true);
            context.putMetadata("routedModelId", modelId);
            context.putMetadata("selectedProviderId", selectedProviderId);
            context.putMetadata("routingScore", routingDecision.getScore());
            context.putMetadata("routingTimestamp", routingDecision.getTimestamp());

            System.out.printf("Model %s routed to provider %s with score %.2f%n",
                            modelId, selectedProviderId, routingDecision.getScore());

        } catch (Exception e) {
            // Log the error but don't expose internal details to the client
            System.err.println("Error during model routing: " + e.getMessage());

            // Add error information to context for observability
            context.putMetadata("routingError", e.getClass().getSimpleName());
            context.putMetadata("routingErrorMessage", e.getMessage());

            throw new PluginException("Model routing failed: " + e.getMessage(), e);
        }
    }

    private String extractModelId(ExecutionContext context) {
        // Try multiple sources for model ID
        Object modelObj = context.variables().get("modelId");
        if (modelObj != null) {
            return modelObj.toString();
        }

        modelObj = context.metadata().get("modelId");
        if (modelObj != null) {
            return modelObj.toString();
        }

        // Try to get from a more generic 'model' field
        modelObj = context.variables().get("model");
        if (modelObj != null) {
            return modelObj.toString();
        }

        modelObj = context.metadata().get("model");
        if (modelObj != null) {
            return modelObj.toString();
        }

        return null;
    }

    private Map<String, Object> extractRequestContext(ExecutionContext context) {
        Map<String, Object> requestContext = new java.util.HashMap<>();

        // Extract relevant context for routing decisions
        Object priority = context.variables().get("priority");
        if (priority != null) {
            requestContext.put("priority", priority);
        }

        Object maxTokens = context.variables().get("max_tokens");
        if (maxTokens != null) {
            requestContext.put("max_tokens", maxTokens);
        }

        Object temperature = context.variables().get("temperature");
        if (temperature != null) {
            requestContext.put("temperature", temperature);
        }

        // Add any other relevant context for routing
        Object requestSize = context.variables().get("request_size");
        if (requestSize != null) {
            requestContext.put("request_size", requestSize);
        }

        return requestContext;
    }
}