package tech.kayys.golek.engine.model;

import tech.kayys.golek.core.provider.LLMProvider;
import tech.kayys.golek.model.ModelManifest;

import java.util.List;

public record RoutingDecision(
        String providerId,
        LLMProvider provider,
        int score,
        List<String> fallbackProviders,
        ModelManifest manifest,
        RoutingContext context
) {
}