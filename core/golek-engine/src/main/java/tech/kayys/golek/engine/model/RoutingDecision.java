package tech.kayys.golek.engine.model;

import java.util.List;

import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.provider.core.spi.LLMProvider;

public record RoutingDecision(
        String providerId,
        LLMProvider provider,
        int score,
        List<String> fallbackProviders,
        ModelManifest manifest,
        RoutingContext context) {
}