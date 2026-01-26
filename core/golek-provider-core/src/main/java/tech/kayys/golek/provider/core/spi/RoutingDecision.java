package tech.kayys.golek.provider.core.spi;

import java.util.List;

import tech.kayys.golek.api.model.ModelManifest;
import tech.kayys.golek.api.provider.RoutingContext;

/**
 * Final routing decision with fallbacks
 */
public record RoutingDecision(
                String providerId,
                LLMProvider provider,
                int score,
                List<String> fallbackProviders,
                ModelManifest manifest,
                RoutingContext context) {
}
