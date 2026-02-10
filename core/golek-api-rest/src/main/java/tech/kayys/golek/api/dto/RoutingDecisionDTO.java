package tech.kayys.golek.api.dto;

import tech.kayys.golek.engine.model.RoutingDecision;

public record RoutingDecisionDTO(String providerId, String reason) {
    public static RoutingDecisionDTO from(RoutingDecision decision) {
        return new RoutingDecisionDTO(decision.providerId(), "Score: " + decision.score());
    }

    public static RoutingDecisionDTO from(tech.kayys.golek.spi.provider.RoutingDecision decision) {
        return new RoutingDecisionDTO(decision.providerId(), "Score: " + decision.score());
    }
}
