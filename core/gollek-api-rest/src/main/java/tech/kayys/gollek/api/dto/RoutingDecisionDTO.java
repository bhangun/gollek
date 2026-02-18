package tech.kayys.gollek.api.dto;

import tech.kayys.gollek.spi.provider.RoutingDecision;

public record RoutingDecisionDTO(String providerId, String reason) {
    public static RoutingDecisionDTO from(RoutingDecision decision) {
        return new RoutingDecisionDTO(decision.providerId(), "Score: " + decision.score());
    }

    public static RoutingDecisionDTO from(tech.kayys.gollek.spi.routing.RoutingDecision decision) {
        return new RoutingDecisionDTO(decision.selectedProviderId(), "Score: " + decision.score());
    }
}
