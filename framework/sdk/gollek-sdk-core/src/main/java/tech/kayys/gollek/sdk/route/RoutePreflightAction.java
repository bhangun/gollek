package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.Action;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Machine-readable remediation action emitted in route-report JSON.
 */
public record RoutePreflightAction(
        String kind,
        String reason,
        String description,
        List<String> argv,
        Map<String, Object> details) {
    public RoutePreflightAction(String kind, String reason, String description, List<String> argv) {
        this(kind, reason, description, argv, Map.of());
    }

    public RoutePreflightAction {
        argv = argv == null ? List.of() : List.copyOf(argv);
        details = details == null || details.isEmpty()
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(details));
    }

    Map<String, Object> toMap() {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put(Action.KIND, kind);
        value.put(Action.REASON, reason);
        value.put(Action.DESCRIPTION, description);
        value.put(Action.ARGV, argv);
        if (!details.isEmpty()) {
            value.put(Action.DETAILS, details);
        }
        return value;
    }
}
