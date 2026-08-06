package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.Problem;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Machine-readable route preflight problem emitted in route-report JSON.
 */
public record RoutePreflightProblem(String code, String severity, String message, Map<String, Object> details) {
    public RoutePreflightProblem(String code, String severity, String message) {
        this(code, severity, message, Map.of());
    }

    public RoutePreflightProblem {
        details = details == null || details.isEmpty()
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(details));
    }

    Map<String, Object> toMap() {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put(Problem.CODE, code);
        value.put(Problem.SEVERITY, severity);
        value.put(Problem.MESSAGE, message);
        if (!details.isEmpty()) {
            value.put(Problem.DETAILS, details);
        }
        return value;
    }
}
