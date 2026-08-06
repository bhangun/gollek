package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ActionKind;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ProblemCode;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

public record RoutePreflightReport(
        boolean requireLocal,
        List<RoutePreflightProblem> problems,
        List<RoutePreflightAction> nextActions) {
    public static final int NOT_READY_EXIT_CODE = 2;

    public static RoutePreflightReport evaluate(
            String requestedModel,
            String effectiveModel,
            String localPath,
            String provider,
            String format,
            boolean mutationAllowed,
            boolean requireLocal) {
        return new RoutePreflightReport(
                requireLocal,
                problems(localPath, provider, format, requireLocal),
                nextActions(requestedModel, effectiveModel, localPath, provider, format, mutationAllowed));
    }

    public boolean passed() {
        return problems.isEmpty();
    }

    public String status() {
        return passed() ? "passed" : "failed";
    }

    public int problemCount() {
        return problems.size();
    }

    public int exitCode() {
        return passed() ? 0 : NOT_READY_EXIT_CODE;
    }

    public List<Map<String, Object>> problemMaps() {
        return problems.stream().map(RoutePreflightProblem::toMap).toList();
    }

    public List<Map<String, Object>> nextActionMaps() {
        return nextActions.stream().map(RoutePreflightAction::toMap).toList();
    }

    public RoutePreflightReport withAdditionalProblems(List<RoutePreflightProblem> additionalProblems) {
        if (additionalProblems == null || additionalProblems.isEmpty()) {
            return this;
        }
        List<RoutePreflightProblem> merged = new ArrayList<>(problems);
        merged.addAll(additionalProblems);
        return new RoutePreflightReport(requireLocal, List.copyOf(merged), nextActions);
    }

    public RoutePreflightReport withAdditionalActions(List<RoutePreflightAction> additionalActions) {
        if (additionalActions == null || additionalActions.isEmpty()) {
            return this;
        }
        LinkedHashSet<RoutePreflightAction> merged = new LinkedHashSet<>(nextActions);
        merged.addAll(additionalActions);
        return new RoutePreflightReport(requireLocal, problems, List.copyOf(merged));
    }

    public static boolean passed(String localPath, String provider, String format, boolean requireLocal) {
        return problems(localPath, provider, format, requireLocal).isEmpty();
    }

    public static int exitCode(String localPath, String provider, String format, boolean requireLocal) {
        return passed(localPath, provider, format, requireLocal) ? 0 : NOT_READY_EXIT_CODE;
    }

    public static List<RoutePreflightProblem> problems(
            String localPath,
            String provider,
            String format,
            boolean requireLocal) {
        if (!requireLocal) {
            return List.of();
        }
        List<RoutePreflightProblem> problems = new ArrayList<>();
        if (!hasText(localPath)) {
            problems.add(new RoutePreflightProblem(
                    ProblemCode.MODEL_NOT_LOCAL,
                    "error",
                    "Strict route preflight requires a local model artifact."));
            return problems;
        }
        if (!hasText(provider)) {
            problems.add(new RoutePreflightProblem(
                    ProblemCode.PROVIDER_NOT_RESOLVED,
                    "error",
                    "Strict route preflight resolved a local artifact but no execution provider was selected."));
        }
        if (!hasText(format)) {
            problems.add(new RoutePreflightProblem(
                    ProblemCode.FORMAT_NOT_RESOLVED,
                    "error",
                    "Strict route preflight resolved a local artifact but no model format was selected."));
        }
        return problems;
    }

    public static List<RoutePreflightAction> nextActions(
            String requestedModel,
            String effectiveModel,
            String localPath,
            String provider,
            String format,
            boolean mutationAllowed) {
        String modelRef = firstText(requestedModel, effectiveModel);
        List<RoutePreflightAction> actions = new ArrayList<>();
        if (!hasText(localPath) && hasText(modelRef)) {
            actions.add(new RoutePreflightAction(
                    ActionKind.PULL_MODEL,
                    ProblemCode.MODEL_NOT_LOCAL,
                    "Install the model artifact locally before strict route preflight.",
                    List.of("gollek", "pull", modelRef)));
            if (!mutationAllowed) {
                actions.add(new RoutePreflightAction(
                        ActionKind.ALLOW_PULL_RESOLUTION,
                        "local_only_preflight",
                        "Opt in to pull-capable route resolution when local-only inspection is not enough.",
                        List.of("gollek", "run", "--model", modelRef,
                                "--route-report-json", "--route-report-allow-pull")));
            }
        }
        if (hasText(localPath) && (!hasText(provider) || !hasText(format))) {
            actions.add(new RoutePreflightAction(
                    ActionKind.INSPECT_MODULES,
                    "route_not_runnable",
                    "Inspect attached provider and runner plugins before serving this local artifact.",
                    List.of("gollek", "modules", "--json")));
        }
        return actions;
    }

    private static String firstText(String first, String second) {
        if (hasText(first)) {
            return first;
        }
        return hasText(second) ? second : null;
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }
}
