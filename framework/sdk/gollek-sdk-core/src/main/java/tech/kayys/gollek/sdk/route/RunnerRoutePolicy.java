package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RunnerRoutePolicyFields;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Normalizes user-facing runner selection before provider/model resolution.
 */
public final class RunnerRoutePolicy {
    public static final String AUTO = RunnerRoutePolicyFields.AUTO;
    public static final String HYBRID = RunnerRoutePolicyFields.HYBRID;
    public static final String SAFETENSOR = RunnerRoutePolicyFields.SAFETENSOR;
    public static final String GGUF = RunnerRoutePolicyFields.GGUF;
    public static final String LITERT = RunnerRoutePolicyFields.LITERT;
    public static final List<String> SUPPORTED_RUNNERS = RunnerRoutePolicyFields.supportedRunners();
    public static final Map<String, List<String>> RUNNER_ALIASES = RunnerRoutePolicyFields.runnerAliases();

    private RunnerRoutePolicy() {
    }

    public static Selection select(
            String requestedRunner,
            String currentProvider,
            String currentFormat,
            boolean providerExplicit,
            boolean preferAlternateRuntime,
            boolean forceGguf) {
        String normalizedRunner = normalizeRunner(requestedRunner);
        if (!SUPPORTED_RUNNERS.contains(normalizedRunner)) {
            return Selection.invalid(
                    normalizedRunner,
                    currentProvider,
                    currentFormat,
                    preferAlternateRuntime,
                    forceGguf,
                    "Unsupported runner '" + requestedRunner + "'. Supported runners: "
                            + String.join(", ", SUPPORTED_RUNNERS) + ".");
        }

        return switch (normalizedRunner) {
            case AUTO -> new Selection(
                    AUTO,
                    false,
                    currentProvider,
                    currentFormat,
                    preferAlternateRuntime,
                    forceGguf,
                    null);
            case HYBRID -> new Selection(
                    HYBRID,
                    true,
                    currentProvider,
                    currentFormat,
                    true,
                    forceGguf,
                    null);
            case SAFETENSOR -> fixedRunner(
                    SAFETENSOR,
                    currentProvider,
                    "safetensor",
                    currentFormat,
                    "safetensors",
                    providerExplicit,
                    preferAlternateRuntime,
                    forceGguf);
            case GGUF -> fixedRunner(
                    GGUF,
                    currentProvider,
                    "gguf",
                    currentFormat,
                    "gguf",
                    providerExplicit,
                    preferAlternateRuntime,
                    true);
            case LITERT -> fixedRunner(
                    LITERT,
                    currentProvider,
                    "litert",
                    currentFormat,
                    "litert",
                    providerExplicit,
                    preferAlternateRuntime,
                    forceGguf);
            default -> throw new IllegalStateException("Unhandled runner: " + normalizedRunner);
        };
    }

    private static Selection fixedRunner(
            String runner,
            String currentProvider,
            String requiredProvider,
            String currentFormat,
            String requiredFormat,
            boolean providerExplicit,
            boolean preferAlternateRuntime,
            boolean forceGguf) {
        if (providerExplicit && !requiredProvider.equalsIgnoreCase(currentProvider)) {
            return Selection.invalid(
                    runner,
                    currentProvider,
                    currentFormat,
                    preferAlternateRuntime,
                    forceGguf,
                    "--runner " + runner + " conflicts with --provider " + currentProvider
                            + ". Use one runner/provider target per invocation.");
        }
        return new Selection(
                runner,
                true,
                requiredProvider,
                requiredFormat,
                preferAlternateRuntime,
                forceGguf,
                null);
    }

    private static String normalizeRunner(String requestedRunner) {
        if (requestedRunner == null || requestedRunner.isBlank()) {
            return AUTO;
        }
        String trimmed = requestedRunner.trim().toLowerCase(Locale.ROOT);
        for (Map.Entry<String, List<String>> aliasGroup : RUNNER_ALIASES.entrySet()) {
            if (aliasGroup.getValue().contains(trimmed)) {
                return aliasGroup.getKey();
            }
        }
        return trimmed;
    }

    public record Selection(
            String runner,
            boolean explicit,
            String providerId,
            String format,
            boolean preferAlternateRuntime,
            boolean forceGguf,
            String error) {
        public static Selection invalid(
                String runner,
                String providerId,
                String format,
                boolean preferAlternateRuntime,
                boolean forceGguf,
                String error) {
            return new Selection(
                    runner,
                    true,
                    providerId,
                    format,
                    preferAlternateRuntime,
                    forceGguf,
                    error);
        }

        public boolean valid() {
            return error == null || error.isBlank();
        }
    }
}
