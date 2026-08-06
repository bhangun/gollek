package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RunnerRouteReportFields;

import java.util.Locale;

/**
 * Lightweight route-performance hint that can later be backed by measured benchmark cache entries.
 */
public record RunnerRoutePerformanceProfile(
        String status,
        String source,
        String provider,
        String format,
        String reason,
        String advice) {

    public static RunnerRoutePerformanceProfile unavailable() {
        return new RunnerRoutePerformanceProfile(
                RunnerRouteReportFields.RouteProfileStatus.UNAVAILABLE,
                RunnerRouteReportFields.RouteProfileSource.NONE,
                null,
                null,
                null,
                null);
    }

    public static RunnerRoutePerformanceProfile from(RunnerRouteReport report) {
        if (report == null) {
            return unavailable();
        }
        if (report.runtimeRedirected()) {
            return redirected(report);
        }
        String provider = normalize(report.effectiveProvider());
        String format = normalize(report.effectiveFormat());
        if ("gguf".equals(provider)) {
            return selected(
                    provider,
                    textOrDefault(format, "gguf"),
                    "GGUF route selected.",
                    "Keep --runner gguf or --provider gguf for production bundles that should avoid direct safetensor loading.");
        }
        if ("litert".equals(provider)) {
            return selected(
                    provider,
                    textOrDefault(format, "litert"),
                    "LiteRT route selected.",
                    "Keep --runner litert or --provider litert for edge/mobile bundles and detachable production runtimes.");
        }
        if ("safetensor".equals(provider) || "safetensors".equals(format)) {
            return new RunnerRoutePerformanceProfile(
                    RunnerRouteReportFields.RouteProfileStatus.CANDIDATE,
                    RunnerRouteReportFields.RouteProfileSource.HEURISTIC,
                    "gguf",
                    "gguf",
                    "Safetensor direct route selected without a faster alternate runtime artifact.",
                    "For faster production inference, attach or pull a GGUF/LiteRT artifact and let auto or hybrid routing redirect.");
        }
        return unavailable();
    }

    public static RunnerRoutePerformanceProfile fromBenchmarkCache(
            RunnerRouteReport report,
            RunnerRouteBenchmarkCache.Entry entry) {
        if (entry == null) {
            return unavailable();
        }
        String status = report != null && report.runtimeRedirected()
                ? RunnerRouteReportFields.RouteProfileStatus.REDIRECTED
                : RunnerRouteReportFields.RouteProfileStatus.SELECTED;
        String provider = normalize(entry.provider());
        String format = normalize(entry.format());
        String speed = String.format(Locale.ROOT, "%.2f tok/s", entry.generationTokensPerSecond());
        String ttft = entry.ttftMs() == null || entry.ttftMs() <= 0.0
                ? ""
                : String.format(Locale.ROOT, ", TTFT %.0f ms", entry.ttftMs());
        String reason = "Cached local benchmark for " + provider + "/" + format
                + ": " + speed + ttft + " across " + entry.observations() + " observation"
                + (entry.observations() == 1 ? "." : "s.");
        String advice = "Use this measured route profile when comparing production runners; "
                + "future auto routing can consume the same benchmark cache.";
        return new RunnerRoutePerformanceProfile(
                status,
                RunnerRouteReportFields.RouteProfileSource.BENCHMARK_CACHE,
                provider,
                format,
                reason,
                advice);
    }

    private static RunnerRoutePerformanceProfile redirected(RunnerRouteReport report) {
        String provider = tokenOrDefault(report.runtimeRedirectToProvider(), report.effectiveProvider());
        String format = tokenOrDefault(report.runtimeRedirectToFormat(), report.effectiveFormat());
        String source = report.runtimeRedirectCacheHit()
                ? RunnerRouteReportFields.RouteProfileSource.ARTIFACT_CACHE
                : RunnerRouteReportFields.RouteProfileSource.RUNTIME_REDIRECT;
        String reason = textOrDefault(
                report.runtimeRedirectReason(),
                "Runtime route redirected to a preferred alternate artifact.");
        String advice = report.runtimeRedirectCacheHit()
                ? "Continue using this route for production inference; cached alternate artifacts avoid repeated discovery cost."
                : "Route redirected to a production-oriented runtime artifact; a cache hit should appear on later matching runs.";
        return new RunnerRoutePerformanceProfile(
                RunnerRouteReportFields.RouteProfileStatus.REDIRECTED,
                source,
                normalize(provider),
                normalize(format),
                reason,
                advice);
    }

    private static RunnerRoutePerformanceProfile selected(
            String provider,
            String format,
            String reason,
            String advice) {
        return new RunnerRoutePerformanceProfile(
                RunnerRouteReportFields.RouteProfileStatus.SELECTED,
                RunnerRouteReportFields.RouteProfileSource.SELECTED_ROUTE,
                normalize(provider),
                normalize(format),
                reason,
                advice);
    }

    private static String normalize(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return value.trim().toLowerCase(Locale.ROOT);
    }

    private static String tokenOrDefault(String value, String fallback) {
        String text = normalize(value);
        return text != null ? text : normalize(fallback);
    }

    private static String textOrDefault(String value, String fallback) {
        if (value != null && !value.isBlank()) {
            return value.trim();
        }
        return fallback == null || fallback.isBlank() ? null : fallback.trim();
    }
}
