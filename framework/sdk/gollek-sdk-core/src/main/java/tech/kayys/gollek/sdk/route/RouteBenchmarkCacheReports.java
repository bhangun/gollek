package tech.kayys.gollek.sdk.route;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Builds stable JSON-friendly reports for cached runner route benchmarks.
 */
public final class RouteBenchmarkCacheReports {
    private RouteBenchmarkCacheReports() {
    }

    public static Map<String, Object> summaryReport(int recentEntryLimit) {
        return summaryReport(RunnerRouteBenchmarkCache.entries(), recentEntryLimit);
    }

    public static Map<String, Object> summaryReport(
            List<RunnerRouteBenchmarkCache.Entry> entries,
            int recentEntryLimit) {
        RunnerRouteBenchmarkCache.Health health = RunnerRouteBenchmarkCache.health();
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("schemaVersion", 1);
        report.put("status", health.status());
        report.put("enabled", health.enabled());
        report.put("healthy", "healthy".equals(health.status()) || "empty".equals(health.status()));
        report.put("cacheFile", health.cacheFile().toString());
        report.put("cacheFileExists", health.cacheFileExists());
        report.put("cacheFileReadable", health.cacheFileReadable());
        report.put("cacheDirectoryWritable", health.cacheDirectoryWritable());
        report.put("staleAfterDays", health.staleAfterDays());
        report.put("entryCount", entries.size());
        int staleEntryCount = staleCount(entries, health.staleAfterDays());
        int freshEntryCount = entries.size() - staleEntryCount;
        int trustedEntryCount = RunnerRouteBenchmarkCache.allowStaleProfiles()
                ? entries.size()
                : freshEntryCount;
        report.put("staleEntryCount", staleEntryCount);
        report.put("freshEntryCount", freshEntryCount);
        report.put("trustedEntryCount", trustedEntryCount);
        report.put("staleProfilesAllowed", RunnerRouteBenchmarkCache.allowStaleProfiles());
        report.put("strictHealthy", strictHealthy(health));
        report.put("profileTrustStatus", profileTrustStatus(health, entries, trustedEntryCount));
        report.put("problems", readinessProblems(health, entries, staleEntryCount, trustedEntryCount));
        report.put("remediationHints", remediationHints(health, entries, staleEntryCount, trustedEntryCount));
        report.put("invalidLineCount", health.invalidLineCount());
        report.put("providers", distinct(entries.stream()
                .map(RunnerRouteBenchmarkCache.Entry::provider)
                .toList()));
        report.put("formats", distinct(entries.stream()
                .map(RunnerRouteBenchmarkCache.Entry::format)
                .toList()));
        entries.stream().findFirst().ifPresentOrElse(
                newest -> {
                    report.put("newestUpdatedAtEpochMs", newest.updatedAtEpochMs());
                    report.put("newestUpdatedAt", Instant.ofEpochMilli(newest.updatedAtEpochMs()).toString());
                },
                () -> {
                    report.put("newestUpdatedAtEpochMs", null);
                    report.put("newestUpdatedAt", null);
                });
        int limit = Math.max(0, recentEntryLimit);
        report.put("recentEntries", entries.stream()
                .limit(limit)
                .map(RouteBenchmarkCacheReports::entryReport)
                .toList());
        return report;
    }

    public static Map<String, Object> entryReport(RunnerRouteBenchmarkCache.Entry entry) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("key", entry.key());
        map.put("identity", entry.identity());
        map.put("provider", entry.provider());
        map.put("format", entry.format());
        map.put("generationTokensPerSecond", entry.generationTokensPerSecond());
        map.put("ttftMs", entry.ttftMs());
        map.put("outputTokens", entry.outputTokens());
        map.put("durationMs", entry.durationMs());
        map.put("observations", entry.observations());
        map.put("stale", RunnerRouteBenchmarkCache.isStale(entry));
        map.put("ageDays", RunnerRouteBenchmarkCache.ageDays(entry));
        map.put("updatedAtEpochMs", entry.updatedAtEpochMs());
        map.put("updatedAt", Instant.ofEpochMilli(entry.updatedAtEpochMs()).toString());
        return map;
    }

    public static Map<String, Object> healthReport(RunnerRouteBenchmarkCache.Health health) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("schemaVersion", 1);
        map.put("status", health.status());
        map.put("enabled", health.enabled());
        map.put("healthy", "healthy".equals(health.status()) || "empty".equals(health.status()));
        map.put("cacheFile", health.cacheFile().toString());
        map.put("cacheFileExists", health.cacheFileExists());
        map.put("cacheFileReadable", health.cacheFileReadable());
        map.put("cacheDirectoryWritable", health.cacheDirectoryWritable());
        map.put("entryCount", health.entryCount());
        map.put("staleEntryCount", health.staleEntryCount());
        map.put("invalidLineCount", health.invalidLineCount());
        map.put("staleAfterDays", health.staleAfterDays());
        map.put("newestUpdatedAtEpochMs", health.newestUpdatedAtEpochMs());
        map.put("newestUpdatedAt", health.newestUpdatedAtEpochMs() == null
                ? null
                : Instant.ofEpochMilli(health.newestUpdatedAtEpochMs()).toString());
        return map;
    }

    public static Map<String, Object> pruneResultReport(RunnerRouteBenchmarkCache.PruneResult result) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("schemaVersion", 1);
        map.put("success", result.success());
        map.put("changed", result.changed());
        map.put("status", result.status());
        map.put("cacheFile", result.cacheFile().toString());
        map.put("olderThanDays", result.olderThanDays());
        map.put("beforeCount", result.beforeCount());
        map.put("removedCount", result.removedCount());
        map.put("retainedCount", result.retainedCount());
        map.put("invalidLineCount", result.invalidLineCount());
        return map;
    }

    private static List<String> distinct(List<String> values) {
        Set<String> seen = new LinkedHashSet<>();
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                seen.add(value);
            }
        }
        return List.copyOf(seen);
    }

    private static int staleCount(List<RunnerRouteBenchmarkCache.Entry> entries, int staleAfterDays) {
        long cutoff = Instant.now().minusSeconds(Math.max(1, staleAfterDays) * 86_400L).toEpochMilli();
        return (int) entries.stream()
                .filter(entry -> entry.updatedAtEpochMs() < cutoff)
                .count();
    }

    private static boolean strictHealthy(RunnerRouteBenchmarkCache.Health health) {
        return "healthy".equals(health.status()) || "empty".equals(health.status());
    }

    private static String profileTrustStatus(
            RunnerRouteBenchmarkCache.Health health,
            List<RunnerRouteBenchmarkCache.Entry> entries,
            int trustedEntryCount) {
        if (!health.enabled()) {
            return "disabled";
        }
        if (!health.cacheFileReadable()) {
            return "unreadable";
        }
        if (health.invalidLineCount() > 0) {
            return "invalid";
        }
        if (!health.cacheDirectoryWritable()) {
            return "unwritable";
        }
        if (entries.isEmpty()) {
            return "empty";
        }
        if (trustedEntryCount > 0) {
            return "trusted_profiles_available";
        }
        return "stale_profiles_only";
    }

    private static List<String> readinessProblems(
            RunnerRouteBenchmarkCache.Health health,
            List<RunnerRouteBenchmarkCache.Entry> entries,
            int staleEntryCount,
            int trustedEntryCount) {
        Set<String> problems = new LinkedHashSet<>();
        if (!health.enabled()) {
            problems.add("cache_disabled");
        }
        if (!health.cacheFileReadable()) {
            problems.add("cache_unreadable");
        }
        if (!health.cacheDirectoryWritable()) {
            problems.add("cache_directory_unwritable");
        }
        if (health.invalidLineCount() > 0) {
            problems.add("invalid_cache_lines");
        }
        if (staleEntryCount > 0 && !RunnerRouteBenchmarkCache.allowStaleProfiles()) {
            problems.add("stale_entries_ignored_for_route_profiles");
        }
        if (!entries.isEmpty() && trustedEntryCount == 0) {
            problems.add("no_trusted_route_profiles");
        }
        return List.copyOf(problems);
    }

    private static List<String> remediationHints(
            RunnerRouteBenchmarkCache.Health health,
            List<RunnerRouteBenchmarkCache.Entry> entries,
            int staleEntryCount,
            int trustedEntryCount) {
        Set<String> hints = new LinkedHashSet<>();
        if (!health.enabled()) {
            hints.add("Enable route benchmark cache with -Dgollek.cli.route_benchmark_cache_enabled=true.");
        }
        if (!health.cacheFileReadable()) {
            hints.add("Fix route benchmark cache file permissions or clear the cache with gollek route-benchmarks clear --yes.");
        }
        if (!health.cacheDirectoryWritable()) {
            hints.add("Set -Dgollek.cli.route_benchmark_cache_dir to a writable deployment cache directory.");
        }
        if (health.invalidLineCount() > 0 || staleEntryCount > 0) {
            hints.add("Inspect stale or invalid measurements with gollek route-benchmarks doctor --strict, then prune with gollek route-benchmarks prune --dry-run --older-than-days "
                    + health.staleAfterDays() + ".");
        }
        if (!entries.isEmpty() && trustedEntryCount == 0) {
            hints.add("Run a fresh local generation for the target provider/format before relying on benchmark-cache route profiles.");
        }
        return List.copyOf(hints);
    }
}
