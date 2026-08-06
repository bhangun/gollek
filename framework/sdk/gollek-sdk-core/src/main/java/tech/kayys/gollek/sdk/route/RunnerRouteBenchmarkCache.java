package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RunnerRouteReportFields;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Small TSV-backed cache for observed local route throughput.
 */
public final class RunnerRouteBenchmarkCache {
    public static final String CACHE_DIR_PROPERTY = "gollek.cli.route_benchmark_cache_dir";
    public static final String CACHE_ENABLED_PROPERTY = "gollek.cli.route_benchmark_cache_enabled";
    public static final String CACHE_STALE_DAYS_PROPERTY = "gollek.cli.route_benchmark_cache_stale_days";
    public static final String CACHE_ALLOW_STALE_PROPERTY = "gollek.cli.route_benchmark_cache_allow_stale";
    public static final String CACHE_FILE = "route-benchmark-profiles.tsv";
    public static final int DEFAULT_STALE_DAYS = 30;

    private RunnerRouteBenchmarkCache() {
    }

    public static Optional<RunnerRoutePerformanceProfile> profileFor(
            RunnerRouteReport report,
            String requestedModel,
            String effectiveModel,
            String localPath) {
        if (report == null) {
            return Optional.empty();
        }
        String provider = textOrDefault(report.effectiveProvider(), report.runtimeRedirectToProvider());
        String format = textOrDefault(report.effectiveFormat(), report.runtimeRedirectToFormat());
        return find(requestedModel, effectiveModel, localPath, provider, format)
                .filter(RunnerRouteBenchmarkCache::usableForRouteProfile)
                .map(entry -> RunnerRoutePerformanceProfile.fromBenchmarkCache(report, entry));
    }

    public static Optional<Entry> find(
            String requestedModel,
            String effectiveModel,
            String localPath,
            String provider,
            String format) {
        if (!enabled() || blank(provider) || blank(format)) {
            return Optional.empty();
        }
        String key = cacheKey(requestedModel, effectiveModel, localPath, provider, format);
        Path cacheFile = cacheFile();
        if (!Files.isRegularFile(cacheFile)) {
            return Optional.empty();
        }
        try {
            for (String line : Files.readAllLines(cacheFile, StandardCharsets.UTF_8)) {
                Optional<Entry> parsed = Entry.parse(line);
                if (parsed.isPresent() && key.equals(parsed.get().key())) {
                    return parsed;
                }
            }
        } catch (Exception ignored) {
            return Optional.empty();
        }
        return Optional.empty();
    }

    public static List<Entry> entries() {
        if (!enabled()) {
            return List.of();
        }
        return snapshot(cacheFile()).entries();
    }

    public static Path cacheFilePath() {
        return cacheFile();
    }

    public static boolean clear() {
        try {
            return Files.deleteIfExists(cacheFile());
        } catch (Exception ignored) {
            return false;
        }
    }

    public static boolean isEnabled() {
        return enabled();
    }

    public static boolean allowStaleProfiles() {
        return Boolean.parseBoolean(System.getProperty(CACHE_ALLOW_STALE_PROPERTY, "false"));
    }

    public static int staleAfterDays() {
        String configured = System.getProperty(CACHE_STALE_DAYS_PROPERTY);
        if (configured == null || configured.isBlank()) {
            return DEFAULT_STALE_DAYS;
        }
        try {
            return Math.max(1, Integer.parseInt(configured.trim()));
        } catch (NumberFormatException ignored) {
            return DEFAULT_STALE_DAYS;
        }
    }

    public static Health health() {
        return health(staleAfterDays());
    }

    public static Health health(int staleAfterDays) {
        Path cacheFile = cacheFile();
        boolean enabled = enabled();
        boolean exists = Files.exists(cacheFile);
        boolean readable = Files.isReadable(cacheFile) || !exists;
        boolean writableDirectory = cacheDirectoryWritable(cacheFile);
        if (!enabled) {
            return Health.from(
                    enabled,
                    cacheFile,
                    exists,
                    readable,
                    writableDirectory,
                    List.of(),
                    0,
                    normalizedDays(staleAfterDays),
                    "disabled");
        }
        if (!exists) {
            return Health.from(
                    enabled,
                    cacheFile,
                    false,
                    true,
                    writableDirectory,
                    List.of(),
                    0,
                    normalizedDays(staleAfterDays),
                    writableDirectory ? "empty" : "unwritable");
        }
        CacheSnapshot snapshot = snapshot(cacheFile);
        if (snapshot.readFailed()) {
            return Health.from(
                    enabled,
                    cacheFile,
                    true,
                    false,
                    writableDirectory,
                    List.of(),
                    snapshot.invalidLineCount(),
                    normalizedDays(staleAfterDays),
                    "unreadable");
        }
        int normalizedDays = normalizedDays(staleAfterDays);
        long cutoff = staleCutoffEpochMs(normalizedDays);
        int staleCount = (int) snapshot.entries().stream()
                .filter(entry -> entry.updatedAtEpochMs() < cutoff)
                .count();
        String status = snapshot.invalidLineCount() > 0
                ? "invalid"
                : staleCount > 0
                        ? "stale"
                        : writableDirectory ? "healthy" : "unwritable";
        return Health.from(
                enabled,
                cacheFile,
                true,
                readable,
                writableDirectory,
                snapshot.entries(),
                snapshot.invalidLineCount(),
                normalizedDays,
                status);
    }

    public static PruneResult pruneOlderThanDays(int olderThanDays) {
        int normalizedDays = normalizedDays(olderThanDays);
        Path cacheFile = cacheFile();
        if (!enabled()) {
            return PruneResult.empty(cacheFile, normalizedDays, "disabled");
        }
        if (!Files.exists(cacheFile)) {
            return PruneResult.empty(cacheFile, normalizedDays, "not_found");
        }
        CacheSnapshot snapshot = snapshot(cacheFile);
        if (snapshot.readFailed()) {
            return new PruneResult(
                    false,
                    false,
                    "unreadable",
                    cacheFile,
                    normalizedDays,
                    0,
                    0,
                    0,
                    snapshot.invalidLineCount());
        }
        long cutoff = staleCutoffEpochMs(normalizedDays);
        List<Entry> retained = snapshot.entries().stream()
                .filter(entry -> entry.updatedAtEpochMs() >= cutoff)
                .sorted(Comparator.comparingLong(Entry::updatedAtEpochMs).reversed())
                .toList();
        int removed = snapshot.entries().size() - retained.size();
        boolean changed = removed > 0 || snapshot.invalidLineCount() > 0;
        if (changed) {
            try {
                if (retained.isEmpty()) {
                    Files.deleteIfExists(cacheFile);
                } else {
                    Files.write(
                            cacheFile,
                            retained.stream().map(Entry::serialize).toList(),
                            StandardCharsets.UTF_8);
                }
            } catch (Exception ignored) {
                return new PruneResult(
                        false,
                        false,
                        "write_failed",
                        cacheFile,
                        normalizedDays,
                        snapshot.entries().size(),
                        removed,
                        retained.size(),
                        snapshot.invalidLineCount());
            }
        }
        return new PruneResult(
                true,
                changed,
                changed ? "pruned" : "unchanged",
                cacheFile,
                normalizedDays,
                snapshot.entries().size(),
                removed,
                retained.size(),
                snapshot.invalidLineCount());
    }

    public static void record(
            String requestedModel,
            String effectiveModel,
            String localPath,
            Map<String, Object> metrics,
            int outputTokens,
            long durationMs) {
        if (!enabled() || metrics == null || metrics.isEmpty() || outputTokens <= 0 || durationMs <= 0L) {
            return;
        }
        String provider = routeMetadata(metrics, RunnerRouteReportFields.Report.EFFECTIVE_PROVIDER);
        String format = routeMetadata(metrics, RunnerRouteReportFields.Report.EFFECTIVE_FORMAT);
        if (blank(provider) || blank(format)) {
            return;
        }
        Double observedTps = metricDouble(metrics, "bench.generation_tps");
        if (observedTps == null || observedTps <= 0.0) {
            observedTps = outputTokens / Math.max(durationMs / 1000.0, 0.001);
        }
        double normalizedObservedTps = observedTps;
        Double ttftMs = metricDouble(metrics, "bench.ttft_ms");
        String key = cacheKey(requestedModel, effectiveModel, localPath, provider, format);
        String identity = identity(requestedModel, effectiveModel, localPath);
        Path cacheFile = cacheFile();
        try {
            Files.createDirectories(cacheFile.getParent());
            List<String> lines = Files.isRegularFile(cacheFile)
                    ? Files.readAllLines(cacheFile, StandardCharsets.UTF_8)
                    : List.of();
            Optional<Entry> existing = lines.stream()
                    .map(Entry::parse)
                    .flatMap(Optional::stream)
                    .filter(entry -> key.equals(entry.key()))
                    .findFirst();
            Entry next = existing
                    .map(entry -> entry.merge(normalizedObservedTps, ttftMs, outputTokens, durationMs))
                    .orElseGet(() -> Entry.create(
                            key,
                            identity,
                            normalize(provider),
                            normalize(format),
                            normalizedObservedTps,
                            ttftMs,
                            outputTokens,
                            durationMs));
            List<String> updated = new ArrayList<>(lines.size() + 1);
            for (String line : lines) {
                if (!line.startsWith(key + "\t")) {
                    updated.add(line);
                }
            }
            updated.add(next.serialize());
            Files.write(cacheFile, updated, StandardCharsets.UTF_8);
        } catch (Exception ignored) {
            // Benchmark cache failures must never affect inference.
        }
    }

    private static String routeMetadata(Map<String, Object> metadata, String field) {
        Object flattened = metadata.get(RunnerRouteReportFields.metadataKey(field));
        if (flattened != null) {
            return String.valueOf(flattened);
        }
        Object route = metadata.get(RunnerRouteReportFields.METADATA_ROOT);
        if (route instanceof Map<?, ?> routeMap) {
            Object value = routeMap.get(field);
            if (value != null) {
                return String.valueOf(value);
            }
        }
        return null;
    }

    private static String cacheKey(
            String requestedModel,
            String effectiveModel,
            String localPath,
            String provider,
            String format) {
        String payload = String.join("\n",
                identity(requestedModel, effectiveModel, localPath),
                normalize(provider),
                normalize(format));
        return "sha256:" + sha256(payload);
    }

    private static String identity(String requestedModel, String effectiveModel, String localPath) {
        String candidate = textOrDefault(localPath, textOrDefault(effectiveModel, requestedModel));
        if (blank(candidate)) {
            return "unknown";
        }
        try {
            Path path = Path.of(candidate);
            if (path.isAbsolute() || candidate.contains("/") || candidate.contains("\\")) {
                return path.toAbsolutePath().normalize().toString();
            }
        } catch (Exception ignored) {
            // Keep raw non-path model ids.
        }
        return candidate.trim();
    }

    private static Path cacheFile() {
        String configured = System.getProperty(CACHE_DIR_PROPERTY);
        Path dir = configured == null || configured.isBlank()
                ? Path.of(System.getProperty("user.home"), ".gollek", "cache")
                : Path.of(configured.trim());
        return dir.resolve(CACHE_FILE);
    }

    private static boolean enabled() {
        String configured = System.getProperty(CACHE_ENABLED_PROPERTY);
        return configured == null || configured.isBlank() || Boolean.parseBoolean(configured);
    }

    private static boolean usableForRouteProfile(Entry entry) {
        return allowStaleProfiles() || !isStale(entry, staleAfterDays());
    }

    public static boolean isStale(Entry entry) {
        return isStale(entry, staleAfterDays());
    }

    public static boolean isStale(Entry entry, int staleAfterDays) {
        return entry != null && entry.updatedAtEpochMs() < staleCutoffEpochMs(staleAfterDays);
    }

    public static long ageDays(Entry entry) {
        if (entry == null) {
            return 0L;
        }
        long ageMs = Math.max(0L, Instant.now().toEpochMilli() - entry.updatedAtEpochMs());
        return ageMs / 86_400_000L;
    }

    private static int normalizedDays(int days) {
        return Math.max(1, days);
    }

    private static long staleCutoffEpochMs(int olderThanDays) {
        return Instant.now().minusSeconds(normalizedDays(olderThanDays) * 86_400L).toEpochMilli();
    }

    private static boolean cacheDirectoryWritable(Path cacheFile) {
        Path parent = cacheFile.getParent();
        if (parent == null) {
            return false;
        }
        if (Files.exists(parent)) {
            return Files.isDirectory(parent) && Files.isWritable(parent);
        }
        Path ancestor = parent.getParent();
        while (ancestor != null && !Files.exists(ancestor)) {
            ancestor = ancestor.getParent();
        }
        return ancestor != null && Files.isDirectory(ancestor) && Files.isWritable(ancestor);
    }

    private static CacheSnapshot snapshot(Path cacheFile) {
        if (!Files.isRegularFile(cacheFile)) {
            return new CacheSnapshot(List.of(), 0, false);
        }
        try {
            List<Entry> entries = new ArrayList<>();
            int invalidLineCount = 0;
            for (String line : Files.readAllLines(cacheFile, StandardCharsets.UTF_8)) {
                Optional<Entry> parsed = Entry.parse(line);
                if (parsed.isPresent()) {
                    entries.add(parsed.get());
                } else if (line != null && !line.isBlank()) {
                    invalidLineCount++;
                }
            }
            return new CacheSnapshot(
                    entries.stream()
                            .sorted(Comparator.comparingLong(Entry::updatedAtEpochMs).reversed())
                            .toList(),
                    invalidLineCount,
                    false);
        } catch (Exception ignored) {
            return new CacheSnapshot(List.of(), 0, true);
        }
    }

    private static Double metricDouble(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof String text && !text.isBlank()) {
            try {
                return Double.parseDouble(text.trim());
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private static String textOrDefault(String value, String fallback) {
        return blank(value) ? fallback : value.trim();
    }

    private static boolean blank(String value) {
        return value == null || value.isBlank();
    }

    private static String sanitize(String value) {
        return value == null
                ? ""
                : value.replace('\t', ' ')
                        .replace('\n', ' ')
                        .replace('\r', ' ')
                        .trim();
    }

    private static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder(digest.length * 2);
            for (byte raw : digest) {
                int valueByte = raw & 0xff;
                if (valueByte < 16) {
                    hex.append('0');
                }
                hex.append(Integer.toHexString(valueByte));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is required for route benchmark cache keys.", error);
        }
    }

    public record Entry(
            String key,
            String identity,
            String provider,
            String format,
            double generationTokensPerSecond,
            Double ttftMs,
            int outputTokens,
            long durationMs,
            int observations,
            long updatedAtEpochMs) {

        public static Entry create(
                String key,
                String identity,
                String provider,
                String format,
                double generationTokensPerSecond,
                Double ttftMs,
                int outputTokens,
                long durationMs) {
            return new Entry(
                    key,
                    sanitize(identity),
                    normalize(provider),
                    normalize(format),
                    generationTokensPerSecond,
                    ttftMs,
                    outputTokens,
                    durationMs,
                    1,
                    Instant.now().toEpochMilli());
        }

        public Entry merge(double observedTps, Double observedTtftMs, int latestOutputTokens, long latestDurationMs) {
            int nextObservations = Math.max(1, observations + 1);
            double nextTps = weighted(generationTokensPerSecond, observations, observedTps);
            Double nextTtft = mergeOptional(ttftMs, observedTtftMs, observations);
            return new Entry(
                    key,
                    identity,
                    provider,
                    format,
                    nextTps,
                    nextTtft,
                    latestOutputTokens,
                    latestDurationMs,
                    nextObservations,
                    Instant.now().toEpochMilli());
        }

        public String serialize() {
            return String.join("\t",
                    sanitize(key),
                    sanitize(identity),
                    sanitize(provider),
                    sanitize(format),
                    Double.toString(generationTokensPerSecond),
                    ttftMs == null ? "" : Double.toString(ttftMs),
                    Integer.toString(outputTokens),
                    Long.toString(durationMs),
                    Integer.toString(observations),
                    Long.toString(updatedAtEpochMs));
        }

        public static Optional<Entry> parse(String line) {
            if (line == null || line.isBlank()) {
                return Optional.empty();
            }
            String[] parts = line.split("\t", -1);
            if (parts.length != 10) {
                return Optional.empty();
            }
            try {
                return Optional.of(new Entry(
                        parts[0],
                        parts[1],
                        parts[2],
                        parts[3],
                        Double.parseDouble(parts[4]),
                        parts[5].isBlank() ? null : Double.parseDouble(parts[5]),
                        Integer.parseInt(parts[6]),
                        Long.parseLong(parts[7]),
                        Integer.parseInt(parts[8]),
                        Long.parseLong(parts[9])));
            } catch (Exception ignored) {
                return Optional.empty();
            }
        }

        private static double weighted(double previous, int previousCount, double observed) {
            int count = Math.max(1, previousCount);
            return ((previous * count) + observed) / (count + 1);
        }

        private static Double mergeOptional(Double previous, Double observed, int previousCount) {
            if (observed == null || observed <= 0.0) {
                return previous;
            }
            if (previous == null || previous <= 0.0) {
                return observed;
            }
            return weighted(previous, previousCount, observed);
        }
    }

    public record Health(
            boolean enabled,
            Path cacheFile,
            boolean cacheFileExists,
            boolean cacheFileReadable,
            boolean cacheDirectoryWritable,
            int entryCount,
            int staleEntryCount,
            int invalidLineCount,
            int staleAfterDays,
            Long newestUpdatedAtEpochMs,
            String status) {
        public static Health from(
                boolean enabled,
                Path cacheFile,
                boolean cacheFileExists,
                boolean cacheFileReadable,
                boolean cacheDirectoryWritable,
                List<Entry> entries,
                int invalidLineCount,
                int staleAfterDays,
                String status) {
            long cutoff = staleCutoffEpochMs(staleAfterDays);
            int staleCount = (int) entries.stream()
                    .filter(entry -> entry.updatedAtEpochMs() < cutoff)
                    .count();
            Long newest = entries.stream()
                    .findFirst()
                    .map(Entry::updatedAtEpochMs)
                    .orElse(null);
            return new Health(
                    enabled,
                    cacheFile,
                    cacheFileExists,
                    cacheFileReadable,
                    cacheDirectoryWritable,
                    entries.size(),
                    staleCount,
                    invalidLineCount,
                    staleAfterDays,
                    newest,
                    status);
        }
    }

    public record PruneResult(
            boolean success,
            boolean changed,
            String status,
            Path cacheFile,
            int olderThanDays,
            int beforeCount,
            int removedCount,
            int retainedCount,
            int invalidLineCount) {
        public static PruneResult empty(Path cacheFile, int olderThanDays, String status) {
            return new PruneResult(true, false, status, cacheFile, olderThanDays, 0, 0, 0, 0);
        }
    }

    private record CacheSnapshot(
            List<Entry> entries,
            int invalidLineCount,
            boolean readFailed) {
    }
}
