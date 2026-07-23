package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;
import tech.kayys.gollek.observability.ObservabilityManager;
import tech.kayys.gollek.metrics.MetricsRegistry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;

import com.google.ai.edge.litertlm.Backend;
import com.google.ai.edge.litertlm.Conversation;
import com.google.ai.edge.litertlm.ConversationConfig;
import com.google.ai.edge.litertlm.Contents;
import com.google.ai.edge.litertlm.Engine;
import com.google.ai.edge.litertlm.EngineConfig;
import com.google.ai.edge.litertlm.ExperimentalFlags;
import com.google.ai.edge.litertlm.LogSeverity;
import com.google.ai.edge.litertlm.MessageCallback;
import com.google.ai.edge.litertlm.SamplerConfig;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.ConnectException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tiny LiteRT-LM text runner used by the local macOS shim for simple
 * {@code gollek run} calls.
 *
 * <p>The full Quarkus CLI is still the source of truth for general commands.
 * This entrypoint intentionally supports only local text LiteRT-LM inference so
 * short prompts do not pay the whole CLI/SDK startup tax.</p>
 */
public final class LiteRtLmFastRun {
    private static final int FALLBACK_TO_FULL_CLI = 42;
    private static final String DAEMON_MAGIC = "GOLLEK_LITERT_FAST_DAEMON_V1";
    private static final String DAEMON_PREWARM_MAGIC = "GOLLEK_LITERT_FAST_DAEMON_PREWARM_V1";
    private static final String DAEMON_STOP_MAGIC = "GOLLEK_LITERT_FAST_DAEMON_STOP_V1";
    private static final int DEFAULT_MAX_NUM_TOKENS = 2048;
    private static final int DEFAULT_MAX_NUM_TOKENS_CEILING = 2048;
    private static final int DEFAULT_MAX_NUM_TOKENS_FLOOR = 256;
    private static final Pattern JSON_OBJECT_PATTERN = Pattern.compile("\\{.*?}", Pattern.DOTALL);
    private static final Pattern BARE_QUESTION_PATTERN = Pattern.compile(
            "(?i)^(who|what|where|when|why|how|which|whom|whose|is|are|was|were|do|does|did|can|could|should|would|will|may|might)\\b.*[^.?!]$");
    private static final String SHORT_QUESTION_INSTRUCTION =
            "Answer directly and concisely in one or two sentences. Do not ask a clarification question. "
                    + "If a term is ambiguous, answer the most common meaning first and mention common alternatives briefly.\n"
                    + "Question: ";

    private LiteRtLmFastRun() {
    }

    public static void main(String[] args) {
        if (args.length > 0 && "__daemon".equals(args[0])) {
            System.exit(runDaemon());
            return;
        }
        if (args.length > 0 && "__daemon-stop".equals(args[0])) {
            System.exit(stopDaemon());
            return;
        }
        if (args.length > 0 && isPrewarmCommand(args[0])) {
            System.exit(prewarm(args));
            return;
        }
        int status = run(args);
        if (status != 0) {
            System.exit(status);
        }
    }

    static int run(String[] args) {
        try {
            FastArgs parsed = FastArgs.parse(args);
            if (!parsed.supported()) {
                return FALLBACK_TO_FULL_CLI;
            }
            if (daemonEnabled()) {
                OptionalInt daemonStatus = requestExistingDaemon(args);
                if (daemonStatus.isPresent()) {
                    return daemonStatus.getAsInt();
                }
            }
            Optional<RouteResolution> route = resolveLiteRtLmRoute(parsed);
            if (route.isEmpty()) {
                return FALLBACK_TO_FULL_CLI;
            }
            if (daemonEnabled()) {
                OptionalInt daemonStatus = requestDaemon(args);
                if (daemonStatus.isPresent()) {
                    return daemonStatus.getAsInt();
                }
            }
            generate(route.get(), parsed);
            return 0;
        } catch (Throwable throwable) {
            if (Boolean.getBoolean("gollek.litert.fast_run.debug")) {
                throwable.printStackTrace(System.err);
            }
            return FALLBACK_TO_FULL_CLI;
        }
    }

    static int prewarm(String[] args) {
        try {
            if (isHelpRequest(args)) {
                printPrewarmUsage(System.out);
                return 0;
            }
            String[] runArgs = prewarmRunArgs(args);
            FastArgs parsed = FastArgs.parse(runArgs);
            if (!parsed.supported()) {
                printPrewarmUsage(System.err);
                return 2;
            }
            if (resolveLiteRtLmRoute(parsed).isEmpty()) {
                System.err.println("LiteRT-LM daemon prewarm could not resolve a .litertlm model.");
                return 2;
            }
            OptionalInt status = requestDaemonPrewarm(runArgs);
            return status.orElse(2);
        } catch (Throwable throwable) {
            System.err.println("LiteRT-LM daemon prewarm failed: " + summarize(throwable));
            if (Boolean.getBoolean("gollek.litert.fast_run.debug")) {
                throwable.printStackTrace(System.err);
            }
            return 2;
        }
    }

    private static void printPrewarmUsage(PrintStream out) {
        out.println("Usage: gollek prewarm --provider litert --model MODEL [--backend metal|cpu]");
        out.println("       gollek warmup  --provider litert --model MODEL [--backend metal|cpu]");
        out.println();
        out.println("Preloads the official LiteRT-LM daemon and opens the selected Metal/CPU engine");
        out.println("so the next matching gollek run can skip cold engine initialization.");
    }

    private static boolean isHelpRequest(String[] args) {
        for (String arg : args) {
            if ("--help".equals(arg) || "-h".equals(arg)) {
                return true;
            }
        }
        return false;
    }

    static String[] prewarmRunArgs(String[] args) {
        List<String> normalized = new ArrayList<>();
        int startIndex = 0;
        if (args.length > 0 && ("run".equals(args[0]) || isPrewarmCommand(args[0]))) {
            startIndex = 1;
        }
        normalized.add("run");
        boolean hasPrompt = false;
        boolean hasMaxTokens = false;
        boolean hasProvider = false;
        for (int i = startIndex; i < args.length; i++) {
            String arg = args[i];
            normalized.add(arg);
            String option = arg;
            int eq = arg.indexOf('=');
            if (arg.startsWith("--") && eq > 0) {
                option = arg.substring(0, eq);
            }
            if (option.equals("--prompt") || option.equals("-p")) {
                hasPrompt = true;
            } else if (option.equals("--max-tokens")) {
                hasMaxTokens = true;
            } else if (option.equals("--provider")) {
                hasProvider = true;
            }
        }
        if (!hasPrompt) {
            normalized.add("--prompt");
            normalized.add(firstNonBlank(
                    System.getProperty("gollek.litert.fast_run.prewarm_prompt"),
                    System.getenv("GOLLEK_LITERT_FAST_PREWARM_PROMPT"),
                    "where is jakarta"));
        }
        if (!hasMaxTokens) {
            normalized.add("--max-tokens");
            normalized.add(firstNonBlank(
                    System.getProperty("gollek.litert.fast_run.prewarm_context_tokens"),
                    System.getenv("GOLLEK_LITERT_FAST_PREWARM_CONTEXT_TOKENS"),
                    "10"));
        }
        if (!hasProvider) {
            normalized.add("--provider");
            normalized.add("litert");
        }
        return normalized.toArray(String[]::new);
    }

    private static boolean isPrewarmCommand(String arg) {
        return "__litert-daemon-prewarm".equals(arg)
                || "prewarm".equals(arg)
                || "warmup".equals(arg);
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return "";
    }

    static Optional<Path> resolveLiteRtLmModel(FastArgs args) {
        return resolveLiteRtLmRoute(args).map(RouteResolution::path);
    }

    static Optional<RouteResolution> resolveLiteRtLmRoute(FastArgs args) {
        long startNanos = System.nanoTime();
        Optional<RouteMatch> match = resolveLiteRtLmRouteMatch(args);
        long durationNanos = System.nanoTime() - startNanos;
        return match.map(found -> new RouteResolution(
                found.path(),
                found.source(),
                found.cacheHit(),
                liteRtModelFormat(found.path()),
                durationNanos));
    }

    private static Optional<RouteMatch> resolveLiteRtLmRouteMatch(FastArgs args) {
        if (args.modelFile != null) {
            Optional<Path> explicit = resolvePath(Path.of(args.modelFile));
            if (explicit.isPresent()) {
                return findLiteRtLmFromExplicitPath(explicit.get())
                        .map(path -> new RouteMatch(path, "modelFile", false));
            }
        }
        if (args.model != null) {
            Optional<Path> direct = resolvePath(Path.of(args.model));
            if (direct.isPresent()) {
                Optional<Path> found = findLiteRtLmFromExplicitPath(direct.get());
                if (found.isPresent()) {
                    return Optional.of(new RouteMatch(found.get(), "model", false));
                }
            }
            Optional<RouteMatch> fastIndexed = findIndexedLiteRtModelRoute(args.model);
            if (fastIndexed.isPresent()) {
                Optional<Path> found = findLiteRtLmFromExplicitPath(fastIndexed.get().path());
                if (found.isPresent()) {
                    return Optional.of(new RouteMatch(
                            found.get(),
                            fastIndexed.get().source(),
                            fastIndexed.get().cacheHit()));
                }
            }
            try {
                Optional<LocalModelIndex.Entry> indexed = LocalModelIndex.find(args.model);
                if (indexed.isPresent()
                        && isLiteRtIndexEntry(indexed.get())
                        && indexed.get().path != null
                        && !indexed.get().path.isBlank()) {
                    return findLiteRtLmFromExplicitPath(Path.of(indexed.get().path))
                            .map(path -> new RouteMatch(path, "local-index-litert", true));
                }
                if (indexed.isPresent()
                        && isSafetensorIndexEntry(indexed.get())
                        && indexed.get().path != null
                        && !indexed.get().path.isBlank()) {
                    Optional<Path> cachedGemma4LiteRt = findCachedGemma4LiteRtModelPath(indexed.get().path);
                    if (cachedGemma4LiteRt.isPresent()) {
                        return Optional.of(new RouteMatch(
                                cachedGemma4LiteRt.get(),
                                "cached-gemma4-litert",
                                true));
                    }
                }
            } catch (Throwable ignored) {
                // Full CLI has the complete resolver if the lightweight index read fails.
            }
        }
        return Optional.empty();
    }

    private static Optional<Path> resolvePath(Path path) {
        try {
            Path normalized = path.isAbsolute()
                    ? path.toAbsolutePath().normalize()
                    : Path.of("").toAbsolutePath().resolve(path).normalize();
            if (Files.exists(normalized)) {
                return Optional.of(normalized);
            }
        } catch (Exception ignored) {
        }
        return Optional.empty();
    }

    static Optional<Path> findIndexedLiteRtModelPath(String ref) {
        return findIndexedLiteRtModelRoute(ref).map(RouteMatch::path);
    }

    static Optional<RouteMatch> findIndexedLiteRtModelRoute(String ref) {
        if (ref == null || ref.isBlank()) {
            return Optional.empty();
        }
        Path indexPath = modelIndexPath();
        if (!Files.isRegularFile(indexPath)) {
            return Optional.empty();
        }
        try {
            String needle = ref.trim();
            String json = Files.readString(indexPath);
            Matcher objects = JSON_OBJECT_PATTERN.matcher(json);
            List<IndexCandidate> candidates = new ArrayList<>();
            while (objects.find()) {
                String object = objects.group();
                IndexCandidate candidate = IndexCandidate.from(object);
                if (candidate.path() != null && !candidate.path().isBlank()) {
                    candidates.add(candidate);
                    if (matchesIndexEntry(candidate, needle) && isLiteRtIndexObject(candidate)) {
                        return Optional.of(new RouteMatch(
                                Path.of(candidate.path()),
                                "fast-index-litert",
                                true));
                    }
                }
            }
            if (equivalentLiteRtFastPathEnabled()) {
                Optional<IndexCandidate> requested = candidates.stream()
                        .filter(candidate -> matchesIndexEntry(candidate, needle))
                        .findFirst();
                if (requested.isPresent()) {
                    return findEquivalentLiteRtModelPath(candidates, requested.get())
                            .map(path -> new RouteMatch(
                                    path,
                                    "fast-index-equivalent-litert",
                                    true));
                }
            }
        } catch (Exception ignored) {
            // Full CLI resolver remains available if the lightweight parse fails.
        }
        return Optional.empty();
    }

    static record RouteResolution(
            Path path,
            String source,
            boolean cacheHit,
            String selectedArtifact,
            long durationNanos) {

        RouteResolution withDurationNanos(long durationNanos) {
            return new RouteResolution(path, source, cacheHit, selectedArtifact, durationNanos);
        }
    }

    static record RouteMatch(
            Path path,
            String source,
            boolean cacheHit) {
    }

    private static boolean matchesIndexEntry(IndexCandidate candidate, String ref) {
        if (candidate == null) {
            return false;
        }
        if (ref.equals(candidate.id())
                || ref.equals(candidate.shortId())
                || ref.equals(candidate.name())
                || ref.equals(candidate.path())) {
            return true;
        }
        String normalized = ref.replace("\\", "/").toLowerCase(Locale.ROOT);
        String normalizedPath = candidate.path() == null
                ? ""
                : candidate.path().replace("\\", "/").toLowerCase(Locale.ROOT);
        return normalizedPath.endsWith(normalized);
    }

    private static boolean isLiteRtIndexEntry(LocalModelIndex.Entry entry) {
        if (entry == null) {
            return false;
        }
        if (isLiteRtFormat(entry.format)) {
            return true;
        }
        return entry.path != null && isLiteRtPathString(entry.path);
    }

    private static boolean isSafetensorIndexEntry(LocalModelIndex.Entry entry) {
        if (entry == null) {
            return false;
        }
        String format = entry.format == null ? "" : entry.format.trim().toLowerCase(Locale.ROOT);
        if (format.equals("safetensor") || format.equals("safetensors")) {
            return true;
        }
        String path = entry.path == null ? "" : entry.path.trim().toLowerCase(Locale.ROOT);
        return path.endsWith(".safetensors") || path.contains("/safetensors/");
    }

    private static Optional<Path> findCachedGemma4LiteRtModelPath(String safetensorPath) {
        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectCachedGemma4MobileQatLiteRtAlternateRuntime(
                        "safetensor",
                        safetensorPath,
                        safetensorPath,
                        false,
                        true);
        if (!selection.selected() || selection.localPath() == null || selection.localPath().isBlank()) {
            return Optional.empty();
        }
        return findLiteRtLmFromExplicitPath(Path.of(selection.localPath()));
    }

    private static boolean isLiteRtIndexObject(IndexCandidate candidate) {
        return candidate != null
                && (isLiteRtFormat(candidate.format())
                || isLiteRtPathString(candidate.path()));
    }

    private static Optional<Path> findEquivalentLiteRtModelPath(
            List<IndexCandidate> candidates,
            IndexCandidate requested) {
        if (candidates == null || candidates.isEmpty() || !canUseLiteRtEquivalentFor(requested)) {
            return Optional.empty();
        }
        Set<String> requestedKeys = canonicalModelKeys(requested);
        if (requestedKeys.isEmpty()) {
            return Optional.empty();
        }
        return candidates.stream()
                .filter(LiteRtLmFastRun::isLiteRtIndexObject)
                .filter(candidate -> !sameIndexPath(candidate, requested))
                .filter(candidate -> compatibleArchitecture(requested, candidate))
                .filter(candidate -> intersects(requestedKeys, canonicalModelKeys(candidate)))
                .sorted((left, right) -> Integer.compare(
                        equivalentLiteRtPreference(left),
                        equivalentLiteRtPreference(right)))
                .map(IndexCandidate::path)
                .filter(path -> path != null && !path.isBlank())
                .map(Path::of)
                .findFirst();
    }

    private static boolean canUseLiteRtEquivalentFor(IndexCandidate requested) {
        if (requested == null || isLiteRtIndexObject(requested)) {
            return false;
        }
        String format = requested.format() == null ? "" : requested.format().trim().toLowerCase(Locale.ROOT);
        if (!format.isBlank()
                && !format.equals("safetensor")
                && !format.equals("safetensors")
                && !format.equals("gguf")) {
            return false;
        }
        String architecture = requested.architecture() == null
                ? ""
                : requested.architecture().trim().toLowerCase(Locale.ROOT);
        String identity = ((requested.id() == null ? "" : requested.id()) + " "
                + (requested.name() == null ? "" : requested.name()) + " "
                + (requested.path() == null ? "" : requested.path())).toLowerCase(Locale.ROOT);
        return architecture.contains("gemma") || identity.contains("gemma");
    }

    private static boolean compatibleArchitecture(IndexCandidate requested, IndexCandidate candidate) {
        String requestedArchitecture = normalizedArchitecture(requested.architecture());
        String candidateArchitecture = normalizedArchitecture(candidate.architecture());
        return requestedArchitecture.isBlank()
                || candidateArchitecture.isBlank()
                || requestedArchitecture.equals(candidateArchitecture);
    }

    private static String normalizedArchitecture(String architecture) {
        if (architecture == null || architecture.isBlank()) {
            return "";
        }
        String normalized = architecture.trim().toLowerCase(Locale.ROOT);
        if (normalized.equals("unknown")) {
            return "";
        }
        if (normalized.contains("gemma")) {
            return "gemma";
        }
        return normalized.replaceAll("[^a-z0-9]+", "");
    }

    static Set<String> canonicalModelKeys(IndexCandidate candidate) {
        Set<String> keys = new HashSet<>();
        if (candidate == null) {
            return keys;
        }
        addCanonicalModelKey(keys, candidate.id());
        addCanonicalModelKey(keys, candidate.name());
        addCanonicalModelKey(keys, lastPathSegment(candidate.path()));
        return keys;
    }

    private static void addCanonicalModelKey(Set<String> keys, String value) {
        String key = canonicalModelKey(value);
        if (!key.isBlank()) {
            keys.add(key);
        }
    }

    static String canonicalModelKey(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        String leaf = lastPathSegment(value)
                .replace("__", "/")
                .replace('\\', '/')
                .toLowerCase(Locale.ROOT);
        int slash = leaf.lastIndexOf('/');
        if (slash >= 0 && slash + 1 < leaf.length()) {
            leaf = leaf.substring(slash + 1);
        }
        leaf = leaf
                .replaceAll("\\.(litertlm|litert|tflite|tfl|task|gguf|safetensors?|bin)$", "")
                .replaceAll("(?i)(-?litert-?lm|-?litertlm|-?tflite|-?safetensors?|\\.safetensors?)$", "");
        String[] tokens = leaf.split("[^a-z0-9]+");
        StringBuilder key = new StringBuilder();
        for (String token : tokens) {
            if (token.isBlank() || isCanonicalNoiseToken(token)) {
                continue;
            }
            key.append(token);
        }
        return key.toString();
    }

    private static String lastPathSegment(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        String normalized = value.replace('\\', '/');
        int slash = normalized.lastIndexOf('/');
        return slash >= 0 && slash + 1 < normalized.length()
                ? normalized.substring(slash + 1)
                : normalized;
    }

    private static boolean isCanonicalNoiseToken(String token) {
        return switch (token) {
            case "google", "litert", "community", "lm", "gguf", "safetensor", "safetensors",
                    "qat", "mobile", "transformers",
                    "q4", "q5", "q6", "q8", "k", "m", "f16", "fp16", "bf16" -> true;
            default -> false;
        };
    }

    private static int equivalentLiteRtPreference(IndexCandidate candidate) {
        String path = candidate.path() == null ? "" : candidate.path().toLowerCase(Locale.ROOT);
        String name = candidate.name() == null ? "" : candidate.name().toLowerCase(Locale.ROOT);
        int score = 0;
        if (!path.endsWith(".litertlm") && !name.endsWith(".litertlm")) {
            score += 20;
        }
        if (path.contains("qualcomm") || name.contains("qualcomm")) {
            score += 100;
        }
        return score;
    }

    static boolean equivalentLiteRtFastPathEnabled() {
        String configured = System.getProperty("gollek.litert.fast_run.auto_equivalent");
        return configured == null || configured.isBlank() || Boolean.parseBoolean(configured);
    }

    private static boolean sameIndexPath(IndexCandidate left, IndexCandidate right) {
        if (left == null || right == null || left.path() == null || right.path() == null) {
            return false;
        }
        return left.path().equals(right.path());
    }

    private static boolean intersects(Set<String> left, Set<String> right) {
        if (left == null || right == null || left.isEmpty() || right.isEmpty()) {
            return false;
        }
        for (String value : right) {
            if (left.contains(value)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isLiteRtFormat(String format) {
        if (format == null || format.isBlank()) {
            return false;
        }
        String normalized = format.trim().toLowerCase(Locale.ROOT);
        return normalized.equals("litert")
                || normalized.equals("litertlm")
                || normalized.equals("tflite")
                || normalized.equals("tfl")
                || normalized.equals("task");
    }

    private static boolean isLiteRtPathString(String path) {
        if (path == null || path.isBlank()) {
            return false;
        }
        return path.replace("\\", "/").toLowerCase(Locale.ROOT).endsWith(".litertlm");
    }

    static record IndexCandidate(
            String object,
            String id,
            String shortId,
            String name,
            String architecture,
            String format,
            String path) {

        static IndexCandidate from(String object) {
            return new IndexCandidate(
                    object,
                    jsonStringField(object, "id").orElse(null),
                    jsonStringField(object, "shortId").orElse(null),
                    jsonStringField(object, "name").orElse(null),
                    jsonStringField(object, "architecture").orElse(null),
                    jsonStringField(object, "format").orElse(null),
                    jsonStringField(object, "path").orElse(null));
        }
    }

    private static Optional<String> jsonStringField(String object, String field) {
        String quotedField = Pattern.quote(field);
        Pattern fieldPattern = Pattern.compile("\"" + quotedField + "\"\\s*:\\s*\"((?:\\\\.|[^\"])*)\"");
        Matcher matcher = fieldPattern.matcher(object);
        if (!matcher.find()) {
            return Optional.empty();
        }
        return Optional.of(unescapeJsonString(matcher.group(1)));
    }

    private static String unescapeJsonString(String value) {
        StringBuilder out = new StringBuilder(value.length());
        boolean escaping = false;
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            if (!escaping) {
                if (ch == '\\') {
                    escaping = true;
                } else {
                    out.append(ch);
                }
                continue;
            }
            switch (ch) {
                case '"', '\\', '/' -> out.append(ch);
                case 'b' -> out.append('\b');
                case 'f' -> out.append('\f');
                case 'n' -> out.append('\n');
                case 'r' -> out.append('\r');
                case 't' -> out.append('\t');
                default -> out.append(ch);
            }
            escaping = false;
        }
        if (escaping) {
            out.append('\\');
        }
        return out.toString();
    }

    private static Optional<Path> findLiteRtLmFromExplicitPath(Path path) {
        try {
            if (Files.isRegularFile(path) && isLiteRtLm(path)) {
                return Optional.of(path.toAbsolutePath().normalize());
            }
            if (Files.isRegularFile(path)) {
                return Optional.empty();
            }
            Path searchDir = Files.isDirectory(path) ? path : path.getParent();
            if (searchDir == null || !Files.isDirectory(searchDir)) {
                return Optional.empty();
            }
            try (var stream = Files.list(searchDir)) {
                return stream
                        .filter(Files::isRegularFile)
                        .filter(LiteRtLmFastRun::isLiteRtLm)
                        .sorted((left, right) -> Integer.compare(
                                liteRtLmPreference(left),
                                liteRtLmPreference(right)))
                        .findFirst()
                        .map(candidate -> candidate.toAbsolutePath().normalize());
            }
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static boolean isLiteRtLm(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".litertlm");
    }

    private static int liteRtLmPreference(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        int score = 0;
        if (name.contains("qualcomm")) {
            score += 100;
        }
        if (!name.contains("gemma-4-e2b-it.litertlm")) {
            score += 10;
        }
        return score;
    }

    private static void generate(RouteResolution route, FastArgs args) throws Exception {
        generate(route, args, System.out, System.err, null, "fast path");
    }

    private static void generate(
            RouteResolution route,
            FastArgs args,
            PrintStream out,
            PrintStream err,
            EngineCache engineCache,
            String runnerName) throws Exception {
        Path modelPath = route.path();
        long startNanos = System.nanoTime();
        Files.createDirectories(cacheDir());
        Engine.Companion.setNativeMinLogSeverity(nativeLogSeverity());

        int engineMaxNumTokens = maxNumTokens(args);
        if (args.banner) {
            printRunHeader(modelPath, args, out);
        }
        long beforeEngineInitNanos = System.nanoTime();
        String requestedBackend = normalizedBackend(args.backend);
        try (EngineLease lease = engineCache == null
                ? openEngine(modelPath, requestedBackend, engineMaxNumTokens, true, false)
                : engineCache.acquire(modelPath, requestedBackend, engineMaxNumTokens)) {
            long afterEngineInitNanos = System.nanoTime();
            String warmSuffix = engineCache == null ? "" : ", warmEngine=" + lease.reused;
            if (args.banner) {
                printExecutionRoute(out, lease, runnerName);
                out.printf("Using official LiteRT-LM JVM %s for %s (backend=%s, speculativeDecoding=%s, engineMaxNumTokens=%d%s).%n",
                        runnerName,
                        modelPath.getFileName(),
                        lease.backendName,
                        lease.speculativeDecoding,
                        engineMaxNumTokens,
                        warmSuffix);
            }
            if (lease.fallbackReason != null && args.banner) {
                err.printf("LiteRT-LM GPU backend failed; using official CPU fallback instead: %s%n",
                        lease.fallbackReason);
            }

            SamplerConfig sampler = new SamplerConfig(
                    Math.max(1, args.topK),
                    Math.max(0.0d, Math.min(1.0d, args.topP)),
                    Math.max(0.0d, args.temperature),
                    0);
            ConversationConfig conversationConfig = new ConversationConfig(
                    null,
                    List.of(),
                    List.of(),
                    sampler,
                    false,
                    null,
                    Map.of());
            try (Conversation conversation = lease.engine.createConversation(conversationConfig)) {
                long afterConversationCreateNanos = System.nanoTime();
                CountDownLatch done = new CountDownLatch(1);
                AtomicBoolean cancelled = new AtomicBoolean(false);
                AtomicInteger streamUpdates = new AtomicInteger(0);
                AtomicLong firstChunkNanos = new AtomicLong(0L);
                AtomicReference<Throwable> error = new AtomicReference<>();
                ApproximateTokenStreamLimiter tokenLimiter = new ApproximateTokenStreamLimiter(args.maxTokens);
                conversation.sendMessageAsync(Contents.Companion.of(promptForModel(args.prompt)), new MessageCallback() {
                    @Override
                    public void onMessage(com.google.ai.edge.litertlm.Message message) {
                        if (cancelled.get()) {
                            return;
                        }
                        streamUpdates.incrementAndGet();
                        firstChunkNanos.compareAndSet(0L, System.nanoTime());
                        String delta = tokenLimiter.offer(message.toString());
                        if (!delta.isEmpty()) {
                            out.print(delta);
                            out.flush();
                        }
                        if (tokenLimiter.atLimit() && cancelled.compareAndSet(false, true)) {
                            conversation.cancelProcess();
                            done.countDown();
                        }
                    }

                    @Override
                    public void onDone() {
                        done.countDown();
                    }

                    @Override
                    public void onError(Throwable throwable) {
                        if (!cancelled.get()) {
                            error.compareAndSet(null, throwable);
                        }
                        done.countDown();
                    }
                }, Map.of());
                if (!done.await(timeout().toMillis(), TimeUnit.MILLISECONDS)) {
                    conversation.cancelProcess();
                    throw new IllegalStateException("LiteRT-LM fast path timed out");
                }
                if (error.get() != null && !cancelled.get()) {
                    throw new IllegalStateException(error.get());
                }
                lease.markPrewarmed(prewarmIterationCount());
                long endNanos = System.nanoTime();
                if (args.banner) {
                    printFastRunStats(out,
                            streamUpdates.get(),
                            tokenLimiter.emittedTokenCount(),
                            startNanos,
                            beforeEngineInitNanos,
                            afterEngineInitNanos,
                            afterConversationCreateNanos,
                            firstChunkNanos.get(),
                            endNanos);
                }
                if (Boolean.getBoolean("gollek.litert.fast_run.profile")) {
                    long first = firstChunkNanos.get();
                    err.printf(Locale.ROOT,
                            "[Fast LiteRT-LM profile] routeResolve=%.3fs routeCacheHit=%s selectedArtifact=%s routeSource=%s cache+config=%.3fs engineInit=%.3fs conversation=%.3fs firstChunk=%.3fs total=%.3fs%n",
                            seconds(route.durationNanos()),
                            route.cacheHit(),
                            route.selectedArtifact(),
                            route.source(),
                            seconds(beforeEngineInitNanos - startNanos),
                            seconds(afterEngineInitNanos - beforeEngineInitNanos),
                            seconds(afterConversationCreateNanos - afterEngineInitNanos),
                            first == 0L ? -1.0d : seconds(first - afterConversationCreateNanos),
                            seconds(System.nanoTime() - startNanos));
                }
            }
        }
    }

    static void printFastRunStats(
            PrintStream out,
            int streamUpdates,
            int outputTokens,
            long startNanos,
            long beforeEngineInitNanos,
            long afterEngineInitNanos,
            long afterConversationCreateNanos,
            long firstChunkNanos,
            long endNanos) {
        long durationMs = TimeUnit.NANOSECONDS.toMillis(Math.max(0L, endNanos - startNanos));
        double durationSeconds = durationMs / 1000.0d;
        double speed = durationSeconds > 0.0d ? outputTokens / durationSeconds : 0.0d;
        double loadMs = nanosToMillis(Math.max(0L, afterEngineInitNanos - beforeEngineInitNanos));
        boolean hasTtft = firstChunkNanos > 0L;
        double ttftMs = hasTtft ? nanosToMillis(Math.max(0L, firstChunkNanos - startNanos)) : -1.0d;
        double engineTtftMs = hasTtft
                ? nanosToMillis(Math.max(0L, firstChunkNanos - afterConversationCreateNanos))
                : -1.0d;
        double tokenLatencyMs = outputTokens > 0
                ? nanosToMillis(Math.max(0L, endNanos - (hasTtft ? firstChunkNanos : startNanos))) / outputTokens
                : 0.0d;

        if (hasTtft) {
            out.printf("%n[Stream updates: %d, Duration: %.2fs, Speed: %.2f t/s, TTFT: %.2f ms]%n",
                    streamUpdates, durationSeconds, speed, ttftMs);
        } else {
            out.printf("%n[Stream updates: %d, Duration: %.2fs, Speed: %.2f t/s]%n",
                    streamUpdates, durationSeconds, speed);
        }
        out.println("Performance Metrics:");
        out.printf("  load time      = %9.2f ms%n", loadMs);
        out.printf("  generation     = %9.2f t/s (%d tokens)%n", speed, outputTokens);
        if (hasTtft) {
            out.printf("  latency (ttft) = %9.2f ms%n", ttftMs);
            out.printf("  engine ttft    = %9.2f ms (model-ready)%n", engineTtftMs);
        }
        out.printf("  token latency  = %9.2f ms/token%n%n", tokenLatencyMs);

        try {
            ObservabilityManager obs = ObservabilityManager.global();
            if (obs != null) {
                MetricsRegistry reg = obs.getMetricsRegistry();
                AttributesBuilder attrs = Attributes.builder()
                        .put("engine", "litert")
                        .put("fast_path", true);

                Attributes finalAttrs = attrs.build();
                reg.record("gollek.llm.generation.duration", durationMs, finalAttrs);
                reg.record("gollek.llm.generation.speed", speed, finalAttrs);
                reg.record("gollek.llm.generation.token_latency", tokenLatencyMs, finalAttrs);
                reg.record("gollek.llm.model.open_time", loadMs, finalAttrs);
                if (hasTtft) {
                    reg.record("gollek.llm.native.prefill", ttftMs, finalAttrs);
                }
            }
        } catch (Exception e) {
            // Ignore telemetry exceptions
        }
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0d;
    }

    static void printRunHeader(Path modelPath, FastArgs args, PrintStream out) {
        printBanner(out);
        if (shouldPrintResolvedModelPath(modelPath, args)) {
            out.printf("Resolved local model index entry: %s%n", modelPath);
        }
        out.printf("Model: %s%n", modelPath);
        out.printf("Provider: litert, format=%s%n", liteRtModelFormat(modelPath));
    }

    static void printExecutionRoute(PrintStream out, EngineLease lease, String runnerName) {
        out.println(executionRouteLine(lease.backendName, runnerName));
        out.println("--------------------------------------------------");
    }

    static String liteRtModelFormat(Path modelPath) {
        if (modelPath == null || modelPath.getFileName() == null) {
            return "litert";
        }
        String name = modelPath.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".litertlm")) {
            return "litertlm";
        }
        if (name.endsWith(".tflite") || name.endsWith(".tfl")) {
            return "tflite";
        }
        if (name.endsWith(".task")) {
            return "task";
        }
        return "litert";
    }

    static String executionRouteLine(String backendName, String runnerName) {
        String backend = backendName == null || backendName.isBlank() ? "unknown" : backendName;
        return "Execution route: litert (backend=" + backend + ") [" + executionRouteTag(runnerName) + "]";
    }

    private static String executionRouteTag(String runnerName) {
        String normalized = runnerName == null ? "" : runnerName.trim().toLowerCase(Locale.ROOT);
        if (normalized.contains("daemon")) {
            return "official_litert_lm_jvm_daemon";
        }
        return "official_litert_lm_jvm_fast_path";
    }

    private static void printBanner(PrintStream out) {
        out.println("  _____       _  _      _    ");
        out.println(" / ____|     | || |    | |   ");
        out.println("| |  __  ___ | || | ___| | __");
        out.println("| | |_ |/ _ \\| || |/ _ \\ |/ /");
        out.println("| |__| | (_) | || |  __/   < ");
        out.println(" \\_____|\\___/|_||_|\\___|_|\\_\\");
        out.println();
    }

    private static boolean shouldPrintResolvedModelPath(Path modelPath, FastArgs args) {
        if (args == null || args.model == null || args.model.isBlank() || args.modelFile != null) {
            return false;
        }
        try {
            Path requested = Path.of(args.model);
            Path resolved = requested.isAbsolute()
                    ? requested.toAbsolutePath().normalize()
                    : Path.of("").toAbsolutePath().resolve(requested).normalize();
            return !resolved.equals(modelPath.toAbsolutePath().normalize());
        } catch (Exception ignored) {
            return true;
        }
    }

    private static EngineLease openEngine(
            Path modelPath,
            String requestedBackend,
            int engineMaxNumTokens,
            boolean closeOnClose,
            boolean reused) {
        boolean speculativeDecoding = speculativeDecodingEnabled(requestedBackend);
        ExperimentalFlags.INSTANCE.setEnableSpeculativeDecoding(speculativeDecoding);
        EngineConfig engineConfig = new EngineConfig(
                modelPath.toString(),
                backend(requestedBackend),
                null,
                null,
                engineMaxNumTokens,
                null,
                cacheDir().toString());

        try {
            return new EngineLease(
                    initializeEngine(engineConfig),
                    backendName(requestedBackend),
                    speculativeDecoding,
                    null,
                    closeOnClose,
                    reused);
        } catch (Throwable gpuFailure) {
            if (!"gpu".equals(requestedBackend)
                    || !Boolean.parseBoolean(System.getProperty("gollek.litert.fast_run.cpu_fallback", "true"))) {
                throw gpuFailure;
            }
            String fallbackReason = summarize(gpuFailure);
            speculativeDecoding = speculativeDecodingEnabled("cpu");
            ExperimentalFlags.INSTANCE.setEnableSpeculativeDecoding(speculativeDecoding);
            EngineConfig cpuEngineConfig = new EngineConfig(
                    modelPath.toString(),
                    new Backend.CPU(),
                    null,
                    null,
                    engineMaxNumTokens,
                    null,
                    cacheDir().toString());
            return new EngineLease(
                    initializeEngine(cpuEngineConfig),
                    "CPU",
                    speculativeDecoding,
                    fallbackReason,
                    closeOnClose,
                    reused);
        }
    }

    private static Path cacheDir() {
        String configured = System.getProperty("gollek.litert.lm_jvm_bridge.cache_dir");
        if (configured != null && !configured.isBlank()) {
            return Path.of(configured).toAbsolutePath();
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "cache", "litert-lm-jvm").toAbsolutePath();
    }

    private static Engine initializeEngine(EngineConfig config) {
        Engine engine = new Engine(config);
        try {
            engine.initialize();
            return engine;
        } catch (Throwable failure) {
            try {
                engine.close();
            } catch (Throwable ignored) {
            }
            throw failure;
        }
    }

    private static LogSeverity nativeLogSeverity() {
        String configured = System.getProperty("gollek.litert.fast_run.native_log_severity", "ERROR");
        try {
            return LogSeverity.valueOf(configured.trim().toUpperCase(Locale.ROOT));
        } catch (Exception ignored) {
            return LogSeverity.ERROR;
        }
    }

    private static String metalAwareGpuBackendName() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        return os.contains("mac") ? "GPU/Metal" : "GPU";
    }

    private static Backend backend(String normalizedBackend) {
        return "cpu".equals(normalizedBackend) ? new Backend.CPU() : new Backend.GPU();
    }

    private static String backendName(String normalizedBackend) {
        return "cpu".equals(normalizedBackend) ? "CPU" : metalAwareGpuBackendName();
    }

    private static String normalizedBackend(String configured) {
        String normalized = configured == null ? "gpu" : configured.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "cpu" -> "cpu";
            case "gpu", "metal", "auto", "" -> "gpu";
            default -> "gpu";
        };
    }

    private static boolean speculativeDecodingEnabled(String normalizedBackend) {
        String configured = System.getProperty("gollek.litert.fast_run.speculative_decoding");
        if (configured != null && !configured.isBlank()) {
            return Boolean.parseBoolean(configured);
        }
        return "gpu".equals(normalizedBackend);
    }

    private static String summarize(Throwable throwable) {
        if (throwable == null) {
            return "unknown failure";
        }
        String message = throwable.getMessage();
        if (message == null || message.isBlank()) {
            message = throwable.getClass().getSimpleName();
        }
        return message.replace('\n', ' ').strip();
    }

    private static int maxNumTokens(FastArgs args) {
        Integer configured = Integer.getInteger("gollek.litert.lm_jvm_bridge.max_num_tokens");
        if (configured != null && configured > 0) {
            return configured;
        }
        if (!dynamicEngineTokensEnabled()) {
            return DEFAULT_MAX_NUM_TOKENS;
        }
        return dynamicEngineTokenBudget(args.prompt, args.maxTokens);
    }

    static int dynamicEngineTokenBudget(String prompt, int maxTokens) {
        int requested = Math.max(1, maxTokens);
        int promptEstimate = approximateTokenCount(prompt);
        int dynamic = requested + promptEstimate + 64;
        int floor = Integer.getInteger("gollek.litert.fast_run.min_engine_tokens", DEFAULT_MAX_NUM_TOKENS_FLOOR);
        int ceiling = Integer.getInteger("gollek.litert.fast_run.max_engine_tokens", DEFAULT_MAX_NUM_TOKENS_CEILING);
        return Math.max(1, Math.min(Math.max(floor, dynamic), Math.max(floor, ceiling)));
    }

    static boolean dynamicEngineTokensEnabled() {
        String configured = System.getProperty("gollek.litert.fast_run.dynamic_engine_tokens");
        return configured == null || configured.isBlank() || Boolean.parseBoolean(configured);
    }

    static String promptForModel(String prompt) {
        if (prompt == null || !questionNormalizationEnabled()) {
            return prompt;
        }
        String stripped = prompt.strip();
        if (stripped.isEmpty() || !BARE_QUESTION_PATTERN.matcher(stripped).matches()) {
            return prompt;
        }
        return SHORT_QUESTION_INSTRUCTION + Character.toUpperCase(stripped.charAt(0)) + stripped.substring(1) + "?";
    }

    static boolean questionNormalizationEnabled() {
        String configured = System.getProperty("gollek.litert.fast_run.normalize_short_questions");
        return configured == null || configured.isBlank() || Boolean.parseBoolean(configured);
    }

    private static Duration timeout() {
        return Duration.ofSeconds(Long.getLong("gollek.litert.lm_jvm_bridge.timeout_seconds", 180L));
    }

    private static double seconds(long nanos) {
        return nanos / 1_000_000_000.0d;
    }

    private static int approximateTokenCount(CharSequence text) {
        int tokens = 0;
        boolean inToken = false;
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (Character.isWhitespace(ch)) {
                inToken = false;
            } else if (!inToken) {
                tokens++;
                inToken = true;
            }
        }
        return tokens;
    }

    private static String limitByApproximateTokens(String text, int maxTokens) {
        int tokens = 0;
        boolean inToken = false;
        int end = text.length();
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (Character.isWhitespace(ch)) {
                inToken = false;
            } else if (!inToken) {
                tokens++;
                inToken = true;
                if (tokens > maxTokens) {
                    end = i;
                    break;
                }
            }
        }
        return text.substring(0, end).stripTrailing();
    }

    private static boolean daemonEnabled() {
        return Boolean.parseBoolean(System.getProperty("gollek.litert.fast_run.daemon", "false"));
    }

    private static OptionalInt requestExistingDaemon(String[] rawArgs) {
        return sendDaemonRequestWithRetries(DAEMON_MAGIC, rawArgs, daemonRequestRetryMillis());
    }

    private static OptionalInt requestDaemon(String[] rawArgs) {
        OptionalInt status = requestExistingDaemon(rawArgs);
        if (status.isPresent()) {
            return status;
        }
        if (!startDaemon()) {
            return OptionalInt.empty();
        }
        return sendDaemonRequestWithRetries(DAEMON_MAGIC, rawArgs, daemonStartTimeoutMillis());
    }

    private static OptionalInt requestDaemonPrewarm(String[] rawArgs) {
        OptionalInt status = sendDaemonRequestWithRetries(DAEMON_PREWARM_MAGIC, rawArgs, daemonRequestRetryMillis());
        if (status.isPresent()) {
            return status;
        }
        if (!startDaemon()) {
            return OptionalInt.empty();
        }
        return sendDaemonRequestWithRetries(DAEMON_PREWARM_MAGIC, rawArgs, daemonStartTimeoutMillis());
    }

    private static OptionalInt sendDaemonRequestWithRetries(String magic, String[] rawArgs, long retryMillis) {
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(Math.max(0L, retryMillis));
        while (true) {
            OptionalInt status = sendDaemonRequest(magic, rawArgs);
            if (status.isPresent()) {
                return status;
            }
            if (readDaemonInfo().isEmpty() || System.nanoTime() >= deadlineNanos) {
                return OptionalInt.empty();
            }
            try {
                Thread.sleep(daemonRequestRetrySleepMillis());
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return OptionalInt.empty();
            }
        }
    }

    private static OptionalInt sendDaemonRequest(String magic, String[] rawArgs) {
        OptionalInt port = readDaemonPort();
        if (port.isEmpty()) {
            return OptionalInt.empty();
        }
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getLoopbackAddress(), port.getAsInt()),
                    daemonConnectTimeoutMillis());
            socket.setSoTimeout((int) timeout().toMillis());
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            DataInputStream in = new DataInputStream(socket.getInputStream());
            writeString(out, magic);
            out.writeInt(rawArgs.length);
            for (String arg : rawArgs) {
                writeString(out, arg);
            }
            out.flush();
            while (true) {
                int channel = in.read();
                if (channel < 0) {
                    return OptionalInt.empty();
                }
                int length = in.readInt();
                if (channel == 0) {
                    return OptionalInt.of(length);
                }
                byte[] payload = in.readNBytes(length);
                PrintStream target = channel == 2 ? System.err : System.out;
                target.write(payload);
                target.flush();
            }
        } catch (ConnectException ignored) {
            deleteDaemonPortFile();
            return OptionalInt.empty();
        } catch (Exception e) {
            if (Boolean.getBoolean("gollek.litert.fast_run.debug")) {
                System.err.println("LiteRT-LM daemon request failed: " + summarize(e));
            }
            return OptionalInt.empty();
        }
    }

    private static boolean startDaemon() {
        deleteDaemonPortFile();
        try {
            Files.createDirectories(daemonRunDir());
            Files.createDirectories(daemonLogFile().getParent());
            List<String> command = new ArrayList<>();
            command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
            command.add("-Xmx" + System.getProperty("gollek.litert.fast_run.daemon_heap",
                    System.getProperty("gollek.litert.fast_run.heap", "8g")));
            command.add("-XX:MaxDirectMemorySize=" + System.getProperty("gollek.litert.fast_run.daemon_direct_memory", "24g"));
            addDaemonCrashDiagnostics(command);
            command.add("--enable-preview");
            command.add("--add-modules");
            command.add("jdk.incubator.vector");
            command.add("--enable-native-access=ALL-UNNAMED");
            copyGollekSystemProperties(command);
            command.add("-Djava.library.path=" + System.getProperty("java.library.path", ""));
            command.add("-cp");
            command.add(System.getProperty("java.class.path"));
            command.add(LiteRtLmFastRun.class.getName());
            command.add("__daemon");

            if (!startDetachedDaemon(command)) {
                return false;
            }
            long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(daemonStartTimeoutMillis());
            while (System.nanoTime() < deadlineNanos) {
                OptionalInt port = readDaemonPort();
                if (port.isPresent()) {
                    return true;
                }
                Thread.sleep(100L);
            }
            return readDaemonPort().isPresent();
        } catch (Exception e) {
            if (Boolean.getBoolean("gollek.litert.fast_run.debug")) {
                System.err.println("LiteRT-LM daemon failed to start: " + summarize(e));
            }
            return false;
        }
    }

    private static void addDaemonCrashDiagnostics(List<String> command) {
        String defaultErrorFile = daemonLogFile().getParent()
                .resolve("litert-fast-daemon-hs_err_pid%p.log")
                .toString();
        command.add("-XX:ErrorFile=" + System.getProperty(
                "gollek.litert.fast_run.daemon_error_file",
                defaultErrorFile));
        if (Boolean.parseBoolean(System.getProperty("gollek.litert.fast_run.daemon_signal_log", "false"))) {
            String defaultSignalLog = daemonLogFile().getParent()
                    .resolve("litert-fast-daemon-signal-%p.log")
                    .toString();
            command.add("-Xlog:os+signal=info:file=" + System.getProperty(
                    "gollek.litert.fast_run.daemon_signal_log_file",
                    defaultSignalLog)
                    + ":time,uptime,level,tags");
        }
    }

    private static boolean startDetachedDaemon(List<String> command) throws IOException, InterruptedException {
        String launcher = daemonLauncherMode();
        if (Boolean.getBoolean("gollek.litert.fast_run.debug")) {
            System.err.println("LiteRT-LM daemon launcher: " + launcher);
        }
        if (launcher.equals("launchctl")) {
            return isMacOs() && startDetachedDaemonWithLaunchctl(command);
        }
        if (launcher.equals("auto") && isMacOs() && startDetachedDaemonWithLaunchctl(command)) {
            return true;
        }
        return startDetachedDaemonWithNohup(command);
    }

    static String daemonLauncherMode() {
        String launcher = System.getProperty("gollek.litert.fast_run.daemon_launcher", "auto")
                .trim()
                .toLowerCase(Locale.ROOT);
        return switch (launcher) {
            case "auto", "launchctl", "nohup" -> launcher;
            default -> "nohup";
        };
    }

    private static boolean startDetachedDaemonWithLaunchctl(List<String> command) throws IOException, InterruptedException {
        String launchctl = "/bin/launchctl";
        if (!Files.isExecutable(Path.of(launchctl))) {
            return false;
        }
        removeLaunchctlDaemon();
        String shellCommand = daemonShellCommand(command, false);
        Process launcher = new ProcessBuilder(
                launchctl,
                "submit",
                "-l",
                launchctlLabel(),
                "--",
                "/bin/sh",
                "-c",
                shellCommand)
                .redirectOutput(ProcessBuilder.Redirect.appendTo(daemonLogFile().toFile()))
                .redirectError(ProcessBuilder.Redirect.appendTo(daemonLogFile().toFile()))
                .start();
        if (!launcher.waitFor(3, TimeUnit.SECONDS)) {
            return true;
        }
        boolean launched = launcher.exitValue() == 0;
        if (!launched && Boolean.getBoolean("gollek.litert.fast_run.debug")) {
            System.err.println("LiteRT-LM launchctl daemon submit failed; falling back to nohup");
        }
        return launched;
    }

    private static boolean startDetachedDaemonWithNohup(List<String> command) throws IOException, InterruptedException {
        Process launcher = new ProcessBuilder("/bin/sh", "-c", daemonShellCommand(command, true)).start();
        launcher.waitFor(2, TimeUnit.SECONDS);
        return true;
    }

    private static String daemonShellCommand(List<String> command, boolean useNohup) {
        String home = System.getProperty("user.home");
        StringBuilder shellCommand = new StringBuilder();
        shellCommand.append("export DYLD_LIBRARY_PATH=")
                .append(shellQuote(home + "/.gollek/libs"))
                .append("${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}; ");
        shellCommand.append("export DYLD_FALLBACK_LIBRARY_PATH=")
                .append(shellQuote(home + "/.gollek/libs"))
                .append("${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}; ");
        if (useNohup) {
            shellCommand.append("nohup ");
        } else {
            shellCommand.append("exec ");
        }
        for (int i = 0; i < command.size(); i++) {
            if (i > 0 || useNohup) {
                shellCommand.append(' ');
            }
            shellCommand.append(shellQuote(command.get(i)));
        }
        shellCommand.append(" >> ").append(shellQuote(daemonLogFile().toString()))
                .append(" 2>&1 < /dev/null");
        if (useNohup) {
            shellCommand.append(" &");
        }
        return shellCommand.toString();
    }

    private static boolean isMacOs() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac");
    }

    private static String launchctlLabel() {
        return System.getProperty("gollek.litert.fast_run.daemon_launchctl_label",
                "tech.kayys.gollek.litert-fast-daemon");
    }

    private static void removeLaunchctlDaemon() {
        if (!isMacOs()) {
            return;
        }
        try {
            new ProcessBuilder("/bin/launchctl", "remove", launchctlLabel())
                    .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                    .redirectError(ProcessBuilder.Redirect.DISCARD)
                    .start()
                    .waitFor(2, TimeUnit.SECONDS);
        } catch (Exception ignored) {
        }
    }

    private static String shellQuote(String value) {
        return "'" + value.replace("'", "'\\''") + "'";
    }

    private static void copyGollekSystemProperties(List<String> command) {
        Properties properties = System.getProperties();
        for (String name : properties.stringPropertyNames()) {
            if (!name.startsWith("gollek.litert.")) {
                continue;
            }
            String value = properties.getProperty(name);
            if (value != null) {
                command.add("-D" + name + "=" + value);
            }
        }
    }

    private static int runDaemon() {
        try {
            Files.createDirectories(daemonRunDir());
            Files.createDirectories(cacheDir());
        } catch (IOException e) {
            System.err.println("Failed to create LiteRT-LM daemon directories: " + summarize(e));
            return 1;
        }
        AtomicBoolean stop = new AtomicBoolean(false);
        int port = -1;
        try (ServerSocket server = new ServerSocket(0, 16, InetAddress.getLoopbackAddress());
             EngineCache engineCache = new EngineCache()) {
            port = server.getLocalPort();
            writeDaemonPort(port);
            final int daemonPort = port;
            Runtime.getRuntime().addShutdownHook(new Thread(
                    () -> daemonLog("shutdown hook port=%d", daemonPort),
                    "gollek-litert-fast-daemon-shutdown-log"));
            server.setSoTimeout(1000);
            long lastRequestNanos = System.nanoTime();
            long idleSeconds = Long.getLong("gollek.litert.fast_run.daemon_idle_seconds", 900L);
            long idleNanos = TimeUnit.SECONDS.toNanos(idleSeconds);
            daemonLog("started port=%d pid=%d idleSeconds=%d keyHash=%s",
                    port,
                    ProcessHandle.current().pid(),
                    idleSeconds,
                    daemonKeyHash());
            while (!stop.get()) {
                try (Socket socket = server.accept()) {
                    lastRequestNanos = System.nanoTime();
                    boolean shouldStop = handleDaemonClient(socket, engineCache);
                    stop.set(shouldStop);
                    daemonLog("request complete port=%d stop=%s", port, shouldStop);
                } catch (SocketTimeoutException ignored) {
                    if (System.nanoTime() - lastRequestNanos > idleNanos) {
                        daemonLog("idle timeout after %ds", idleSeconds);
                        break;
                    }
                }
            }
            return 0;
        } catch (Throwable throwable) {
            System.err.println("LiteRT-LM daemon failed: " + summarize(throwable));
            if (Boolean.getBoolean("gollek.litert.fast_run.debug")) {
                throwable.printStackTrace(System.err);
            }
            return 1;
        } finally {
            daemonLog("stopped port=%d", port);
            deleteDaemonPortFileIfPort(port);
        }
    }

    private static boolean handleDaemonClient(Socket socket, EngineCache engineCache) {
        try {
            DataInputStream in = new DataInputStream(socket.getInputStream());
            DataOutputStream framed = new DataOutputStream(socket.getOutputStream());
            String magic = readString(in);
            if (DAEMON_STOP_MAGIC.equals(magic)) {
                daemonLog("stop requested");
                writeStatus(framed, 0);
                return true;
            }
            boolean prewarmRequest = DAEMON_PREWARM_MAGIC.equals(magic);
            if (!DAEMON_MAGIC.equals(magic) && !prewarmRequest) {
                writeStatus(framed, FALLBACK_TO_FULL_CLI);
                return false;
            }
            int argCount = in.readInt();
            if (argCount < 0 || argCount > 256) {
                writeStatus(framed, FALLBACK_TO_FULL_CLI);
                return false;
            }
            String[] rawArgs = new String[argCount];
            for (int i = 0; i < argCount; i++) {
                rawArgs[i] = readString(in);
            }
            FastArgs parsed = FastArgs.parse(rawArgs);
            if (!parsed.supported()) {
                writeStatus(framed, FALLBACK_TO_FULL_CLI);
                return false;
            }
            Optional<RouteResolution> route = engineCache.resolveRoute(parsed);
            if (route.isEmpty()) {
                writeStatus(framed, FALLBACK_TO_FULL_CLI);
                return false;
            }
            daemonLog("%s request model=%s backend=%s maxTokens=%d",
                    prewarmRequest ? "prewarm" : "run",
                    route.get().path().getFileName(),
                    normalizedBackend(parsed.backend),
                    parsed.maxTokens);
            try (PrintStream out = new PrintStream(new FramedOutputStream(framed, 1), true, StandardCharsets.UTF_8);
                 PrintStream err = new PrintStream(new FramedOutputStream(framed, 2), true, StandardCharsets.UTF_8)) {
                if (prewarmRequest) {
                    prewarmEngine(route.get().path(), parsed, out, engineCache);
                } else {
                    generate(route.get(), parsed, out, err, engineCache, "daemon");
                }
                out.flush();
                err.flush();
            }
            writeStatus(framed, 0);
        } catch (Throwable throwable) {
            daemonLog("request failed reason=%s", summarize(throwable));
            try {
                DataOutputStream framed = new DataOutputStream(socket.getOutputStream());
                writeFrame(framed, 2, ("LiteRT-LM daemon request failed: " + summarize(throwable) + System.lineSeparator())
                        .getBytes(StandardCharsets.UTF_8));
                writeStatus(framed, FALLBACK_TO_FULL_CLI);
            } catch (IOException ignored) {
            }
        }
        return false;
    }

    private static void prewarmEngine(
            Path modelPath,
            FastArgs args,
            PrintStream out,
            EngineCache engineCache) throws Exception {
        long startNanos = System.nanoTime();
        Files.createDirectories(cacheDir());
        Engine.Companion.setNativeMinLogSeverity(nativeLogSeverity());
        int engineMaxNumTokens = maxNumTokens(args);
        String requestedBackend = normalizedBackend(args.backend);
        try (EngineLease lease = engineCache.acquire(modelPath, requestedBackend, engineMaxNumTokens)) {
            int decodeWarmupTokens = prewarmTokenCount();
            int targetWarmupIterations = prewarmIterationCount();
            int decodeWarmupIterations = 0;
            if (decodeWarmupTokens > 0) {
                decodeWarmupIterations = lease.remainingPrewarmIterations(targetWarmupIterations);
                for (int i = 0; i < decodeWarmupIterations; i++) {
                    warmConversation(lease, args, decodeWarmupTokens);
                }
            }
            lease.markPrewarmed(targetWarmupIterations);
            long durationMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);
            out.printf("Using official LiteRT-LM JVM daemon for %s (backend=%s, speculativeDecoding=%s, engineMaxNumTokens=%d, warmEngine=%s, decodeWarmupTokens=%d, decodeWarmupIterations=%d, warmupTargetIterations=%d).%n",
                    modelPath.getFileName(),
                    lease.backendName,
                    lease.speculativeDecoding,
                    engineMaxNumTokens,
                    lease.reused,
                    decodeWarmupTokens,
                    decodeWarmupIterations,
                    targetWarmupIterations);
            out.printf("[LiteRT-LM daemon prewarm, Duration: %.2fs]%n", durationMs / 1000.0d);
        }
    }

    private static void warmConversation(EngineLease lease, FastArgs args, int maxWarmupTokens) throws Exception {
        SamplerConfig sampler = new SamplerConfig(
                Math.max(1, args.topK),
                Math.max(0.0d, Math.min(1.0d, args.topP)),
                Math.max(0.0d, args.temperature),
                0);
        ConversationConfig conversationConfig = new ConversationConfig(
                null,
                List.of(),
                List.of(),
                sampler,
                false,
                null,
                Map.of());
        try (Conversation conversation = lease.engine.createConversation(conversationConfig)) {
            CountDownLatch done = new CountDownLatch(1);
            AtomicBoolean cancelled = new AtomicBoolean(false);
            AtomicReference<Throwable> error = new AtomicReference<>();
            ApproximateTokenStreamLimiter tokenLimiter = new ApproximateTokenStreamLimiter(maxWarmupTokens);
            conversation.sendMessageAsync(Contents.Companion.of(promptForModel(args.prompt)), new MessageCallback() {
                @Override
                public void onMessage(com.google.ai.edge.litertlm.Message message) {
                    if (cancelled.get()) {
                        return;
                    }
                    tokenLimiter.offer(message.toString());
                    if (tokenLimiter.atLimit() && cancelled.compareAndSet(false, true)) {
                        conversation.cancelProcess();
                        done.countDown();
                    }
                }

                @Override
                public void onDone() {
                    done.countDown();
                }

                @Override
                public void onError(Throwable throwable) {
                    if (!cancelled.get()) {
                        error.compareAndSet(null, throwable);
                    }
                    done.countDown();
                }
            }, Map.of());
            if (!done.await(timeout().toMillis(), TimeUnit.MILLISECONDS)) {
                conversation.cancelProcess();
                throw new IllegalStateException("LiteRT-LM prewarm timed out");
            }
            if (error.get() != null && !cancelled.get()) {
                throw new IllegalStateException(error.get());
            }
        }
    }

    private static ConversationConfig daemonKeepAliveConversationConfig() {
        SamplerConfig sampler = new SamplerConfig(1, 1.0d, 0.0d, 0);
        return new ConversationConfig(
                null,
                List.of(),
                List.of(),
                sampler,
                false,
                null,
                Map.of());
    }

    static boolean keepAliveConversationEnabled() {
        return Boolean.parseBoolean(System.getProperty("gollek.litert.fast_run.keepalive_conversation", "false"));
    }

    static int prewarmTokenCount() {
        return Math.max(0, Integer.getInteger("gollek.litert.fast_run.prewarm_tokens", 0));
    }

    static int prewarmIterationCount() {
        return Math.max(1, Integer.getInteger("gollek.litert.fast_run.prewarm_iterations", 3));
    }

    private static int stopDaemon() {
        OptionalInt port = readDaemonPort();
        if (port.isEmpty()) {
            return 0;
        }
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getLoopbackAddress(), port.getAsInt()),
                    daemonConnectTimeoutMillis());
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            DataInputStream in = new DataInputStream(socket.getInputStream());
            writeString(out, DAEMON_STOP_MAGIC);
            out.flush();
            while (true) {
                int channel = in.read();
                if (channel < 0) {
                    break;
                }
                int length = in.readInt();
                if (channel == 0) {
                    break;
                }
                in.skipNBytes(length);
            }
        } catch (Exception ignored) {
        } finally {
            deleteDaemonPortFile();
            removeLaunchctlDaemon();
        }
        return 0;
    }

    private static Path daemonRunDir() {
        return Path.of(System.getProperty("user.home"), ".gollek", "run");
    }

    private static Path daemonPortFile() {
        return daemonRunDir().resolve("litert-fast-daemon.port");
    }

    private static Path daemonLogFile() {
        return Path.of(System.getProperty("user.home"), ".gollek", "logs", "litert-fast-daemon.log");
    }

    private static OptionalInt readDaemonPort() {
        Optional<DaemonInfo> info = readDaemonInfo();
        if (info.isEmpty()) {
            return OptionalInt.empty();
        }
        if (strictDaemonKey() && !daemonKey().equals(info.get().key())) {
            discardStaleDaemon(info.get());
            return OptionalInt.empty();
        }
        if (verifyDaemonPid() && info.get().pid() > 0 && !isDaemonProcessAlive(info.get().pid())) {
            discardStaleDaemon(info.get());
            return OptionalInt.empty();
        }
        return OptionalInt.of(info.get().port());
    }

    private static Optional<DaemonInfo> readDaemonInfo() {
        try {
            Path portFile = daemonPortFile();
            if (!Files.isRegularFile(portFile)) {
                return Optional.empty();
            }
            List<String> lines = Files.readAllLines(portFile, StandardCharsets.UTF_8);
            if (lines.isEmpty()) {
                return Optional.empty();
            }
            int port = Integer.parseInt(lines.get(0).trim());
            long pid = lines.size() > 1 ? Long.parseLong(lines.get(1).trim()) : -1L;
            String key = lines.size() > 2 ? lines.get(2).trim() : "";
            return port > 0 && port <= 65535
                    ? Optional.of(new DaemonInfo(port, pid, key))
                    : Optional.empty();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static void writeDaemonPort(int port) throws IOException {
        Files.writeString(daemonPortFile(),
                port + System.lineSeparator()
                        + ProcessHandle.current().pid() + System.lineSeparator()
                        + daemonKey() + System.lineSeparator(),
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static String daemonKey() {
        String classPath = System.getProperty("java.class.path", "");
        if (!classPath.isBlank() && classPath.indexOf(File.pathSeparatorChar) < 0) {
            try {
                Path jar = Path.of(classPath);
                if (Files.isRegularFile(jar)) {
                    Path normalized = jar.toAbsolutePath().normalize();
                    return normalized
                            + "|size=" + Files.size(normalized)
                            + "|mtime=" + Files.getLastModifiedTime(normalized).toMillis();
                }
            } catch (Exception ignored) {
                // Fall through to the raw classpath identity below.
            }
        }
        return classPath;
    }

    private static String daemonKeyHash() {
        return Integer.toHexString(daemonKey().hashCode());
    }

    static boolean isDaemonProcessAlive(long pid) {
        if (pid <= 0 || pid == ProcessHandle.current().pid()) {
            return false;
        }
        try {
            Optional<ProcessHandle> handle = ProcessHandle.of(pid);
            return handle.isPresent() && handle.get().isAlive();
        } catch (Exception ignored) {
            return false;
        }
    }

    private static void daemonLog(String format, Object... args) {
        System.err.printf("%tF %<tT [litert-daemon] %s%n",
                System.currentTimeMillis(),
                String.format(Locale.ROOT, format, args));
        System.err.flush();
    }

    private static void discardStaleDaemon(DaemonInfo info) {
        deleteDaemonPortFile();
        try {
            terminateStaleDaemonProcess(info.pid());
        } catch (Exception ignored) {
        } finally {
            removeLaunchctlDaemon();
        }
    }

    private static void terminateStaleDaemonProcess(long pid) {
        if (!isDaemonProcessAlive(pid)) {
            return;
        }
        try {
            ProcessHandle process = ProcessHandle.of(pid).orElse(null);
            if (process == null || !process.isAlive()) {
                return;
            }
            if (staleDaemonForceKill()) {
                process.destroyForcibly();
            } else {
                process.destroy();
            }
            process.onExit().get(staleDaemonKillWaitMillis(), TimeUnit.MILLISECONDS);
        } catch (Exception ignored) {
            ProcessHandle.of(pid).ifPresent(ProcessHandle::destroyForcibly);
        }
    }

    static boolean staleDaemonForceKill() {
        return Boolean.parseBoolean(System.getProperty("gollek.litert.fast_run.stale_daemon_force_kill", "true"));
    }

    static long staleDaemonKillWaitMillis() {
        return Math.max(100L, Long.getLong("gollek.litert.fast_run.stale_daemon_kill_wait_ms", 2000L));
    }

    static boolean strictDaemonKey() {
        return Boolean.parseBoolean(System.getProperty("gollek.litert.fast_run.strict_daemon_key", "true"));
    }

    static boolean verifyDaemonPid() {
        return Boolean.parseBoolean(System.getProperty("gollek.litert.fast_run.verify_daemon_pid", "true"));
    }

    private static void deleteDaemonPortFile() {
        try {
            Files.deleteIfExists(daemonPortFile());
        } catch (IOException ignored) {
        }
    }

    private static void deleteDaemonPortFileIfPort(int port) {
        if (port <= 0) {
            return;
        }
        OptionalInt current = readDaemonPort();
        if (current.isPresent() && current.getAsInt() == port) {
            deleteDaemonPortFile();
        }
    }

    private static int daemonConnectTimeoutMillis() {
        return Integer.getInteger("gollek.litert.fast_run.daemon_connect_timeout_ms", 1000);
    }

    private static int daemonStartTimeoutMillis() {
        return Integer.getInteger("gollek.litert.fast_run.daemon_start_timeout_ms", 20_000);
    }

    static long daemonRequestRetryMillis() {
        return Math.max(0L, Long.getLong("gollek.litert.fast_run.daemon_request_retry_ms", 2_000L));
    }

    static long daemonRequestRetrySleepMillis() {
        return Math.max(10L, Long.getLong("gollek.litert.fast_run.daemon_request_retry_sleep_ms", 75L));
    }

    private record DaemonInfo(int port, long pid, String key) {
    }

    private static void writeString(DataOutputStream out, String value) throws IOException {
        byte[] bytes = (value == null ? "" : value).getBytes(StandardCharsets.UTF_8);
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    private static String readString(DataInputStream in) throws IOException {
        int length = in.readInt();
        if (length < 0 || length > 16 * 1024 * 1024) {
            throw new IOException("Invalid LiteRT-LM daemon string length: " + length);
        }
        byte[] bytes = in.readNBytes(length);
        if (bytes.length != length) {
            throw new IOException("Unexpected end of LiteRT-LM daemon string");
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static void writeStatus(DataOutputStream out, int status) throws IOException {
        synchronized (out) {
            out.writeByte(0);
            out.writeInt(status);
            out.flush();
        }
    }

    private static void writeFrame(DataOutputStream out, int channel, byte[] payload) throws IOException {
        synchronized (out) {
            out.writeByte(channel);
            out.writeInt(payload.length);
            out.write(payload);
            out.flush();
        }
    }

    private static final class FramedOutputStream extends OutputStream {
        private final DataOutputStream out;
        private final int channel;
        private final ByteArrayOutputStream buffer = new ByteArrayOutputStream(4096);

        private FramedOutputStream(DataOutputStream out, int channel) {
            this.out = out;
            this.channel = channel;
        }

        @Override
        public void write(int value) throws IOException {
            buffer.write(value);
            if (value == '\n' || buffer.size() >= 4096) {
                flush();
            }
        }

        @Override
        public void write(byte[] bytes, int offset, int length) throws IOException {
            if (length <= 0) {
                return;
            }
            if (buffer.size() == 0 && length >= 4096) {
                byte[] copy = new byte[length];
                System.arraycopy(bytes, offset, copy, 0, length);
                writeFrame(out, channel, copy);
                return;
            }
            buffer.write(bytes, offset, length);
            if (buffer.size() >= 4096 || bytes[offset + length - 1] == '\n') {
                flush();
            }
        }

        @Override
        public void flush() throws IOException {
            if (buffer.size() == 0) {
                return;
            }
            byte[] payload = buffer.toByteArray();
            buffer.reset();
            writeFrame(out, channel, payload);
        }
    }

    private static final class EngineLease implements AutoCloseable {
        private final Engine engine;
        private final String backendName;
        private final boolean speculativeDecoding;
        private final String fallbackReason;
        private final boolean closeOnClose;
        private final boolean reused;
        private final WarmupState warmupState;
        private Conversation keepAliveConversation;

        private EngineLease(
                Engine engine,
                String backendName,
                boolean speculativeDecoding,
                String fallbackReason,
                boolean closeOnClose,
                boolean reused) {
            this(engine, backendName, speculativeDecoding, fallbackReason, closeOnClose, reused, new WarmupState());
        }

        private EngineLease(
                Engine engine,
                String backendName,
                boolean speculativeDecoding,
                String fallbackReason,
                boolean closeOnClose,
                boolean reused,
                WarmupState warmupState) {
            this.engine = engine;
            this.backendName = backendName;
            this.speculativeDecoding = speculativeDecoding;
            this.fallbackReason = fallbackReason;
            this.closeOnClose = closeOnClose;
            this.reused = reused;
            this.warmupState = warmupState;
        }

        private EngineLease asBorrowed(boolean reused) {
            EngineLease borrowed = new EngineLease(
                    engine,
                    backendName,
                    speculativeDecoding,
                    fallbackReason,
                    false,
                    reused,
                    warmupState);
            borrowed.keepAliveConversation = keepAliveConversation;
            return borrowed;
        }

        private void ensureKeepAliveConversation() {
            if (!keepAliveConversationEnabled() || keepAliveConversation != null) {
                return;
            }
            try {
                keepAliveConversation = engine.createConversation(daemonKeepAliveConversationConfig());
                daemonLog("engine keepalive conversation opened backend=%s", backendName);
            } catch (Throwable throwable) {
                daemonLog("engine keepalive conversation unavailable backend=%s reason=%s",
                        backendName,
                        summarize(throwable));
            }
        }

        private int remainingPrewarmIterations(int targetIterations) {
            return warmupState.remainingIterations(targetIterations);
        }

        private void markPrewarmed(int completedIterations) {
            warmupState.markCompleted(completedIterations);
        }

        @Override
        public void close() {
            if (!closeOnClose) {
                return;
            }
            closeKeepAliveConversation();
            try {
                engine.close();
            } catch (Throwable ignored) {
            }
        }

        private void forceClose() {
            closeKeepAliveConversation();
            try {
                engine.close();
            } catch (Throwable ignored) {
            }
        }

        private void closeKeepAliveConversation() {
            Conversation keepAlive = keepAliveConversation;
            keepAliveConversation = null;
            if (keepAlive == null) {
                return;
            }
            try {
                keepAlive.close();
            } catch (Throwable ignored) {
            }
        }
    }

    static final class ApproximateTokenStreamLimiter {
        private final int maxTokens;
        private final StringBuilder pendingWhitespace = new StringBuilder();
        private int emittedTokenCount;
        private boolean inToken;
        private boolean atLimit;

        ApproximateTokenStreamLimiter(int maxTokens) {
            this.maxTokens = Math.max(1, maxTokens);
        }

        String offer(String chunk) {
            if (chunk == null || chunk.isEmpty() || atLimit) {
                return "";
            }
            StringBuilder delta = new StringBuilder(chunk.length());
            for (int i = 0; i < chunk.length(); i++) {
                char ch = chunk.charAt(i);
                if (Character.isWhitespace(ch)) {
                    pendingWhitespace.append(ch);
                    inToken = false;
                    continue;
                }
                if (!inToken) {
                    int nextTokenCount = emittedTokenCount + 1;
                    if (nextTokenCount > maxTokens) {
                        pendingWhitespace.setLength(0);
                        atLimit = true;
                        break;
                    }
                    emittedTokenCount = nextTokenCount;
                    inToken = true;
                }
                if (pendingWhitespace.length() > 0) {
                    delta.append(pendingWhitespace);
                    pendingWhitespace.setLength(0);
                }
                delta.append(ch);
            }
            return delta.toString();
        }

        boolean atLimit() {
            return atLimit;
        }

        int emittedTokenCount() {
            return emittedTokenCount;
        }
    }

    static final class WarmupState {
        private int prewarmIterationsCompleted;

        synchronized int remainingIterations(int targetIterations) {
            return Math.max(0, targetIterations - prewarmIterationsCompleted);
        }

        synchronized void markCompleted(int completedIterations) {
            prewarmIterationsCompleted = Math.max(prewarmIterationsCompleted, completedIterations);
        }
    }

    static final class RouteResolutionCache {
        private final Map<String, RouteResolution> routes = new HashMap<>();
        private String modelIndexFingerprint = "";

        synchronized Optional<RouteResolution> resolve(FastArgs args) {
            if (!daemonRouteCacheEnabled()) {
                return resolveLiteRtLmRoute(args);
            }
            long startNanos = System.nanoTime();
            String currentIndexFingerprint = modelIndexFingerprint();
            if (!currentIndexFingerprint.equals(modelIndexFingerprint)) {
                routes.clear();
                modelIndexFingerprint = currentIndexFingerprint;
                daemonLog("route cache cleared after model index change");
            }
            String key = routeCacheKey(args, currentIndexFingerprint);
            RouteResolution cached = routes.get(key);
            if (cached != null && routeStillAvailable(cached)) {
                daemonLog("route cache hit source=%s artifact=%s path=%s",
                        cached.source(),
                        cached.selectedArtifact(),
                        cached.path().getFileName());
                return Optional.of(cached.withDurationNanos(System.nanoTime() - startNanos));
            }
            if (cached != null) {
                routes.remove(key);
                daemonLog("route cache evicted stale path=%s", cached.path().getFileName());
            }
            Optional<RouteResolution> resolved = resolveLiteRtLmRoute(args);
            resolved.ifPresent(route -> {
                routes.put(key, route);
                daemonLog("route cache stored source=%s artifact=%s path=%s",
                        route.source(),
                        route.selectedArtifact(),
                        route.path().getFileName());
            });
            return resolved;
        }

        synchronized int size() {
            return routes.size();
        }
    }

    private static final class EngineCache implements AutoCloseable {
        private final RouteResolutionCache routeCache = new RouteResolutionCache();
        private final Map<String, EngineLease> engines = new HashMap<>();

        private Optional<RouteResolution> resolveRoute(FastArgs args) {
            return routeCache.resolve(args);
        }

        private synchronized EngineLease acquire(Path modelPath, String requestedBackend, int engineMaxNumTokens) {
            String key = engineCacheKey(modelPath, requestedBackend, engineMaxNumTokens);
            String label = engineCacheLabel(modelPath, requestedBackend, engineMaxNumTokens);
            EngineLease existing = engines.get(key);
            if (existing != null) {
                daemonLog("engine cache hit %s", label);
                return existing.asBorrowed(true);
            }
            daemonLog("engine cache miss %s", label);
            try {
                EngineLease created = openEngine(modelPath, requestedBackend, engineMaxNumTokens, false, false);
                created.ensureKeepAliveConversation();
                engines.put(key, created);
                daemonLog("engine cache stored %s resolvedBackend=%s speculativeDecoding=%s",
                        label,
                        created.backendName,
                        created.speculativeDecoding);
                return created.asBorrowed(false);
            } catch (Throwable throwable) {
                daemonLog("engine cache open failed %s reason=%s", label, summarize(throwable));
                throw throwable;
            }
        }

        @Override
        public synchronized void close() {
            daemonLog("closing engine cache count=%d", engines.size());
            for (EngineLease engine : engines.values()) {
                engine.forceClose();
            }
            engines.clear();
        }
    }

    static boolean daemonRouteCacheEnabled() {
        return Boolean.parseBoolean(System.getProperty("gollek.litert.fast_run.daemon_route_cache", "true"));
    }

    private static boolean routeStillAvailable(RouteResolution route) {
        return route != null && route.path() != null && Files.isRegularFile(route.path());
    }

    private static String routeCacheKey(FastArgs args, String indexFingerprint) {
        return "model=" + nullToEmpty(args.model)
                + "|modelFile=" + nullToEmpty(args.modelFile)
                + "|cwd=" + Path.of("").toAbsolutePath().normalize()
                + "|home=" + System.getProperty("user.home", "")
                + "|autoEquivalent=" + equivalentLiteRtFastPathEnabled()
                + "|index=" + indexFingerprint;
    }

    private static String nullToEmpty(String value) {
        return value == null ? "" : value;
    }

    private static Path modelIndexPath() {
        return Path.of(System.getProperty("user.home"), ".gollek", "models", "index.json");
    }

    private static String modelIndexFingerprint() {
        Path indexPath = modelIndexPath();
        try {
            Path normalized = indexPath.toAbsolutePath().normalize();
            if (Files.isRegularFile(normalized)) {
                return normalized + "|size=" + Files.size(normalized)
                        + "|mtime=" + Files.getLastModifiedTime(normalized).toMillis();
            }
            return normalized + "|missing";
        } catch (Exception ignored) {
            return indexPath.toString() + "|unknown";
        }
    }

    private static String engineCacheKey(Path modelPath, String requestedBackend, int engineMaxNumTokens) {
        return modelPath.toAbsolutePath().normalize()
                + "|" + requestedBackend
                + "|" + engineMaxNumTokens
                + "|" + cacheDir()
                + "|" + System.getProperty("gollek.litert.fast_run.speculative_decoding", "");
    }

    private static String engineCacheLabel(Path modelPath, String requestedBackend, int engineMaxNumTokens) {
        return "model=" + modelPath.getFileName()
                + " backend=" + requestedBackend
                + " engineMaxNumTokens=" + engineMaxNumTokens
                + " cacheHash=" + Integer.toHexString(cacheDir().hashCode())
                + " spec=" + System.getProperty("gollek.litert.fast_run.speculative_decoding", "");
    }

    static final class FastArgs {
        private String model;
        private String modelFile;
        private String prompt;
        private int maxTokens = 256;
        private double temperature = 0.2d;
        private double topP = 0.9d;
        private int topK = 40;
        private String backend = System.getProperty("gollek.litert.fast_run.backend", "gpu");
        private boolean supported = true;
        private boolean banner = true;

        private boolean supported() {
            return supported && prompt != null && !prompt.isBlank()
                    && ((model != null && !model.isBlank()) || (modelFile != null && !modelFile.isBlank()));
        }

        static FastArgs parse(String[] args) {
            FastArgs parsed = new FastArgs();
            if (args.length == 0 || !"run".equals(args[0])) {
                parsed.supported = false;
                return parsed;
            }
            for (int i = 1; i < args.length; i++) {
                String arg = args[i];
                String value = null;
                int eq = arg.indexOf('=');
                if (arg.startsWith("--") && eq > 0) {
                    value = arg.substring(eq + 1);
                    arg = arg.substring(0, eq);
                }
                switch (arg) {
                    case "--no-banner", "--suppress-banner" -> parsed.banner = false;
                    case "--model", "-m" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.model = next.value();
                        i = next.index();
                    }
                    case "--modelFile", "--model-file", "--model-path" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.modelFile = next.value();
                        i = next.index();
                    }
                    case "--prompt", "-p" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.prompt = next.value();
                        i = next.index();
                    }
                    case "--max-tokens" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.maxTokens = parseInt(next.value(), parsed.maxTokens);
                        i = next.index();
                    }
                    case "--temperature" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.temperature = parseDouble(next.value(), parsed.temperature);
                        i = next.index();
                    }
                    case "--top-p" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.topP = parseDouble(next.value(), parsed.topP);
                        i = next.index();
                    }
                    case "--top-k" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.topK = parseInt(next.value(), parsed.topK);
                        i = next.index();
                    }
                    case "--backend" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.backend = next.value();
                        i = next.index();
                    }
                    case "--use-cpu" -> parsed.backend = "cpu";
                    case "--enable-cpu" -> {
                        // CPU fallback is already enabled by default for GPU init failures.
                    }
                    case "--platform" -> {
                        Value next = valueOrNext(value, args, i);
                        String platform = next.value();
                        i = next.index();
                        if ("cpu".equalsIgnoreCase(platform)) {
                            parsed.backend = "cpu";
                        } else if ("metal".equalsIgnoreCase(platform) || "gpu".equalsIgnoreCase(platform)) {
                            parsed.backend = "gpu";
                        } else {
                            parsed.supported = false;
                        }
                    }
                    case "--provider" -> {
                        Value next = valueOrNext(value, args, i);
                        String provider = next.value();
                        i = next.index();
                        if (provider != null && !provider.isBlank() && !"litert".equalsIgnoreCase(provider)) {
                            parsed.supported = false;
                        }
                    }
                    case "--format" -> {
                        Value next = valueOrNext(value, args, i);
                        String requestedFormat = next.value();
                        i = next.index();
                        if (requestedFormat != null
                                && !requestedFormat.isBlank()
                                && !isLiteRtFormat(requestedFormat)) {
                            parsed.supported = false;
                        }
                    }
                    case "--engine", "--gguf-engine" -> {
                        Value next = valueOrNext(value, args, i);
                        i = next.index();
                    }
                    case "--stream" -> {
                        String stream = value != null ? value : "true";
                        if ("false".equalsIgnoreCase(stream)) {
                            parsed.supported = false;
                        }
                    }
                    case "--no-cache", "--offline", "--local" -> {
                        // Safe no-op for the local fast path.
                    }
                    default -> {
                        if (arg.startsWith("-")) {
                            parsed.supported = false;
                        }
                    }
                }
            }
            return parsed;
        }

        private record Value(String value, int index) {
        }

        private static Value valueOrNext(String inline, String[] args, int currentIndex) {
            if (inline != null) {
                return new Value(inline, currentIndex);
            }
            int nextIndex = currentIndex + 1;
            if (nextIndex >= args.length) {
                return new Value("", currentIndex);
            }
            return new Value(args[nextIndex], nextIndex);
        }

        private static int parseInt(String value, int fallback) {
            try {
                return Math.max(1, Integer.parseInt(value));
            } catch (Exception ignored) {
                return fallback;
            }
        }

        private static double parseDouble(String value, double fallback) {
            try {
                return Double.parseDouble(value);
            } catch (Exception ignored) {
                return fallback;
            }
        }
    }
}
