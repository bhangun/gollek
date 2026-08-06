package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;
import tech.kayys.gollek.observability.ObservabilityManager;
import tech.kayys.gollek.metrics.MetricsRegistry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;
import io.opentelemetry.api.common.AttributeKey;

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
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

import tech.kayys.alkhawarizm.gguf.runtime.GgufRuntimeProfile;
import tech.kayys.alkhawarizm.gguf.runtime.GgufRuntimeProbe;

/**
 * Small GGUF/llama.cpp fast path used by the local macOS shim for simple
 * {@code gollek run} calls.
 */
public final class GgufFastRun {
    private static final int FALLBACK_TO_FULL_CLI = 42;
    private static final int COMMAND_ERROR = 2;
    private static final String DAEMON_MAGIC = "GOLLEK_GGUF_FAST_DAEMON_V1";
    private static final String DAEMON_STOP_MAGIC = "GOLLEK_GGUF_FAST_DAEMON_STOP_V1";
    private static final String DAEMON_PREWARM_MAGIC = "GOLLEK_GGUF_FAST_DAEMON_PREWARM_V1";
    private static final long MIN_OUTPUT_BUFFER_BYTES = 64L * 1024L;
    private static final int OUTPUT_BUFFER_RETRY_LIMIT = 2;
    private static final ThreadLocal<Boolean> REQUEST_TIMING = new ThreadLocal<>();

    private GgufFastRun() {
    }

    static boolean isFallbackToFullCliStatus(int status) {
        return status == FALLBACK_TO_FULL_CLI;
    }

    public static void main(String[] args) {
        if (args.length > 0 && "__daemon".equals(args[0])) {
            hardExitProcess(runDaemon());
            return;
        }
        if (args.length > 0 && ("__daemon-stop".equals(args[0]) || "__gguf-daemon-stop".equals(args[0]))) {
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
        if (hardExitAfterRun()) {
            hardExitProcess(0);
        }
    }

    static int run(String[] args) {
        try {
            FastArgs parsed = FastArgs.parse(args);
            if (!parsed.supported()) {
                return FALLBACK_TO_FULL_CLI;
            }
            if (hasKnownNonGgufTarget(parsed)) {
                return FALLBACK_TO_FULL_CLI;
            }
            Optional<Path> modelPath = resolveGgufModel(parsed);
            if (modelPath.isEmpty()) {
                return FALLBACK_TO_FULL_CLI;
            }
            EngineMode engine = parsed.engineMode();
            if (daemonEnabled() && engine != EngineMode.JAVA && engine != EngineMode.BENCHMARK) {
                OptionalInt daemonStatus = requestDaemon(args);
                if (daemonStatus.isPresent()) {
                    return daemonStatus.getAsInt();
                }
            }
            return generate(modelPath.get(), parsed);
        } catch (Throwable throwable) {
            if (fastRunDebug()) {
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
            if (parsed.hasNonGgufProvider()) {
                return FALLBACK_TO_FULL_CLI;
            }
            if (!parsed.supported()) {
                printPrewarmUsage(System.err);
                return COMMAND_ERROR;
            }
            EngineMode engine = parsed.engineMode();
            if (engine == EngineMode.JAVA || engine == EngineMode.BENCHMARK) {
                System.err.println("GGUF daemon prewarm requires the llama.cpp fast path.");
                return COMMAND_ERROR;
            }
            if (resolveGgufModel(parsed).isEmpty()) {
                if (hasKnownNonGgufTarget(parsed)) {
                    return FALLBACK_TO_FULL_CLI;
                }
                System.err.println("GGUF daemon prewarm could not resolve a GGUF model.");
                return COMMAND_ERROR;
            }
            OptionalInt status = requestDaemonPrewarm(runArgs);
            return status.orElse(COMMAND_ERROR);
        } catch (Throwable throwable) {
            System.err.println("GGUF daemon prewarm failed: " + briefMessage(throwable));
            if (fastRunDebug()) {
                throwable.printStackTrace(System.err);
            }
            return COMMAND_ERROR;
        }
    }

    private static boolean hasKnownNonGgufTarget(FastArgs args) {
        if (args.modelFile != null && hasNonGgufExtension(args.modelFile)) {
            return true;
        }
        if (args.model != null && hasNonGgufExtension(args.model)) {
            return true;
        }
        if (args.model != null) {
            Optional<Path> direct = resolvePath(Path.of(args.model));
            if (direct.isPresent()) {
                return false;
            }
            try {
                Optional<LocalModelIndex.Entry> indexed = LocalModelIndex.find(args.model);
                if (indexed.isPresent()) {
                    String format = indexed.get().format;
                    if (format != null && !format.isBlank() && !"gguf".equals(format.trim().toLowerCase(Locale.ROOT))) {
                        return !equivalentGgufFastPathEnabled()
                                || findEquivalentGgufModelPath(indexed.get()).isEmpty();
                    }
                }
            } catch (Throwable ignored) {
                // The full CLI can still resolve unknown references after this lightweight fallback.
            }
        }
        return false;
    }

    private static boolean hasNonGgufExtension(String value) {
        String normalized = value.replace('\\', '/').toLowerCase(Locale.ROOT);
        return normalized.endsWith(".litertlm")
                || normalized.endsWith(".task")
                || normalized.endsWith(".tflite")
                || normalized.endsWith(".safetensors")
                || normalized.endsWith(".safetensor")
                || normalized.endsWith(".onnx")
                || normalized.endsWith(".pt")
                || normalized.endsWith(".pth")
                || normalized.endsWith(".bin");
    }

    private static void printPrewarmUsage(PrintStream out) {
        out.println("Usage: gollek prewarm --model MODEL [--prompt TEXT] [--max-tokens N] [--backend metal|cpu]");
        out.println("       gollek warmup  --model MODEL [--prompt TEXT] [--max-tokens N] [--backend metal|cpu]");
        out.println();
        out.println("Preloads the local llama.cpp GGUF daemon and builds the first prompt cache so the next");
        out.println("matching gollek run can reuse the Metal/CPU session instead of paying cold-start cost.");
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
        boolean hasEngine = false;
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
            } else if (option.equals("--engine") || option.equals("--gguf-engine")
                    || option.equals("--llamacpp") || option.equals("--llama-cpp")) {
                hasEngine = true;
            }
        }
        if (!hasPrompt) {
            normalized.add("--prompt");
            normalized.add(firstNonBlank(
                    System.getProperty("gollek.gguf.fast_run.prewarm_prompt"),
                    System.getenv("GOLLEK_GGUF_FAST_PREWARM_PROMPT"),
                    "where is jakarta"));
        }
        if (!hasMaxTokens) {
            normalized.add("--max-tokens");
            normalized.add(firstNonBlank(
                    System.getProperty("gollek.gguf.fast_run.prewarm_context_tokens"),
                    System.getenv("GOLLEK_GGUF_FAST_PREWARM_CONTEXT_TOKENS"),
                    "10"));
        }
        if (!hasProvider) {
            normalized.add("--provider");
            normalized.add("gguf");
        }
        if (!hasEngine) {
            normalized.add("--engine");
            normalized.add("auto");
        }
        return normalized.toArray(String[]::new);
    }

    private static boolean isPrewarmCommand(String arg) {
        return "__gguf-daemon-prewarm".equals(arg)
                || "prewarm".equals(arg)
                || "warmup".equals(arg);
    }

    static Optional<Path> resolveGgufModel(FastArgs args) {
        if (args.modelFile != null) {
            Optional<Path> explicit = resolvePath(Path.of(args.modelFile));
            if (explicit.isPresent()) {
                return findGguf(explicit.get());
            }
        }
        if (args.model != null) {
            Optional<Path> direct = resolvePath(Path.of(args.model));
            if (direct.isPresent()) {
                Optional<Path> found = findGguf(direct.get());
                if (found.isPresent()) {
                    return found;
                }
            }
            try {
                Optional<LocalModelIndex.Entry> indexed = LocalModelIndex.find(args.model);
                if (indexed.isPresent() && indexed.get().path != null && !indexed.get().path.isBlank()) {
                    Optional<Path> indexedGguf = findGguf(Path.of(indexed.get().path));
                    if (indexedGguf.isPresent()) {
                        return indexedGguf;
                    }
                    if (equivalentGgufFastPathEnabled()) {
                        return findEquivalentGgufModelPath(indexed.get());
                    }
                }
            } catch (Throwable ignored) {
                // Full CLI has the complete resolver if the lightweight index read fails.
            }
        }
        return Optional.empty();
    }

    private static Optional<Path> findEquivalentGgufModelPath(LocalModelIndex.Entry requested) {
        if (!canUseGgufEquivalentFor(requested)) {
            return Optional.empty();
        }
        var requestedKeys = canonicalModelKeys(requested);
        if (requestedKeys.isEmpty()) {
            return Optional.empty();
        }
        try {
            return LocalModelIndex.entries().stream()
                    .filter(candidate -> candidate != null && candidate.path != null && !candidate.path.isBlank())
                    .filter(candidate -> !sameIndexPath(candidate, requested))
                    .filter(GgufFastRun::isGgufIndexEntry)
                    .filter(candidate -> compatibleArchitecture(requested, candidate))
                    .filter(candidate -> intersects(requestedKeys, canonicalModelKeys(candidate)))
                    .sorted((left, right) -> Integer.compare(
                            equivalentGgufPreference(left),
                            equivalentGgufPreference(right)))
                    .map(candidate -> Path.of(candidate.path))
                    .map(GgufFastRun::findGguf)
                    .flatMap(Optional::stream)
                    .findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static boolean canUseGgufEquivalentFor(LocalModelIndex.Entry requested) {
        if (requested == null || isGgufIndexEntry(requested)) {
            return false;
        }
        String format = requested.format == null ? "" : requested.format.trim().toLowerCase(Locale.ROOT);
        if (!format.isBlank()
                && !format.equals("safetensor")
                && !format.equals("safetensors")) {
            return false;
        }
        String architecture = requested.architecture == null
                ? ""
                : requested.architecture.trim().toLowerCase(Locale.ROOT);
        String identity = ((requested.id == null ? "" : requested.id) + " "
                + (requested.name == null ? "" : requested.name) + " "
                + (requested.path == null ? "" : requested.path)).toLowerCase(Locale.ROOT);
        return architecture.contains("gemma") || identity.contains("gemma");
    }

    private static boolean isGgufIndexEntry(LocalModelIndex.Entry entry) {
        if (entry == null) {
            return false;
        }
        String format = entry.format == null ? "" : entry.format.trim().toLowerCase(Locale.ROOT);
        if (format.equals("gguf")) {
            return true;
        }
        return entry.path != null && entry.path.toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    private static boolean compatibleArchitecture(LocalModelIndex.Entry requested, LocalModelIndex.Entry candidate) {
        String requestedArchitecture = normalizedArchitecture(requested.architecture);
        String candidateArchitecture = normalizedArchitecture(candidate.architecture);
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

    private static List<String> canonicalModelKeys(LocalModelIndex.Entry entry) {
        List<String> keys = new ArrayList<>();
        if (entry == null) {
            return keys;
        }
        addCanonicalModelKey(keys, entry.id);
        addCanonicalModelKey(keys, entry.name);
        addCanonicalModelKey(keys, entry.path);
        return keys;
    }

    private static void addCanonicalModelKey(List<String> keys, String value) {
        String key = LiteRtLmFastRun.canonicalModelKey(value);
        if (!key.isBlank() && !keys.contains(key)) {
            keys.add(key);
        }
    }

    private static boolean sameIndexPath(LocalModelIndex.Entry left, LocalModelIndex.Entry right) {
        if (left == null || right == null || left.path == null || right.path == null) {
            return false;
        }
        return left.path.equals(right.path);
    }

    private static boolean intersects(List<String> left, List<String> right) {
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

    private static int equivalentGgufPreference(LocalModelIndex.Entry entry) {
        String name = ((entry.name == null ? "" : entry.name) + " "
                + (entry.path == null ? "" : entry.path)).toLowerCase(Locale.ROOT);
        int score = 0;
        if (!name.endsWith(".gguf")) {
            score += 10;
        }
        if (name.contains("q4_k_m")) {
            score -= 5;
        } else if (name.contains("q4")) {
            score -= 3;
        } else if (name.contains("q5")) {
            score += 2;
        } else if (name.contains("q8")) {
            score += 8;
        } else if (name.contains("f16") || name.contains("fp16") || name.contains("bf16")) {
            score += 20;
        }
        return score;
    }

    static boolean equivalentGgufFastPathEnabled() {
        String configured = configValue("gollek.gguf.fast_run.auto_equivalent");
        return configured == null || configured.isBlank() || Boolean.parseBoolean(configured);
    }

    private static Optional<Path> resolvePath(Path raw) {
        Path expanded = expandHome(raw);
        if (Files.exists(expanded)) {
            return Optional.of(expanded.toAbsolutePath().normalize());
        }
        return Optional.empty();
    }

    private static Path expandHome(Path path) {
        String text = path.toString();
        if (text.equals("~")) {
            return Path.of(System.getProperty("user.home"));
        }
        if (text.startsWith("~/")) {
            return Path.of(System.getProperty("user.home"), text.substring(2));
        }
        return path;
    }

    private static Optional<Path> findGguf(Path path) {
        try {
            if (Files.isRegularFile(path) && isGguf(path)) {
                return Optional.of(path.toAbsolutePath().normalize());
            }
            Path searchDir = Files.isDirectory(path) ? path : path.getParent();
            if (searchDir == null || !Files.isDirectory(searchDir)) {
                return Optional.empty();
            }
            try (var stream = Files.list(searchDir)) {
                return stream
                        .filter(Files::isRegularFile)
                        .filter(GgufFastRun::isGguf)
                        .findFirst()
                        .map(candidate -> candidate.toAbsolutePath().normalize());
            }
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static boolean isGguf(Path path) {
        return path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    private static int generate(Path modelPath, FastArgs args) throws Throwable {
        return generate(modelPath, args, System.out, System.err, null, "fast path");
    }

    private static int generate(
            Path modelPath,
            FastArgs args,
            PrintStream out,
            PrintStream err,
            GgufSessionCache sessionCache,
            String runnerName) throws Throwable {
        EngineMode engine = args.engineMode();
        if (engine == EngineMode.JAVA) {
            try {
                System.out.println("Executing Java Native GGUF Engine...");
                tech.kayys.gollek.plugin.runner.gguf.JavaNativeGgufBackend backend = new tech.kayys.gollek.plugin.runner.gguf.JavaNativeGgufBackend(modelPath);
                
                tech.kayys.gollek.plugin.runner.RunnerRequest request = new tech.kayys.gollek.plugin.runner.RunnerRequest(tech.kayys.gollek.plugin.runner.RequestType.INFER);
                
                tech.kayys.gollek.plugin.runner.RunnerResult<?> result = backend.execute(request);
                
                if (result.isSuccess()) {
                    System.out.println(result.getData());
                } else {
                    System.err.println("Execution failed: " + result.getErrorMessage().orElse("Unknown error"));
                }
                return 0;
            } catch (Throwable e) {
                System.err.println("Java Native Engine crashed:");
                e.printStackTrace(System.err);
                return COMMAND_ERROR;
            }
        }
        if (engine == EngineMode.BENCHMARK) {
            out.println("GGUF engine benchmark: Java-native loader/probe vs llama.cpp generation fallback.");
            printJavaNativeProbe(modelPath, "Java-native GGUF loader");
        }
        generateWithLlamaCpp(modelPath, args, engine == EngineMode.BENCHMARK, out, err, sessionCache, runnerName);
        return 0;
    }

    private static void printJavaNativeProbe(Path modelPath, String label) {
        try {
            int probeRows = Math.max(1, Integer.getInteger("gollek.gguf.java_probe_rows", 16));
            int probeMatVecRows = Math.max(1, Integer.getInteger("gollek.gguf.java_probe_matvec_rows", 4096));
            GgufRuntimeProbe probe = GgufRuntimeProbe.load(modelPath, probeRows, probeMatVecRows);
            printProfile(label, probe.profile());
            printRowDotProbe(label, probe);
        } catch (Throwable throwable) {
            System.out.printf("%s: unavailable, reason=%s.%n", label, briefMessage(throwable));
            if (fastRunDebug()) {
                throwable.printStackTrace(System.err);
            }
        }
    }

    private static void generateWithLlamaCpp(
            Path modelPath,
            FastArgs args,
            boolean benchmarkMode,
            PrintStream out,
            PrintStream err,
            GgufSessionCache sessionCache,
            String runnerName) throws Throwable {
        long startNanos = System.nanoTime();
        String requestedBackend = normalizedBackend(args.backend);
        int nGpuLayers = "cpu".equals(requestedBackend)
                ? 0
                : intConfig("gollek.gguf.fast_run.gpu_layers", -1);
        int threads = Math.max(1, intConfig("gollek.gguf.fast_run.threads", defaultThreads()));
        int batchThreads = Math.max(1, intConfig("gollek.gguf.fast_run.batch_threads", threads));
        String prompt = formatPromptForModel(args.prompt, modelPath);
        int context = fastRunContext(prompt, args.maxTokens);
        int batch = boundedBatch("gollek.gguf.fast_run.batch", Math.min(context, 1024), context);
        int microBatch = boundedBatch("gollek.gguf.fast_run.ubatch", Math.min(batch, 512), batch);
        boolean swaFull = fastRunSwaFull();
        printRunHeader(modelPath, args, out);

        try {
            generateWithLlamaCppSession(
                    modelPath,
                    args,
                    benchmarkMode,
                    startNanos,
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    nGpuLayers,
                    swaFull,
                    prompt,
                    false,
                    out,
                    err,
                    sessionCache,
                    runnerName);
        } catch (Throwable throwable) {
            if (nGpuLayers == 0 || !fastRunCpuFallback()) {
                throw throwable;
            }
            err.printf("WARN: GGUF Metal fast path failed (%s); retrying with CPU fallback.%n",
                    briefMessage(throwable));
            generateWithLlamaCppSession(
                    modelPath,
                    args,
                    benchmarkMode,
                    startNanos,
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    0,
                    swaFull,
                    prompt,
                    true,
                    out,
                    err,
                    sessionCache,
                    runnerName);
        }
    }

    private static int prewarmWithLlamaCpp(
            Path modelPath,
            FastArgs args,
            PrintStream out,
            PrintStream err,
            GgufSessionCache sessionCache) throws Throwable {
        long startNanos = System.nanoTime();
        String requestedBackend = normalizedBackend(args.backend);
        int nGpuLayers = "cpu".equals(requestedBackend)
                ? 0
                : intConfig("gollek.gguf.fast_run.gpu_layers", -1);
        int threads = Math.max(1, intConfig("gollek.gguf.fast_run.threads", defaultThreads()));
        int batchThreads = Math.max(1, intConfig("gollek.gguf.fast_run.batch_threads", threads));
        String prompt = formatPromptForModel(args.prompt, modelPath);
        int context = fastRunPrewarmContext(prompt, args.maxTokens);
        int batch = boundedBatch("gollek.gguf.fast_run.batch", Math.min(context, 1024), context);
        int microBatch = boundedBatch("gollek.gguf.fast_run.ubatch", Math.min(batch, 512), batch);
        boolean swaFull = fastRunSwaFull();

        try {
            prewarmWithLlamaCppSession(
                    modelPath,
                    args,
                    startNanos,
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    nGpuLayers,
                    swaFull,
                    prompt,
                    false,
                    out,
                    err,
                    sessionCache);
            return 0;
        } catch (Throwable throwable) {
            if (nGpuLayers == 0 || !fastRunCpuFallback()) {
                throw throwable;
            }
            err.printf("WARN: GGUF Metal prewarm failed (%s); retrying with CPU fallback.%n",
                    briefMessage(throwable));
            prewarmWithLlamaCppSession(
                    modelPath,
                    args,
                    startNanos,
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    0,
                    swaFull,
                    prompt,
                    true,
                    out,
                    err,
                    sessionCache);
            return 0;
        }
    }

    private static void prewarmWithLlamaCppSession(
            Path modelPath,
            FastArgs args,
            long startNanos,
            int context,
            int batch,
            int microBatch,
            int threads,
            int batchThreads,
            int nGpuLayers,
            boolean swaFull,
            String prompt,
            boolean cpuFallback,
            PrintStream out,
            PrintStream err,
            GgufSessionCache sessionCache) throws Throwable {
        long openStartNanos = System.nanoTime();
        boolean useMmap = booleanConfig("gollek.gguf.fast_run.mmap", true);
        boolean useMlock = booleanConfig("gollek.gguf.fast_run.mlock", false);
        SessionLease opened = sessionCache == null
                ? SessionLease.open(modelPath, context, batch, microBatch, threads, batchThreads, nGpuLayers,
                        useMmap, useMlock, swaFull)
                : sessionCache.acquire(modelPath, context, batch, microBatch, threads, batchThreads, nGpuLayers,
                        useMmap, useMlock, swaFull);
        long openNanos = System.nanoTime() - openStartNanos;
        try (SessionLease lease = opened) {
            NativeGgufSession session = lease.session();
            int tokens = prewarmTokenCount();
            String sessionLabel = sessionReuseLabel(sessionCache != null, lease.reused(), lease.reuseKind());
            String coverage = sessionCoverageSummary(context, batch, microBatch);
            long generateStartNanos = System.nanoTime();
            session.generate(prompt, tokens, 0.0d, 1, 1.0d);
            long generateNanos = System.nanoTime() - generateStartNanos;
            long durationMs = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
            out.printf("Prewarmed llama.cpp GGUF daemon for %s "
                            + "(backend=%s, nGpuLayers=%d, threads=%d, context=%d, batch=%d, ubatch=%d, "
                            + "tokens=%d, session=%s, coverage=%s, cpuFallback=%s).%n",
                    modelPath.getFileName(),
                    session.backendName(),
                    nGpuLayers,
                    threads,
                    context,
                    batch,
                    microBatch,
                    tokens,
                    sessionLabel,
                    coverage,
                    cpuFallback);
            out.printf("[GGUF daemon prewarm, Duration: %.2fs]%n", durationMs / 1000.0d);
            if (fastRunTiming()) {
                String nativeMetrics = session.metrics().map(metrics -> ", native={" + metrics + "}").orElse("");
                err.printf("GGUF timing: open=%.3fms, generateCall=%.3fms%s%n",
                        openNanos / 1_000_000.0d,
                        generateNanos / 1_000_000.0d,
                        nativeMetrics);
            }
            if (shouldHardExitAfterNativeRun(sessionCache != null)) {
                hardExitProcess(0);
            }
        }
    }

    static int prewarmTokenCount() {
        // Two tokens force one post-sample decode, which warms the hot decode path.
        return Math.max(2, intConfig("gollek.gguf.fast_run.prewarm_tokens", 2));
    }

    private static void generateWithLlamaCppSession(
            Path modelPath,
            FastArgs args,
            boolean benchmarkMode,
            long startNanos,
            int context,
            int batch,
            int microBatch,
            int threads,
            int batchThreads,
            int nGpuLayers,
            boolean swaFull,
            String prompt,
            boolean cpuFallback,
            PrintStream out,
            PrintStream err,
            GgufSessionCache sessionCache,
            String runnerName) throws Throwable {
        long openStartNanos = System.nanoTime();
        boolean useMmap = booleanConfig("gollek.gguf.fast_run.mmap", true);
        boolean useMlock = booleanConfig("gollek.gguf.fast_run.mlock", false);
        SessionLease opened = sessionCache == null
                ? SessionLease.open(modelPath, context, batch, microBatch, threads, batchThreads, nGpuLayers,
                        useMmap, useMlock, swaFull)
                : sessionCache.acquire(modelPath, context, batch, microBatch, threads, batchThreads, nGpuLayers,
                        useMmap, useMlock, swaFull);
        long openNanos = System.nanoTime() - openStartNanos;
        try (SessionLease lease = opened) {
            NativeGgufSession session = lease.session();
            String sessionLabel = sessionReuseLabel(sessionCache != null, lease.reused(), lease.reuseKind());
            String warmSuffix = ", session=" + sessionLabel;
            printExecutionRoute(out, session.backendName(), runnerName);
            if (!fastRunQuiet()) {
                out.printf("Using llama.cpp GGUF %s for %s "
                                + "(backend=%s, nGpuLayers=%d, threads=%d, context=%d, batch=%d, ubatch=%d, "
                                + "swaFull=%s, cpuFallback=%s%s).%n",
                        runnerName,
                        modelPath.getFileName(),
                        session.backendName(),
                        nGpuLayers,
                        threads,
                        context,
                        batch,
                        microBatch,
                        swaFull,
                        cpuFallback,
                        warmSuffix);
            }
            long generateStartNanos = System.nanoTime();
            String output = session.generate(prompt, args.maxTokens, args.temperature, args.topK, args.topP);
            long afterGenerateNanos = System.nanoTime();
            long generateNanos = afterGenerateNanos - generateStartNanos;
            out.print(output);
            String label = benchmarkMode ? "llama.cpp fallback" : "Fast GGUF";
            Optional<String> nativeMetrics = safeMetrics(
                    session,
                    sessionCache != null && lease.reused(),
                    lease.reuseKind());
            if (!fastRunQuiet()) {
                printFastRunStats(out, label, session.lastGeneratedTokens(), startNanos, openNanos, generateNanos,
                        afterGenerateNanos, nativeMetrics);
            }
            if (fastRunTiming()) {
                String timingNativeMetrics = nativeMetrics.map(metrics -> ", native={" + metrics + "}").orElse("");
                err.printf("GGUF timing: open=%.3fms, generateCall=%.3fms%s%n",
                        openNanos / 1_000_000.0d,
                        generateNanos / 1_000_000.0d,
                        timingNativeMetrics);
            }
        }
    }

    private static Optional<String> safeMetrics(NativeGgufSession session) {
        return safeMetrics(session, false, "cold");
    }

    private static Optional<String> safeMetrics(NativeGgufSession session, boolean warmSession, String reuseKind) {
        try {
            return session.metrics(warmSession, reuseKind);
        } catch (Throwable ignored) {
            return Optional.empty();
        }
    }

    static String sessionReuseLabel(boolean daemonSession, boolean reused, String reuseKind) {
        if (!daemonSession) {
            return "one-shot";
        }
        if (!reused) {
            return "cold";
        }
        String normalized = reuseKind == null ? "" : reuseKind.trim().toLowerCase(Locale.ROOT);
        if (normalized.equals("compatible")) {
            return "warm-compatible";
        }
        if (normalized.equals("exact")) {
            return "warm-exact";
        }
        return "warm";
    }

    static String sessionCoverageSummary(int context, int batch, int microBatch) {
        return "ctx<=" + Math.max(1, context)
                + ",batch<=" + Math.max(1, batch)
                + ",ubatch<=" + Math.max(1, microBatch);
    }

    static long outputBufferBytesForTokens(int maxTokens) {
        long requested = Math.max(1L, (long) maxTokens) * 512L;
        return Math.max(MIN_OUTPUT_BUFFER_BYTES, requested);
    }

    static long nextOutputBufferBytes(long currentBytes) {
        long current = Math.max(MIN_OUTPUT_BUFFER_BYTES, currentBytes);
        if (current > Long.MAX_VALUE / 2L) {
            return Long.MAX_VALUE;
        }
        return Math.max(MIN_OUTPUT_BUFFER_BYTES, current * 2L);
    }

    static boolean outputBufferTooSmall(int nativeStatus, String message) {
        return nativeStatus == -9
                || (message != null && message.toLowerCase(Locale.ROOT).contains("output buffer too small"));
    }

    static void printFastRunStats(
            PrintStream out,
            String label,
            int outputTokens,
            long startNanos,
            long openNanos,
            long generateNanos,
            long endNanos) {
        printFastRunStats(out, label, outputTokens, startNanos, openNanos, generateNanos, endNanos,
                Optional.empty());
    }

    static void printFastRunStats(
            PrintStream out,
            String label,
            int outputTokens,
            long startNanos,
            long openNanos,
            long generateNanos,
            long endNanos,
            Optional<String> nativeMetrics) {
        long durationNanos = Math.max(0L, endNanos - startNanos);
        long durationMs = Duration.ofNanos(durationNanos).toMillis();
        double durationSeconds = durationMs / 1000.0d;
        double totalSpeed = durationSeconds > 0.0d && outputTokens > 0 ? outputTokens / durationSeconds : 0.0d;
        double generateMs = openOrGenerateMillis(generateNanos);
        double generateSeconds = generateNanos / 1_000_000_000.0d;
        double generationSpeed = generateSeconds > 0.0d && outputTokens > 0 ? outputTokens / generateSeconds : 0.0d;
        double tokenLatencyMs = outputTokens > 0 ? generateMs / outputTokens : 0.0d;

        out.printf("%n[%s, Duration: %.2fs, Speed: %.2f t/s]%n", label, durationSeconds, totalSpeed);
        out.println("Performance Metrics:");
        out.printf("  open time      = %9.2f ms%n", openOrGenerateMillis(openNanos));
        out.printf("  generate call  = %9.2f ms%n", generateMs);
        out.printf("  generation     = %9.2f t/s (%d tokens)%n", generationSpeed, Math.max(0, outputTokens));
        out.printf("  token latency  = %9.2f ms/token%n", tokenLatencyMs);
        printNativeMetrics(out, nativeMetrics);
        out.println();

        try {
            ObservabilityManager obs = ObservabilityManager.global();
            if (obs != null) {
                MetricsRegistry reg = obs.getMetricsRegistry();
                AttributesBuilder attrs = Attributes.builder()
                        .put("engine", "gguf")
                        .put("fast_path", true);

                if (nativeMetrics.isPresent()) {
                    Map<String, String> m = parseNativeMetrics(nativeMetrics.get());
                    String backend = m.get("backend");
                    if (backend != null && !backend.isBlank()) {
                        attrs.put("native_backend", backend);
                    }
                    if (m.containsKey("gpuLayers")) attrs.put("gpuLayers", Long.parseLong(m.get("gpuLayers")));
                    if (m.containsKey("threads")) attrs.put("threads", Long.parseLong(m.get("threads")));
                    if (m.containsKey("context")) attrs.put("context", Long.parseLong(m.get("context")));
                    if (m.containsKey("batch")) attrs.put("batch", Long.parseLong(m.get("batch")));
                    
                    if (m.containsKey("modelLoad")) reg.record("gollek.llm.native.load", Double.parseDouble(m.get("modelLoad")), attrs.build());
                    if (m.containsKey("contextInit")) reg.record("gollek.llm.native.context_init", Double.parseDouble(m.get("contextInit")), attrs.build());
                    if (m.containsKey("tokenize")) reg.record("gollek.llm.native.tokenize", Double.parseDouble(m.get("tokenize")), attrs.build());
                    if (m.containsKey("prefill")) reg.record("gollek.llm.native.prefill", Double.parseDouble(m.get("prefill")), attrs.build());
                    if (m.containsKey("decode")) reg.record("gollek.llm.native.decode", Double.parseDouble(m.get("decode")), attrs.build());
                }

                Attributes finalAttrs = attrs.build();
                reg.record("gollek.llm.generation.duration", durationMs, finalAttrs);
                reg.record("gollek.llm.generation.speed", totalSpeed, finalAttrs);
                reg.record("gollek.llm.generation.token_latency", tokenLatencyMs, finalAttrs);
                reg.record("gollek.llm.model.open_time", openOrGenerateMillis(openNanos), finalAttrs);
            }
        } catch (Exception e) {
            // Ignore telemetry exceptions
        }
    }

    private static double openOrGenerateMillis(long nanos) {
        return Math.max(0L, nanos) / 1_000_000.0d;
    }

    private static void printNativeMetrics(PrintStream out, Optional<String> nativeMetrics) {
        if (nativeMetrics.isEmpty()) {
            return;
        }
        Map<String, String> metrics = parseNativeMetrics(nativeMetrics.get());
        if (metrics.isEmpty()) {
            return;
        }
        String backend = metrics.get("backend");
        if (backend != null && !backend.isBlank()) {
            out.printf("  native backend = %s%s%n", backend, nativeRuntimeShape(metrics));
        }
        out.printf("  native load    = %9.2f ms, context init = %.2f ms%n",
                nativeMillis(metrics, "modelLoad"),
                nativeMillis(metrics, "contextInit"));
        out.printf("  native tokenize= %9.2f ms (%s)%n",
                nativeMillis(metrics, "tokenize"),
                metrics.getOrDefault("tokenizeCache", "unknown"));
        out.printf("  native prompt  = %9.2f ms (%s tokens)%n",
                nativeMillis(metrics, "prefill"),
                metrics.getOrDefault("promptTokens", "0"));
        out.printf("  native decode  = %9.2f ms (%s sampled, %s decoded)%n",
                nativeMillis(metrics, "decode"),
                metrics.getOrDefault("generatedTokens", "0"),
                metrics.getOrDefault("decodedTokens", "0"));
        String sampler = metrics.get("sampler");
        if (sampler != null && !sampler.isBlank()) {
            out.printf("  native sampler = %s, %.2f ms%n",
                    sampler,
                    nativeMillis(metrics, "samplerMs"));
        }
        String warmSession = metrics.get("warmSession");
        if (warmSession != null && !warmSession.isBlank()) {
            String reuseKind = metrics.getOrDefault("sessionReuse", "").trim();
            out.printf("  native session = %s%s%n",
                    Boolean.parseBoolean(warmSession) ? "warm" : "cold",
                    reuseKind.isBlank() ? "" : " (" + reuseKind + ")");
        }
        out.printf("  native cache   = %s, %.2f ms%s%n",
                metrics.getOrDefault("promptCache", "unknown"),
                nativeMillis(metrics, "promptCacheMs"),
                nativePromptCacheFlags(metrics));
        printNativeOutputMetrics(out, metrics);
    }

    private static String nativePromptCacheFlags(Map<String, String> metrics) {
        List<String> flags = new ArrayList<>();
        if (Boolean.parseBoolean(metrics.getOrDefault("promptCacheEagerShort", "false"))) {
            flags.add("eager-short");
        }
        if (Boolean.parseBoolean(metrics.getOrDefault("repeatedPrompt", "false"))) {
            flags.add("repeat");
        }
        return flags.isEmpty() ? "" : " (" + String.join(", ", flags) + ")";
    }

    private static void printNativeOutputMetrics(PrintStream out, Map<String, String> metrics) {
        String outputBytes = metrics.get("outputBytes");
        String javaBufferBytes = metrics.get("javaOutputBufferBytes");
        String javaRetries = metrics.get("javaOutputRetries");
        if ((outputBytes == null || outputBytes.isBlank())
                && (javaBufferBytes == null || javaBufferBytes.isBlank())
                && (javaRetries == null || javaRetries.isBlank())) {
            return;
        }
        out.printf("  native output  = %s bytes, java buffer = %s bytes, retries = %s%n",
                outputBytes == null || outputBytes.isBlank() ? "unknown" : outputBytes,
                javaBufferBytes == null || javaBufferBytes.isBlank() ? "unknown" : javaBufferBytes,
                javaRetries == null || javaRetries.isBlank() ? "0" : javaRetries);
    }

    private static String nativeRuntimeShape(Map<String, String> metrics) {
        List<String> parts = new ArrayList<>();
        addNativeRuntimePart(parts, "gpuLayers", metrics.get("gpuLayers"));
        addNativeRuntimePart(parts, "threads", metrics.get("threads"));
        addNativeRuntimePart(parts, "ctx", metrics.get("ctx"));
        addNativeRuntimePart(parts, "batch", metrics.get("batch"));
        return parts.isEmpty() ? "" : " (" + String.join(", ", parts) + ")";
    }

    private static void addNativeRuntimePart(List<String> parts, String name, String value) {
        if (value != null && !value.isBlank()) {
            parts.add(name + "=" + value);
        }
    }

    static Map<String, String> parseNativeMetrics(String rawMetrics) {
        Map<String, String> parsed = new LinkedHashMap<>();
        if (rawMetrics == null || rawMetrics.isBlank()) {
            return parsed;
        }
        for (String part : rawMetrics.split(",")) {
            int split = part.indexOf('=');
            if (split <= 0) {
                continue;
            }
            String key = part.substring(0, split).trim();
            String value = part.substring(split + 1).trim();
            if (!key.isBlank() && !value.isBlank()) {
                parsed.put(key, value);
            }
        }
        return parsed;
    }

    private static double nativeMillis(Map<String, String> metrics, String key) {
        String value = metrics.get(key);
        if (value == null || value.isBlank()) {
            return 0.0d;
        }
        String normalized = value.trim();
        if (normalized.endsWith("ms")) {
            normalized = normalized.substring(0, normalized.length() - 2);
        }
        try {
            return Double.parseDouble(normalized);
        } catch (NumberFormatException ignored) {
            return 0.0d;
        }
    }

    static void printRunHeader(Path modelPath, FastArgs args, PrintStream out) {
        if (fastRunQuiet()) {
            return;
        }
        if (args == null || args.banner) {
            printBanner(out);
        }
        if (shouldPrintResolvedModelPath(modelPath, args)) {
            out.printf("Resolved local model index entry: %s%n", modelPath);
        }
        out.printf("Model: %s%n", modelPath);
        out.printf("Provider: gguf, format=%s%n", ggufModelFormat(modelPath));
    }

    static void printExecutionRoute(PrintStream out, String backendName, String runnerName) {
        if (fastRunQuiet()) {
            return;
        }
        out.println(executionRouteLine(backendName, runnerName));
        out.println("--------------------------------------------------");
    }

    static String ggufModelFormat(Path modelPath) {
        if (modelPath == null || modelPath.getFileName() == null) {
            return "gguf";
        }
        String name = modelPath.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".gguf")) {
            return "gguf";
        }
        return "gguf";
    }

    static String executionRouteLine(String backendName, String runnerName) {
        String backend = backendName == null || backendName.isBlank() ? "unknown" : backendName;
        return "Execution route: gguf (backend=" + backend + ") [" + executionRouteTag(runnerName) + "]";
    }

    private static String executionRouteTag(String runnerName) {
        String normalized = runnerName == null ? "" : runnerName.trim().toLowerCase(Locale.ROOT);
        if (normalized.contains("daemon")) {
            return "llama_cpp_gguf_daemon";
        }
        if (normalized.contains("fallback")) {
            return "llama_cpp_gguf_fallback";
        }
        return "llama_cpp_gguf_fast_path";
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

    private enum EngineMode {
        AUTO,
        JAVA,
        LLAMA_CPP,
        BENCHMARK;

        static EngineMode parse(String value) {
            String normalized = value == null ? "auto" : value.trim().toLowerCase(Locale.ROOT);
            return switch (normalized) {
                case "java", "java-native", "jvm" -> JAVA;
                case "llamacpp", "llama.cpp", "llama-cpp", "binding", "native" -> LLAMA_CPP;
                case "bench", "benchmark", "compare" -> BENCHMARK;
                default -> AUTO;
            };
        }

        static EngineMode effective(String provider, String engine) {
            EngineMode requested = parse(engine);
            if (requested != AUTO) {
                return requested;
            }
            String normalizedProvider = provider == null ? "" : provider.trim().toLowerCase(Locale.ROOT);
            return switch (normalizedProvider) {
                case "java", "java-native", "jvm" -> JAVA;
                case "llamacpp", "llama.cpp", "llama-cpp", "binding", "native" -> LLAMA_CPP;
                default -> AUTO;
            };
        }
    }

    private static void printProfile(String label, GgufRuntimeProfile profile) {
        System.out.printf(
                "%s: architecture=%s, ggufVersion=%d, tensors=%d, metadata=%d, size=%.2fGiB, "
                        + "load=%.2fs, knownTypes=%.1f%%, decoderTensors=%d/%d, types=%s, status=%s.%n",
                label,
                profile.architecture(),
                profile.ggufVersion(),
                profile.tensorCount(),
                profile.metadataCount(),
                profile.modelBytes() / 1024.0d / 1024.0d / 1024.0d,
                profile.loadMillis() / 1000.0d,
                profile.knownTensorTypeRatio() * 100.0d,
                profile.presentDecoderTensorCount(),
                profile.requiredDecoderTensorCount(),
                profile.compactTypeSummary(4),
                profile.javaStatus());
        if (!profile.missingDecoderTensorExamples().isEmpty()) {
            System.out.printf(
                    "%s missing decoder tensors: count=%d, examples=%s.%n",
                    label,
                    profile.missingDecoderTensorCount(),
                    String.join(", ", profile.missingDecoderTensorExamples()));
        }
        if (!profile.malformedDecoderTensorExamples().isEmpty()) {
            System.out.printf(
                    "%s invalid decoder tensor shapes: count=%d, examples=%s.%n",
                    label,
                    profile.malformedDecoderTensorCount(),
                    String.join(", ", profile.malformedDecoderTensorExamples()));
        }
        System.out.printf("%s readiness: %s.%n", label, javaReadinessSummary(profile));
        if (profile.modelConfig() != null) {
            System.out.printf(
                    "%s config: type=%s, layers=%d, hidden=%d, heads=%d/%d, headDim=%d, context=%d, vocab=%d.%n",
                    label,
                    profile.modelConfig().getModelType(),
                    profile.modelConfig().getNumHiddenLayers(),
                    profile.modelConfig().getHiddenSize(),
                    profile.modelConfig().getNumAttentionHeads(),
                    profile.modelConfig().getResolvedNumKvHeads(),
                    profile.modelConfig().getResolvedHeadDim(),
                    profile.modelConfig().getMaxPositionEmbeddings(),
                    profile.modelConfig().getVocabSize());
        }
    }

    static String javaReadinessSummary(GgufRuntimeProfile profile) {
        boolean loaderReady = profile != null && profile.knownTensorTypeRatio() > 0.0d;
        boolean decoderTensorsReady = profile != null && profile.decoderTensorSetComplete();
        boolean rowDotReady = profile != null && profile.rowDotPrimitivesReady();
        return "loaderReady=" + loaderReady
                + ", decoderTensorsReady=" + decoderTensorsReady
                + ", rowDotReady=" + rowDotReady
                + ", generationReady=false";
    }

    private static void printRowDotProbe(String label, GgufRuntimeProbe probe) {
        if (!probe.hasTensorProbe()) {
            System.out.printf("%s row-dot probe: unavailable, no supported matrix tensor.%n", label);
            return;
        }
        System.out.printf("%s tensor probe: %s.%n", label, probe.compactSummary());
        if (probe.hasPreparedMatrixCachePlan()) {
            System.out.printf(
                    "%s decoder prepared-cache decision: %s.%n",
                    label,
                    probe.preparedMatrixCacheDecision().selectionSummary());
            System.out.printf(
                    "%s decoder prepared-cache plan: %s.%n",
                    label,
                    probe.preparedMatrixCachePlan().compactSummary());
        }
        if (probe.hasPreparedMatrixCacheProbe()) {
            System.out.printf(
                    "%s decoder prepared-cache probe: %s.%n",
                    label,
                    probe.preparedMatrixCacheStats().compactSummary());
        }
    }

    static String formatPromptForModel(String prompt, Path modelPath) {
        String mode = stringConfig("gollek.gguf.fast_run.prompt_format", "auto")
                .trim()
                .toLowerCase(Locale.ROOT);
        if (mode.equals("raw") || mode.equals("none") || prompt == null || prompt.isBlank()) {
            return prompt;
        }
        if (prompt.contains("<|turn>") || prompt.contains("<start_of_turn>")) {
            return prompt;
        }
        boolean gemma4 = mode.equals("gemma4") || (mode.equals("auto") && looksLikeGemma4(modelPath));
        if (!gemma4) {
            return prompt;
        }
        String userPrompt = prompt;
        if (useConciseQuestionInstruction(prompt)) {
            userPrompt = "Answer directly and concisely.\nQuestion: " + prompt;
        }
        return "<|turn>user\n" + userPrompt + "<turn|>\n<|turn>model\n";
    }

    static String effectiveEngineModeName(String provider, String engine) {
        return EngineMode.effective(provider, engine).name();
    }

    private static boolean looksLikeGemma4(Path modelPath) {
        if (modelPath == null || modelPath.getFileName() == null) {
            return false;
        }
        String fileName = modelPath.getFileName().toString().toLowerCase(Locale.ROOT);
        return fileName.contains("gemma-4") || fileName.contains("gemma4");
    }

    private static boolean useConciseQuestionInstruction(String prompt) {
        if (!booleanConfig("gollek.gguf.fast_run.concise_qa_prompt", true)) {
            return false;
        }
        if (prompt == null) {
            return false;
        }
        String trimmed = prompt.trim();
        if (trimmed.isEmpty() || trimmed.getBytes(StandardCharsets.UTF_8).length > 256) {
            return false;
        }
        String normalized = trimmed.toLowerCase(Locale.ROOT);
        return normalized.endsWith("?")
                || normalized.startsWith("where ")
                || normalized.startsWith("what ")
                || normalized.startsWith("who ")
                || normalized.startsWith("when ")
                || normalized.startsWith("why ")
                || normalized.startsWith("how ")
                || normalized.startsWith("which ");
    }

    private static String normalizedBackend(String backend) {
        if (backend == null || backend.isBlank()) {
            return "metal";
        }
        String normalized = backend.trim().toLowerCase(Locale.ROOT);
        if (normalized.equals("gpu")) {
            return "metal";
        }
        return normalized;
    }

    private static int defaultThreads() {
        int cpus = Runtime.getRuntime().availableProcessors();
        return Math.max(1, Math.min(8, cpus));
    }

    static int boundedBatch(String property, int defaultValue, int maxValue) {
        int requested = intConfig(property, defaultValue);
        int upperBound = Math.max(1, maxValue);
        return Math.max(1, Math.min(requested, upperBound));
    }

    static int fastRunContext(String prompt, int maxTokens) {
        String configured = configValue("gollek.gguf.fast_run.context");
        if (configured != null && !configured.isBlank() && !configured.trim().equalsIgnoreCase("auto")) {
            return parsePositiveInt(configured, 2048);
        }

        int maxAutoContext = fastRunMaxAutoContext();
        int promptBytes = prompt == null ? 0 : prompt.getBytes(StandardCharsets.UTF_8).length;
        int requested = (int) Math.min(Integer.MAX_VALUE, (long) promptBytes + Math.max(1, maxTokens) + 128L);
        return nextPowerOfTwoBounded(requested, 512, maxAutoContext);
    }

    static int fastRunPrewarmContext(String prompt, int maxTokens) {
        String configured = configValue("gollek.gguf.fast_run.context");
        if (configured != null && !configured.isBlank() && !configured.trim().equalsIgnoreCase("auto")) {
            return parsePositiveInt(configured, 2048);
        }
        int maxAutoContext = fastRunMaxAutoContext();
        int requested = Math.max(fastRunContext(prompt, maxTokens), fastRunPrewarmMinContext(maxAutoContext));
        return nextPowerOfTwoBounded(requested, 512, maxAutoContext);
    }

    static int fastRunPrewarmMinContext(int maxAutoContext) {
        String configured = firstNonBlank(configValue("gollek.gguf.fast_run.prewarm_min_context"),
                Integer.toString(maxAutoContext));
        return Math.min(maxAutoContext, parsePositiveInt(configured, maxAutoContext));
    }

    private static int fastRunMaxAutoContext() {
        return positiveIntConfig("gollek.gguf.fast_run.max_auto_context", 2048);
    }

    private static String configValue(String property) {
        String configured = System.getProperty(property);
        if (configured != null && !configured.isBlank()) {
            return configured;
        }
        for (String envName : configEnvNames(property)) {
            configured = System.getenv(envName);
            if (configured != null && !configured.isBlank()) {
                return configured;
            }
        }
        return null;
    }

    static List<String> configEnvNames(String property) {
        List<String> names = new ArrayList<>();
        addEnvName(names, envName(property));
        String prefix = "gollek.gguf.fast_run.";
        if (property != null && property.startsWith(prefix)) {
            addEnvName(names, "GOLLEK_GGUF_FAST_" + envName(property.substring(prefix.length())));
        }
        addLegacyEnvNames(names, property);
        return names;
    }

    private static void addLegacyEnvNames(List<String> names, String property) {
        if (property == null) {
            return;
        }
        switch (property) {
            case "gollek.gguf.fast_run.gpu_layers" -> addEnvName(names, "GGUF_GPU_LAYERS");
            case "gollek.gguf.fast_run.batch" -> addEnvName(names, "GGUF_BATCH_SIZE");
            case "gollek.gguf.fast_run.ubatch" -> {
                addEnvName(names, "GGUF_UBATCH_SIZE");
                addEnvName(names, "GGUF_MICRO_BATCH_SIZE");
            }
            case "gollek.gguf.fast_run.threads" -> addEnvName(names, "GGUF_THREADS");
            case "gollek.gguf.fast_run.batch_threads" -> addEnvName(names, "GGUF_BATCH_THREADS");
            case "gollek.gguf.fast_run.context" -> {
                addEnvName(names, "GGUF_CONTEXT");
                addEnvName(names, "GGUF_CONTEXT_SIZE");
                addEnvName(names, "GGUF_CTX_SIZE");
            }
            case "gollek.gguf.fast_run.llama_lib_dir" -> addEnvName(names, "GOLLEK_LLAMA_LIB_DIR");
            case "gollek.gguf.fast_run.bridge" -> addEnvName(names, "GOLLEK_GGUF_FAST_BRIDGE");
            default -> {
            }
        }
    }

    private static void addEnvName(List<String> names, String name) {
        if (name != null && !name.isBlank() && !names.contains(name)) {
            names.add(name);
        }
    }

    private static String envName(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        String normalized = value.trim().toUpperCase(Locale.ROOT).replaceAll("[^A-Z0-9]+", "_");
        return normalized.replaceAll("^_+|_+$", "");
    }

    private static String stringConfig(String property, String fallback) {
        return firstNonBlank(configValue(property), fallback);
    }

    private static boolean booleanConfig(String property, boolean fallback) {
        String configured = configValue(property);
        if (configured == null || configured.isBlank()) {
            return fallback;
        }
        return switch (configured.trim().toLowerCase(Locale.ROOT)) {
            case "true", "1", "yes", "y", "on" -> true;
            case "false", "0", "no", "n", "off" -> false;
            default -> fallback;
        };
    }

    private static int intConfig(String property, int fallback) {
        String configured = configValue(property);
        if (configured == null || configured.isBlank()) {
            return fallback;
        }
        try {
            return Integer.parseInt(configured.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static int positiveIntConfig(String property, int fallback) {
        return parsePositiveInt(configValue(property), fallback);
    }

    private static long longConfig(String property, long fallback) {
        String configured = configValue(property);
        if (configured == null || configured.isBlank()) {
            return fallback;
        }
        try {
            return Long.parseLong(configured.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static int parsePositiveInt(String value, int fallback) {
        if (value == null || value.isBlank()) {
            return Math.max(1, fallback);
        }
        try {
            return Math.max(1, Integer.parseInt(value.trim()));
        } catch (NumberFormatException ignored) {
            return Math.max(1, fallback);
        }
    }

    private static int nextPowerOfTwoBounded(int value, int minimum, int maximum) {
        int lowerBound = Math.max(1, minimum);
        int upperBound = Math.max(lowerBound, maximum);
        int context = lowerBound;
        int target = Math.max(lowerBound, value);
        while (context < target && context < upperBound) {
            context = Math.min(context << 1, upperBound);
        }
        return context;
    }

    static boolean fastRunSwaFull() {
        return booleanConfig("gollek.gguf.fast_run.swa_full", false);
    }

    static boolean fastRunCpuFallback() {
        return booleanConfig("gollek.gguf.fast_run.cpu_fallback", true);
    }

    static boolean fastRunTiming() {
        Boolean requestTiming = REQUEST_TIMING.get();
        if (requestTiming != null) {
            return requestTiming;
        }
        return booleanConfig("gollek.gguf.fast_run.timing", false);
    }

    private static boolean fastRunDebug() {
        return booleanConfig("gollek.gguf.fast_run.debug", false);
    }

    /**
     * Quiet mode: suppress all verbose run headers, execution-route lines, performance
     * metrics blocks, and the 'Using llama.cpp GGUF...' status line. Only the raw
     * generated tokens are printed to stdout. Activated by system property
     * {@code gollek.gguf.fast_run.quiet=true} or env {@code GOLLEK_GGUF_FAST_QUIET=1}.
     * This is automatically set when Wayang uses the SDK in embedded mode.
     */
    static boolean fastRunQuiet() {
        return booleanConfig("gollek.gguf.fast_run.quiet", false);
    }

    private static boolean daemonEnabled() {
        return booleanConfig("gollek.gguf.fast_run.daemon", false);
    }

    private record SessionKey(
            Path modelPath,
            int context,
            int batch,
            int microBatch,
            int threads,
            int batchThreads,
            int gpuLayers,
            boolean useMmap,
            boolean useMlock,
            boolean swaFull) {
    }

    private record SessionLease(
            NativeGgufSession session,
            boolean reused,
            boolean closeOnClose,
            String reuseKind) implements AutoCloseable {
        static SessionLease open(
                Path modelPath,
                int context,
                int batch,
                int microBatch,
                int threads,
                int batchThreads,
                int gpuLayers,
                boolean useMmap,
                boolean useMlock,
                boolean swaFull) throws Throwable {
            return new SessionLease(NativeGgufSession.open(
                    modelPath,
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    gpuLayers,
                    useMmap,
                    useMlock,
                    swaFull), false, true, "cold");
        }

        @Override
        public void close() {
            if (closeOnClose) {
                session.close();
            }
        }
    }

    private static final class GgufSessionCache implements AutoCloseable {
        private final Map<SessionKey, NativeGgufSession> sessions = new LinkedHashMap<>(4, 0.75f, true);

        private SessionLease acquire(
                Path modelPath,
                int context,
                int batch,
                int microBatch,
                int threads,
                int batchThreads,
                int gpuLayers,
                boolean useMmap,
                boolean useMlock,
                boolean swaFull) throws Throwable {
            SessionKey key = new SessionKey(
                    modelPath.toAbsolutePath().normalize(),
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    gpuLayers,
                    useMmap,
                    useMlock,
                    swaFull);
            NativeGgufSession existing = sessions.get(key);
            if (existing != null) {
                daemonLog("session cache hit: " + key.modelPath().getFileName()
                        + ", context=" + context
                        + ", gpuLayers=" + gpuLayers
                        + ", sessions=" + sessions.size());
                return new SessionLease(existing, true, false, "exact");
            }
            SessionLease compatible = findCompatible(key);
            if (compatible != null) {
                return compatible;
            }
            daemonLog("session cache miss: " + key.modelPath().getFileName()
                    + ", context=" + context
                    + ", gpuLayers=" + gpuLayers
                    + ", sessions=" + sessions.size());
            NativeGgufSession opened = NativeGgufSession.open(
                    key.modelPath(),
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    gpuLayers,
                    useMmap,
                    useMlock,
                    swaFull);
            sessions.put(key, opened);
            evictOverflowSessions();
            return new SessionLease(opened, false, false, "cold");
        }

        private SessionLease findCompatible(SessionKey requested) {
            Map.Entry<SessionKey, NativeGgufSession> best = null;
            for (Map.Entry<SessionKey, NativeGgufSession> entry : sessions.entrySet()) {
                if (!isCompatibleWarmSession(entry.getKey(), requested)) {
                    continue;
                }
                if (best == null || smallerCompatibleSession(entry.getKey(), best.getKey())) {
                    best = entry;
                }
            }
            if (best == null) {
                return null;
            }
            SessionKey selected = best.getKey();
            NativeGgufSession session = sessions.get(selected);
            daemonLog("session cache compatible hit: " + selected.modelPath().getFileName()
                    + ", requestedContext=" + requested.context()
                    + ", warmContext=" + selected.context()
                    + ", requestedBatch=" + requested.batch()
                    + ", warmBatch=" + selected.batch()
                    + ", gpuLayers=" + selected.gpuLayers()
                    + ", sessions=" + sessions.size());
            return new SessionLease(session, true, false, "compatible");
        }

        private static boolean isCompatibleWarmSession(SessionKey warm, SessionKey requested) {
            return warm.modelPath().equals(requested.modelPath())
                    && warm.context() >= requested.context()
                    && warm.batch() >= requested.batch()
                    && warm.microBatch() >= requested.microBatch()
                    && warm.threads() == requested.threads()
                    && warm.batchThreads() == requested.batchThreads()
                    && warm.gpuLayers() == requested.gpuLayers()
                    && warm.useMmap() == requested.useMmap()
                    && warm.useMlock() == requested.useMlock()
                    && warm.swaFull() == requested.swaFull();
        }

        private static boolean smallerCompatibleSession(SessionKey left, SessionKey right) {
            int contextCompare = Integer.compare(left.context(), right.context());
            if (contextCompare != 0) {
                return contextCompare < 0;
            }
            int batchCompare = Integer.compare(left.batch(), right.batch());
            if (batchCompare != 0) {
                return batchCompare < 0;
            }
            return Integer.compare(left.microBatch(), right.microBatch()) < 0;
        }

        private void evictOverflowSessions() {
            int maxSessions = daemonMaxSessions();
            while (sessions.size() > maxSessions) {
                Map.Entry<SessionKey, NativeGgufSession> eldest = sessions.entrySet().iterator().next();
                sessions.remove(eldest.getKey());
                daemonLog("session cache evict: " + eldest.getKey().modelPath().getFileName()
                        + ", context=" + eldest.getKey().context()
                        + ", gpuLayers=" + eldest.getKey().gpuLayers()
                        + ", maxSessions=" + maxSessions);
                eldest.getValue().close();
            }
        }

        @Override
        public void close() {
            for (NativeGgufSession session : sessions.values()) {
                session.close();
            }
            sessions.clear();
        }
    }

    private static final class NativeGgufSession implements AutoCloseable {
        private static final long ERROR_BUFFER_BYTES = 8192L;

        private final MemorySegment handle;
        private final String backendName;
        private final Arena scratchArena = Arena.ofShared();
        private MemorySegment promptBuffer = MemorySegment.NULL;
        private MemorySegment outputBuffer = MemorySegment.NULL;
        private MemorySegment errorBuffer = MemorySegment.NULL;
        private MemorySegment metricsBuffer = MemorySegment.NULL;
        private long promptBufferCapacity;
        private long outputBufferCapacity;
        private long metricsBufferCapacity;
        private int lastGeneratedTokens;
        private int lastOutputBufferRetries;
        private boolean closed;

        private NativeGgufSession(MemorySegment handle, String backendName) {
            this.handle = handle;
            this.backendName = backendName;
        }

        private static NativeGgufSession open(
                Path modelPath,
                int context,
                int batch,
                int microBatch,
                int threads,
                int batchThreads,
                int gpuLayers,
                boolean useMmap,
                boolean useMlock,
                boolean swaFull) throws Throwable {
            NativeGgufBinding binding = NativeGgufBinding.INSTANCE;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment model = arena.allocateFrom(modelPath.toString(), StandardCharsets.UTF_8);
                MemorySegment backendDir = arena.allocateFrom(llamaLibDir().toString(), StandardCharsets.UTF_8);
                MemorySegment error = arena.allocate(8192);
                MemorySegment handle = (MemorySegment) binding.open.invokeExact(
                        model,
                        backendDir,
                        context,
                        batch,
                        microBatch,
                        threads,
                        batchThreads,
                        gpuLayers,
                        useMmap ? 1 : 0,
                        useMlock ? 1 : 0,
                        swaFull ? 1 : 0,
                        error,
                        8192L);
                if (isNull(handle)) {
                    throw new IllegalStateException(error.getString(0));
                }
                String backendName = "unknown";
                MemorySegment backendPtr = (MemorySegment) binding.backendName.invokeExact(handle);
                if (!isNull(backendPtr)) {
                    backendName = backendPtr.reinterpret(128).getString(0);
                }
                return new NativeGgufSession(handle, backendName);
            }
        }

        private synchronized String generate(
                String prompt,
                int maxTokens,
                double temperature,
                int topK,
                double topP) throws Throwable {
            ensureOpen();
            MemorySegment promptSegment = promptSegment(prompt);
            MemorySegment error = errorSegment();
            long requestedOutputBytes = outputBufferBytesForTokens(maxTokens);
            int retries = 0;
            for (int attempt = 0; attempt <= OUTPUT_BUFFER_RETRY_LIMIT; ++attempt) {
                MemorySegment output = outputSegment(requestedOutputBytes);
                int generated = (int) NativeGgufBinding.INSTANCE.generate.invokeExact(
                        handle,
                        promptSegment,
                        maxTokens,
                        (float) temperature,
                        topK,
                        (float) topP,
                        0,
                        output,
                        outputBufferCapacity,
                        error,
                        ERROR_BUFFER_BYTES);
                if (generated >= 0) {
                    lastGeneratedTokens = generated;
                    lastOutputBufferRetries = retries;
                    return output.getString(0, StandardCharsets.UTF_8);
                }
                String message = error.getString(0);
                if (!outputBufferTooSmall(generated, message) || attempt == OUTPUT_BUFFER_RETRY_LIMIT) {
                    lastOutputBufferRetries = retries;
                    throw new IllegalStateException(message);
                }
                retries++;
                requestedOutputBytes = nextOutputBufferBytes(outputBufferCapacity);
            }
            throw new IllegalStateException("GGUF generation failed");
        }

        private String backendName() {
            return backendName;
        }

        private int lastGeneratedTokens() {
            return lastGeneratedTokens;
        }

        private synchronized Optional<String> metrics() throws Throwable {
            return metrics(false, "cold");
        }

        private synchronized Optional<String> metrics(boolean warmSession, String reuseKind) throws Throwable {
            ensureOpen();
            NativeGgufBinding binding = NativeGgufBinding.INSTANCE;
            if (binding.metrics == null) {
                return Optional.empty();
            }
            MemorySegment output = metricsSegment();
            int written = (int) binding.metrics.invokeExact(handle, output, metricsBufferCapacity);
            if (written <= 0) {
                return Optional.empty();
            }
            return Optional.of(output.getString(0, StandardCharsets.UTF_8)
                    + ", warmSession=" + warmSession
                    + ", sessionReuse=" + metricToken(reuseKind, warmSession ? "warm" : "cold")
                    + ", javaOutputBufferBytes=" + outputBufferCapacity
                    + ", javaOutputRetries=" + lastOutputBufferRetries);
        }

        private static String metricToken(String value, String fallback) {
            String normalized = value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
            if (normalized.isBlank()) {
                return fallback;
            }
            return normalized.replaceAll("[^a-z0-9_-]+", "_");
        }

        private MemorySegment promptSegment(String prompt) {
            byte[] bytes = prompt.getBytes(StandardCharsets.UTF_8);
            long required = bytes.length + 1L;
            if (promptBufferCapacity < required) {
                promptBufferCapacity = nextScratchCapacity(required);
                promptBuffer = scratchArena.allocate(promptBufferCapacity);
            }
            promptBuffer.asSlice(0, bytes.length).copyFrom(MemorySegment.ofArray(bytes));
            promptBuffer.set(ValueLayout.JAVA_BYTE, bytes.length, (byte) 0);
            return promptBuffer;
        }

        private MemorySegment outputSegment(long requested) {
            if (outputBufferCapacity < requested) {
                outputBufferCapacity = nextScratchCapacity(requested);
                outputBuffer = scratchArena.allocate(outputBufferCapacity);
            }
            return outputBuffer;
        }

        private MemorySegment errorSegment() {
            if (isNull(errorBuffer)) {
                errorBuffer = scratchArena.allocate(ERROR_BUFFER_BYTES);
            }
            return errorBuffer;
        }

        private MemorySegment metricsSegment() {
            long requested = 4096L;
            if (metricsBufferCapacity < requested) {
                metricsBufferCapacity = requested;
                metricsBuffer = scratchArena.allocate(metricsBufferCapacity);
            }
            return metricsBuffer;
        }

        private static long nextScratchCapacity(long requested) {
            long capacity = 1024L;
            while (capacity < requested && capacity <= Long.MAX_VALUE / 2L) {
                capacity <<= 1;
            }
            return Math.max(requested, capacity);
        }

        private void ensureOpen() {
            if (closed) {
                throw new IllegalStateException("GGUF native session is closed");
            }
        }

        @Override
        public synchronized void close() {
            if (closed) {
                return;
            }
            closed = true;
            try {
                NativeGgufBinding.INSTANCE.close.invokeExact(handle);
            } catch (Throwable ignored) {
            } finally {
                scratchArena.close();
            }
        }
    }

    private static Path llamaLibDir() {
        String configured = firstNonBlank(
                configValue("gollek.gguf.fast_run.llama_lib_dir"),
                System.getenv("GOLLEK_LLAMA_LIB_DIR"),
                null);
        if (configured != null && !configured.isBlank()) {
            return Path.of(configured).toAbsolutePath();
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "libs", "llama").toAbsolutePath();
    }

    private static Path bridgePath() {
        String configured = configValue("gollek.gguf.fast_run.bridge");
        if (configured != null && !configured.isBlank()) {
            return Path.of(configured).toAbsolutePath();
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "libs", "libgollek_gguf_fast.dylib").toAbsolutePath();
    }

    private static boolean isNull(MemorySegment segment) {
        return segment == null || segment.equals(MemorySegment.NULL) || segment.address() == 0L;
    }

    private static String briefMessage(Throwable throwable) {
        Throwable cursor = throwable;
        while (cursor.getCause() != null) {
            cursor = cursor.getCause();
        }
        String message = cursor.getMessage();
        if (message == null || message.isBlank()) {
            return cursor.getClass().getSimpleName();
        }
        return message.replace('\n', ' ').replace('\r', ' ');
    }

    private static final class NativeGgufBinding {
        private static final NativeGgufBinding INSTANCE = new NativeGgufBinding();

        private final MethodHandle open;
        private final MethodHandle generate;
        private final MethodHandle backendName;
        private final MethodHandle metrics;
        private final MethodHandle hardExit;
        private final MethodHandle close;

        private NativeGgufBinding() {
            try {
                Path bridge = bridgePath();
                if (!Files.isRegularFile(bridge)) {
                    throw new IOException("GGUF fast bridge is not installed at " + bridge);
                }
                System.load(bridge.toString());
                Linker linker = Linker.nativeLinker();
                SymbolLookup lookup = SymbolLookup.libraryLookup(bridge, Arena.global());
                this.open = linker.downcallHandle(
                        lookup.find("gollek_gguf_open").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_LONG));
                this.generate = linker.downcallHandle(
                        lookup.find("gollek_gguf_generate").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_FLOAT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_FLOAT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_LONG,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_LONG));
                this.backendName = linker.downcallHandle(
                        lookup.find("gollek_gguf_backend_name").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
                Optional<MemorySegment> metricsSymbol = lookup.find("gollek_gguf_last_metrics");
                this.metrics = metricsSymbol.isPresent()
                        ? linker.downcallHandle(
                                metricsSymbol.get(),
                                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                                        ValueLayout.ADDRESS,
                                        ValueLayout.ADDRESS,
                                        ValueLayout.JAVA_LONG))
                        : null;
                Optional<MemorySegment> hardExitSymbol = lookup.find("gollek_gguf_hard_exit");
                this.hardExit = hardExitSymbol.isPresent()
                        ? linker.downcallHandle(
                                hardExitSymbol.get(),
                                FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT))
                        : null;
                this.close = linker.downcallHandle(
                        lookup.find("gollek_gguf_close").orElseThrow(),
                        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            } catch (Throwable throwable) {
                throw new ExceptionInInitializerError(throwable);
            }
        }
    }

    private static OptionalInt requestDaemon(String[] rawArgs) {
        OptionalInt status = sendDaemonRequestWithRetries(rawArgs, DAEMON_MAGIC, daemonRequestRetryMillis());
        if (status.isPresent()) {
            return status;
        }
        if (!startDaemon()) {
            return OptionalInt.empty();
        }
        return sendDaemonRequestWithRetries(rawArgs, DAEMON_MAGIC, daemonStartTimeoutMillis());
    }

    private static OptionalInt requestDaemonPrewarm(String[] rawArgs) {
        OptionalInt status = sendDaemonRequestWithRetries(rawArgs, DAEMON_PREWARM_MAGIC, daemonRequestRetryMillis());
        if (status.isPresent()) {
            return status;
        }
        if (!startDaemon()) {
            return OptionalInt.empty();
        }
        return sendDaemonRequestWithRetries(rawArgs, DAEMON_PREWARM_MAGIC, daemonStartTimeoutMillis());
    }

    private static OptionalInt sendDaemonRequestWithRetries(String[] rawArgs, String magic, long retryMillis) {
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(Math.max(0L, retryMillis));
        while (true) {
            OptionalInt status = sendDaemonRequest(rawArgs, magic);
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

    private static OptionalInt sendDaemonRequest(String[] rawArgs, String magic) {
        OptionalInt port = readDaemonPort();
        if (port.isEmpty()) {
            return OptionalInt.empty();
        }
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getLoopbackAddress(), port.getAsInt()),
                    daemonConnectTimeoutMillis());
            socket.setSoTimeout((int) Math.min(Integer.MAX_VALUE, timeoutMillis()));
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            DataInputStream in = new DataInputStream(socket.getInputStream());
            writeString(out, magic);
            out.writeInt(rawArgs.length);
            for (String arg : rawArgs) {
                writeString(out, arg);
            }
            out.writeBoolean(fastRunTiming());
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
            if (fastRunDebug()) {
                System.err.println("GGUF daemon request failed: " + briefMessage(e));
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
            command.add("-Xmx" + stringConfig("gollek.gguf.fast_run.daemon_heap", "1g"));
            command.add("-XX:MaxDirectMemorySize="
                    + stringConfig("gollek.gguf.fast_run.daemon_direct_memory", "8g"));
            command.add("--enable-preview");
            command.add("--add-modules");
            command.add("jdk.incubator.vector");
            command.add("--enable-native-access=ALL-UNNAMED");
            copyGollekSystemProperties(command);
            command.add("-Djava.library.path=" + System.getProperty("java.library.path", ""));
            command.add("-cp");
            command.add(System.getProperty("java.class.path"));
            command.add(GgufFastRun.class.getName());
            command.add("__daemon");

            if (!startDetachedDaemon(command)) {
                return false;
            }

            long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(daemonStartTimeoutMillis());
            while (System.nanoTime() < deadlineNanos) {
                if (readDaemonPort().isPresent()) {
                    return true;
                }
                Thread.sleep(100L);
            }
            return readDaemonPort().isPresent();
        } catch (Exception e) {
            if (fastRunDebug()) {
                System.err.println("GGUF daemon failed to start: " + briefMessage(e));
            }
            return false;
        }
    }

    private static boolean startDetachedDaemon(List<String> command) throws IOException, InterruptedException {
        String launcher = daemonLauncherMode();
        if (fastRunDebug()) {
            System.err.println("GGUF daemon launcher: " + launcher);
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
        String defaultLauncher = defaultDaemonLauncherMode();
        String launcher = stringConfig("gollek.gguf.fast_run.daemon_launcher", defaultLauncher)
                .trim()
                .toLowerCase(Locale.ROOT);
        return switch (launcher) {
            case "auto", "launchctl", "nohup" -> launcher;
            default -> defaultLauncher;
        };
    }

    static String defaultDaemonLauncherMode() {
        return isMacOs() ? "launchctl" : "nohup";
    }

    private static boolean startDetachedDaemonWithLaunchctl(List<String> command) throws IOException, InterruptedException {
        String launchctl = "/bin/launchctl";
        if (!Files.isExecutable(Path.of(launchctl))) {
            return false;
        }
        removeLaunchctlDaemon();
        Process launcher = new ProcessBuilder(
                launchctl,
                "submit",
                "-l",
                launchctlLabel(),
                "--",
                "/bin/sh",
                "-c",
                daemonShellCommand(command, false))
                .redirectOutput(ProcessBuilder.Redirect.appendTo(daemonLogFile().toFile()))
                .redirectError(ProcessBuilder.Redirect.appendTo(daemonLogFile().toFile()))
                .start();
        if (!launcher.waitFor(3, TimeUnit.SECONDS)) {
            return true;
        }
        boolean launched = launcher.exitValue() == 0;
        if (!launched && fastRunDebug()) {
            System.err.println("GGUF launchctl daemon submit failed; falling back to nohup");
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
        String nativePaths = home + "/.gollek/libs" + File.pathSeparator + home + "/.gollek/libs/llama";
        StringBuilder shellCommand = new StringBuilder();
        shellCommand.append("export DYLD_LIBRARY_PATH=")
                .append(shellQuote(nativePaths))
                .append("${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}; ");
        shellCommand.append("export DYLD_FALLBACK_LIBRARY_PATH=")
                .append(shellQuote(nativePaths))
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

    private static int runDaemon() {
        int port = -1;
        try {
            Files.createDirectories(daemonRunDir());
            try (ServerSocket server = new ServerSocket(0, 16, InetAddress.getLoopbackAddress());
                    GgufSessionCache sessionCache = new GgufSessionCache()) {
                port = server.getLocalPort();
                writeDaemonPort(port);
                long idleNanos = TimeUnit.SECONDS.toNanos(
                        longConfig("gollek.gguf.fast_run.daemon_idle_seconds", 900L));
                long lastRequestNanos = System.nanoTime();
                server.setSoTimeout(1000);
                daemonLog("daemon started: pid=" + ProcessHandle.current().pid()
                        + ", port=" + port
                        + ", idleSeconds=" + TimeUnit.NANOSECONDS.toSeconds(idleNanos));
                boolean stop = false;
                while (!stop) {
                    try (Socket socket = server.accept()) {
                        lastRequestNanos = System.nanoTime();
                        stop = handleDaemonClient(socket, sessionCache);
                    } catch (SocketTimeoutException ignored) {
                        if (System.nanoTime() - lastRequestNanos > idleNanos) {
                            daemonLog("daemon idle timeout");
                            break;
                        }
                    }
                }
                daemonLog("daemon stopping");
                if (daemonHardExit()) {
                    hardExitDaemon(0, port);
                }
            }
            return 0;
        } catch (Throwable throwable) {
            System.err.println("GGUF daemon failed: " + briefMessage(throwable));
            if (fastRunDebug()) {
                throwable.printStackTrace(System.err);
            }
            return COMMAND_ERROR;
        } finally {
            deleteDaemonPortFileIfPort(port);
        }
    }

    private static boolean handleDaemonClient(Socket socket, GgufSessionCache sessionCache) {
        try {
            socket.setSoTimeout((int) Math.min(Integer.MAX_VALUE, timeoutMillis()));
            DataInputStream in = new DataInputStream(socket.getInputStream());
            DataOutputStream framed = new DataOutputStream(socket.getOutputStream());
            String magic = readString(in);
            if (DAEMON_STOP_MAGIC.equals(magic)) {
                writeStatusFrame(framed, 0);
                daemonLog("daemon stop requested");
                return true;
            }
            boolean prewarmRequest = DAEMON_PREWARM_MAGIC.equals(magic);
            if (!DAEMON_MAGIC.equals(magic) && !prewarmRequest) {
                writeStatusFrame(framed, FALLBACK_TO_FULL_CLI);
                return false;
            }
            int argCount = in.readInt();
            if (argCount < 0 || argCount > 512) {
                throw new IOException("Invalid GGUF daemon arg count: " + argCount);
            }
            String[] rawArgs = new String[argCount];
            for (int i = 0; i < argCount; i++) {
                rawArgs[i] = readString(in);
            }
            boolean requestTiming = in.readBoolean();

            PrintStream out = new PrintStream(new FramedOutputStream(framed, 1), true, StandardCharsets.UTF_8);
            PrintStream err = new PrintStream(new FramedOutputStream(framed, 2), true, StandardCharsets.UTF_8);
            int status;
            REQUEST_TIMING.set(requestTiming);
            try {
                FastArgs parsed = FastArgs.parse(rawArgs);
                if (!parsed.supported()) {
                    status = FALLBACK_TO_FULL_CLI;
                } else {
                    Optional<Path> modelPath = resolveGgufModel(parsed);
                    if (modelPath.isEmpty()) {
                        status = FALLBACK_TO_FULL_CLI;
                    } else if (prewarmRequest) {
                        status = prewarmWithLlamaCpp(modelPath.get(), parsed, out, err, sessionCache);
                    } else {
                        status = generate(modelPath.get(), parsed, out, err, sessionCache, "daemon");
                    }
                }
            } catch (Throwable throwable) {
                err.println("GGUF daemon request failed: " + briefMessage(throwable));
                if (fastRunDebug()) {
                    throwable.printStackTrace(err);
                }
                status = COMMAND_ERROR;
            } finally {
                REQUEST_TIMING.remove();
            }
            out.flush();
            err.flush();
            writeStatusFrame(framed, status);
            return false;
        } catch (Exception e) {
            if (fastRunDebug()) {
                System.err.println("GGUF daemon client failed: " + briefMessage(e));
            }
            return false;
        }
    }

    private static int stopDaemon() {
        OptionalInt port = readDaemonPort();
        if (port.isEmpty()) {
            removeLaunchctlDaemon();
            return 0;
        }
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getLoopbackAddress(), port.getAsInt()),
                    daemonConnectTimeoutMillis());
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            DataInputStream in = new DataInputStream(socket.getInputStream());
            writeString(out, DAEMON_STOP_MAGIC);
            out.flush();
            int channel = in.read();
            int status = in.readInt();
            return channel == 0 ? status : COMMAND_ERROR;
        } catch (Exception ignored) {
            return 0;
        } finally {
            deleteDaemonPortFile();
            removeLaunchctlDaemon();
        }
    }

    private static void copyGollekSystemProperties(List<String> command) {
        Properties properties = System.getProperties();
        for (String name : properties.stringPropertyNames()) {
            if (name.startsWith("gollek.gguf.fast_run.")) {
                command.add("-D" + name + "=" + properties.getProperty(name));
            }
        }
    }

    private static Path daemonRunDir() {
        return Path.of(System.getProperty("user.home"), ".gollek", "run");
    }

    private static Path daemonPortFile() {
        return daemonRunDir().resolve("gguf-fast-daemon.port");
    }

    private static Path daemonLogFile() {
        return Path.of(System.getProperty("user.home"), ".gollek", "logs", "gguf-fast-daemon.log");
    }

    private static OptionalInt readDaemonPort() {
        Optional<DaemonInfo> info = readDaemonInfo();
        if (info.isEmpty()) {
            return OptionalInt.empty();
        }
        if (!isProcessAlive(info.get().pid())) {
            discardStaleDaemon(info.get());
            return OptionalInt.empty();
        }
        if (strictDaemonKey() && !daemonKey().equals(info.get().key())) {
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
            if (lines.size() < 3) {
                return Optional.empty();
            }
            int port = Integer.parseInt(lines.get(0).trim());
            long pid = Long.parseLong(lines.get(1).trim());
            String key = lines.get(2);
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
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static String daemonKey() {
        return classPathKey(System.getProperty("java.class.path", ""))
                + "|" + pathKey(bridgePath())
                + "|" + pathKey(llamaLibDir());
    }

    private static String classPathKey(String classPath) {
        if (classPath == null || classPath.isBlank()) {
            return "";
        }
        String[] entries = classPath.split(java.util.regex.Pattern.quote(File.pathSeparator));
        StringBuilder key = new StringBuilder();
        for (int i = 0; i < entries.length; i++) {
            if (i > 0) {
                key.append(File.pathSeparator);
            }
            String entry = entries[i];
            if (entry == null || entry.isBlank()) {
                continue;
            }
            Path path = Path.of(entry);
            key.append(Files.exists(path) ? pathKey(path) : entry);
        }
        return key.toString();
    }

    private static String pathKey(Path path) {
        try {
            long modified = Files.exists(path) ? Files.getLastModifiedTime(path).toMillis() : -1L;
            long size = Files.isRegularFile(path) ? Files.size(path) : -1L;
            return path.toAbsolutePath().normalize() + ":size=" + size + ":mtime=" + modified;
        } catch (Exception ignored) {
            return path.toAbsolutePath().normalize().toString();
        }
    }

    private static boolean isProcessAlive(long pid) {
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
        if (!isProcessAlive(pid)) {
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
        return booleanConfig("gollek.gguf.fast_run.stale_daemon_force_kill", true);
    }

    static long staleDaemonKillWaitMillis() {
        return Math.max(100L, longConfig("gollek.gguf.fast_run.stale_daemon_kill_wait_ms", 2000L));
    }

    static boolean strictDaemonKey() {
        return booleanConfig("gollek.gguf.fast_run.strict_daemon_key", false);
    }

    static boolean hardExitAfterRun() {
        return booleanConfig("gollek.gguf.fast_run.hard_exit_after_run", true);
    }

    static boolean shouldHardExitAfterNativeRun(boolean daemonSession) {
        return !daemonSession && hardExitAfterRun();
    }

    private static void daemonLog(String message) {
        System.err.printf("%tF %<tT [gguf-daemon] %s%n", System.currentTimeMillis(), message);
    }

    private static boolean daemonHardExit() {
        return booleanConfig("gollek.gguf.fast_run.daemon_hard_exit", true);
    }

    private static void hardExitDaemon(int status, int port) {
        deleteDaemonPortFileIfPort(port);
        hardExitProcess(status);
    }

    static void hardExitProcess(int status) {
        System.out.flush();
        System.err.flush();
        try {
            MethodHandle hardExit = NativeGgufBinding.INSTANCE.hardExit;
            if (hardExit != null) {
                hardExit.invokeExact(status);
            }
        } catch (Throwable ignored) {
        }
        Runtime.getRuntime().halt(status);
    }

    private static boolean isMacOs() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac");
    }

    private static String launchctlLabel() {
        return stringConfig("gollek.gguf.fast_run.daemon_launchctl_label",
                "tech.kayys.alkhawarizm.gguf-fast-daemon");
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

    private static void deleteDaemonPortFile() {
        try {
            Files.deleteIfExists(daemonPortFile());
        } catch (Exception ignored) {
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
        return intConfig("gollek.gguf.fast_run.daemon_connect_timeout_ms", 1000);
    }

    private static int daemonStartTimeoutMillis() {
        return intConfig("gollek.gguf.fast_run.daemon_start_timeout_ms", 20_000);
    }

    static long daemonRequestRetryMillis() {
        return Math.max(0L, longConfig("gollek.gguf.fast_run.daemon_request_retry_ms", 2_000L));
    }

    static long daemonRequestRetrySleepMillis() {
        return Math.max(10L, longConfig("gollek.gguf.fast_run.daemon_request_retry_sleep_ms", 75L));
    }

    static int daemonMaxSessions() {
        return Math.max(1, intConfig("gollek.gguf.fast_run.daemon_max_sessions", 2));
    }

    private static long timeoutMillis() {
        return TimeUnit.SECONDS.toMillis(longConfig("gollek.gguf.fast_run.timeout_seconds", 180L));
    }

    private static void writeStatusFrame(DataOutputStream out, int status) throws IOException {
        synchronized (out) {
            out.write(0);
            out.writeInt(status);
            out.flush();
        }
    }

    private static void writeFrame(DataOutputStream out, int channel, byte[] payload) throws IOException {
        if (payload.length == 0) {
            return;
        }
        synchronized (out) {
            out.write(channel);
            out.writeInt(payload.length);
            out.write(payload);
            out.flush();
        }
    }

    private static void writeString(DataOutputStream out, String value) throws IOException {
        byte[] bytes = (value == null ? "" : value).getBytes(StandardCharsets.UTF_8);
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    private static String readString(DataInputStream in) throws IOException {
        int length = in.readInt();
        if (length < 0 || length > 1_048_576) {
            throw new IOException("Invalid GGUF daemon string length: " + length);
        }
        byte[] bytes = in.readNBytes(length);
        if (bytes.length != length) {
            throw new IOException("Unexpected end of GGUF daemon string");
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private record DaemonInfo(int port, long pid, String key) {
    }

    private static final class FramedOutputStream extends OutputStream {
        private final DataOutputStream target;
        private final int channel;
        private final ByteArrayOutputStream buffer = new ByteArrayOutputStream();

        private FramedOutputStream(DataOutputStream target, int channel) {
            this.target = target;
            this.channel = channel;
        }

        @Override
        public void write(int b) {
            buffer.write(b);
        }

        @Override
        public void write(byte[] b, int off, int len) {
            buffer.write(b, off, len);
        }

        @Override
        public void flush() throws IOException {
            byte[] payload = buffer.toByteArray();
            buffer.reset();
            writeFrame(target, channel, payload);
        }
    }

    static final class FastArgs {
        private String model;
        private String modelFile;
        private String prompt;
        private int maxTokens = 256;
        private double temperature = 0.2d;
        private double topP = 0.9d;
        private int topK = 40;
        private String provider;
        private String backend = stringConfig("gollek.gguf.fast_run.backend", "metal");
        private String engine = firstNonBlank(
                configValue("gollek.gguf.fast_run.engine"),
                System.getenv("GOLLEK_GGUF_ENGINE"),
                "auto");
        private boolean banner = true;
        private boolean supported = true;

        private boolean supported() {
            if (!supported || prompt == null || prompt.isBlank()) {
                return false;
            }
            if ((model == null || model.isBlank()) && (modelFile == null || modelFile.isBlank())) {
                return false;
            }
            if (provider == null || provider.isBlank()) {
                return true;
            }
            String normalized = provider.toLowerCase(Locale.ROOT);
            return normalized.equals("gguf")
                    || normalized.equals("native")
                    || normalized.equals("llamacpp")
                    || normalized.equals("llama.cpp")
                    || normalized.equals("llama-cpp")
                    || normalized.equals("java")
                    || normalized.equals("java-native")
                    || normalized.equals("jvm");
        }

        private boolean hasNonGgufProvider() {
            if (provider == null || provider.isBlank()) {
                return false;
            }
            return !supported();
        }

        private EngineMode engineMode() {
            return EngineMode.effective(provider, engine);
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
                    case "--provider" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.provider = next.value();
                        i = next.index();
                    }
                    case "--backend", "--platform" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.backend = next.value();
                        i = next.index();
                    }
                    case "--gguf-engine", "--engine" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.engine = next.value();
                        i = next.index();
                    }
                    case "--java-native", "--java" -> parsed.engine = "java";
                    case "--llamacpp", "--llama-cpp" -> parsed.engine = "llamacpp";
                    case "--benchmark", "--bench" -> parsed.engine = "benchmark";
                    case "--no-banner", "--suppress-banner" -> parsed.banner = false;
                    case "--use-cpu" -> parsed.backend = "cpu";
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
                    default -> {
                        if (arg.startsWith("--")) {
                            parsed.supported = false;
                        }
                    }
                }
            }
            parsed.maxTokens = Math.max(1, parsed.maxTokens);
            parsed.topK = Math.max(1, parsed.topK);
            parsed.topP = Math.max(0.0d, Math.min(1.0d, parsed.topP));
            parsed.temperature = Math.max(0.0d, parsed.temperature);
            return parsed;
        }

        private static Value valueOrNext(String value, String[] args, int index) {
            if (value != null) {
                return new Value(value, index);
            }
            if (index + 1 >= args.length) {
                return new Value("", index);
            }
            return new Value(args[index + 1], index + 1);
        }

        private static int parseInt(String value, int fallback) {
            try {
                return Integer.parseInt(value);
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

        private record Value(String value, int index) {
        }
    }

    private static String firstNonBlank(String first, String fallback) {
        if (first != null && !first.isBlank()) {
            return first;
        }
        return fallback;
    }

    private static String firstNonBlank(String first, String second, String fallback) {
        if (first != null && !first.isBlank()) {
            return first;
        }
        if (second != null && !second.isBlank()) {
            return second;
        }
        return fallback;
    }
}
