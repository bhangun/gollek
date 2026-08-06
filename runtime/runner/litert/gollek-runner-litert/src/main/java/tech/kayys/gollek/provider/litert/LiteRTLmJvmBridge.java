package tech.kayys.gollek.provider.litert;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.regex.Pattern;

/**
 * Optional bridge to Google's official LiteRT-LM JVM engine.
 *
 * <p>This is the real macOS Metal path for {@code .litertlm}: the Google JVM
 * artifact owns the LiteRT-LM engine, KV cache, prompt templating, and backend
 * selection. The older raw-signature fallback below this bridge is kept only as
 * a no-extra-dependency fallback because driving the container's TFLite
 * signatures directly is both slower and unsafe with the current Metal
 * accelerator on this export.</p>
 */
final class LiteRTLmJvmBridge implements AutoCloseable {

    static final String ENABLE_PROPERTY = "gollek.litert.enable_official_litertlm_jvm_bridge";
    static final String BACKEND_PROPERTY = "gollek.litert.lm_jvm_bridge.backend";
    static final String CACHE_DIR_PROPERTY = "gollek.litert.lm_jvm_bridge.cache_dir";
    static final String TIMEOUT_SECONDS_PROPERTY = "gollek.litert.lm_jvm_bridge.timeout_seconds";
    static final String MAX_NUM_TOKENS_PROPERTY = "gollek.litert.lm_jvm_bridge.max_num_tokens";
    static final String NORMALIZE_SHORT_QUESTIONS_PROPERTY =
            "gollek.litert.lm_jvm_bridge.normalize_short_questions";
    static final String ENABLE_SPECULATIVE_DECODING_PROPERTY =
            "gollek.litert.lm_jvm_bridge.enable_speculative_decoding";
    private static final int DEFAULT_GEMMA4_MAX_NUM_TOKENS = 2048;
    private static final Pattern BARE_QUESTION_PATTERN = Pattern.compile(
            "(?i)^(who|what|where|when|why|how|which|whom|whose|is|are|was|were|do|does|did|can|could|should|would|will|may|might)\\b.*[^.?!]$");
    private static final String SHORT_QUESTION_INSTRUCTION =
            "Answer directly and concisely in one or two sentences. Do not ask a clarification question. "
                    + "If a term is ambiguous, answer the most common meaning first and mention common alternatives briefly.\n"
                    + "Question: ";

    private static final org.slf4j.Logger log =
            org.slf4j.LoggerFactory.getLogger(LiteRTLmJvmBridge.class);
    private static final String PACKAGE_NAME = "com.google.ai.edge.litertlm.";
    private static volatile Boolean available;

    private final Path modelPath;
    private final LiteRTTokenizer tokenizer;
    private final boolean useGpu;
    private final int numThreads;

    private Object engine;
    private boolean initialized;

    LiteRTLmJvmBridge(Path modelPath, LiteRTTokenizer tokenizer, boolean useGpu, int numThreads) {
        this.modelPath = modelPath;
        this.tokenizer = tokenizer;
        this.useGpu = useGpu;
        this.numThreads = numThreads;
    }

    static boolean enabled() {
        return !"false".equalsIgnoreCase(System.getProperty(ENABLE_PROPERTY, "true"));
    }

    static boolean available() {
        if (!enabled()) {
            return false;
        }
        Boolean cached = available;
        if (cached != null) {
            return cached;
        }
        boolean detected;
        try {
            classLoader().loadClass(PACKAGE_NAME + "Engine");
            classLoader().loadClass(PACKAGE_NAME + "EngineConfig");
            classLoader().loadClass(PACKAGE_NAME + "Backend");
            detected = true;
        } catch (ClassNotFoundException | LinkageError e) {
            detected = false;
        }
        available = detected;
        return detected;
    }

    static String backendName(boolean useGpu) {
        String configured = System.getProperty(BACKEND_PROPERTY);
        if (configured == null || configured.isBlank()) {
            return useGpu ? "GPU" : "CPU";
        }
        String normalized = configured.trim().toUpperCase(Locale.ROOT);
        return normalized.equals("METAL") ? "GPU" : normalized;
    }

    static String promptForModel(String prompt) {
        if (prompt == null) {
            return "";
        }
        if (!shortQuestionNormalizationEnabled()) {
            return prompt;
        }
        String stripped = prompt.strip();
        if (stripped.isEmpty() || !BARE_QUESTION_PATTERN.matcher(stripped).matches()) {
            return prompt;
        }
        return SHORT_QUESTION_INSTRUCTION + Character.toUpperCase(stripped.charAt(0)) + stripped.substring(1) + "?";
    }

    static boolean shortQuestionNormalizationEnabled() {
        String configured = System.getProperty(NORMALIZE_SHORT_QUESTIONS_PROPERTY);
        return configured == null || configured.isBlank() || Boolean.parseBoolean(configured);
    }

    static Duration timeout() {
        String configured = System.getProperty(TIMEOUT_SECONDS_PROPERTY);
        if (configured == null || configured.isBlank()) {
            return Duration.ofSeconds(180);
        }
        try {
            long seconds = Long.parseLong(configured.trim());
            return Duration.ofSeconds(Math.max(5, seconds));
        } catch (NumberFormatException e) {
            log.warn("Ignoring invalid {} value '{}'; using 180 seconds",
                    TIMEOUT_SECONDS_PROPERTY, configured);
            return Duration.ofSeconds(180);
        }
    }

    static Path cacheDir() {
        String configured = System.getProperty(CACHE_DIR_PROPERTY);
        if (configured != null && !configured.isBlank()) {
            return Path.of(configured.trim());
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "cache", "litert-lm-jvm");
    }

    static Integer maxNumTokens(Path modelPath) {
        String configured = System.getProperty(MAX_NUM_TOKENS_PROPERTY);
        if (configured == null || configured.isBlank() || "auto".equalsIgnoreCase(configured.trim())) {
            return looksLikeGemma4(modelPath) ? DEFAULT_GEMMA4_MAX_NUM_TOKENS : null;
        }
        String normalized = configured.trim().toLowerCase(Locale.ROOT);
        if ("model".equals(normalized) || "default".equals(normalized) || "none".equals(normalized)) {
            return null;
        }
        try {
            int value = Integer.parseInt(normalized);
            return value > 0 ? value : null;
        } catch (NumberFormatException e) {
            log.warn("Ignoring invalid {} value '{}'; using {}",
                    MAX_NUM_TOKENS_PROPERTY,
                    configured,
                    looksLikeGemma4(modelPath) ? DEFAULT_GEMMA4_MAX_NUM_TOKENS : "model default");
            return looksLikeGemma4(modelPath) ? DEFAULT_GEMMA4_MAX_NUM_TOKENS : null;
        }
    }

    static Boolean speculativeDecodingPreference(boolean useGpu, Path modelPath) {
        String configured = System.getProperty(ENABLE_SPECULATIVE_DECODING_PROPERTY);
        if (configured != null && !configured.isBlank()) {
            String normalized = configured.trim().toLowerCase(Locale.ROOT);
            if ("default".equals(normalized) || "auto".equals(normalized)) {
                return useGpu && looksLikeGemma4(modelPath) ? Boolean.TRUE : null;
            }
            return Boolean.parseBoolean(normalized);
        }
        return useGpu && looksLikeGemma4(modelPath) ? Boolean.TRUE : null;
    }

    void initialize() {
        if (initialized) {
            return;
        }
        Boolean speculative = speculativeDecodingPreference(useGpu, modelPath);
        try {
            initializeEngine(speculative);
        } catch (Exception firstFailure) {
            closeQuietly();
            if (!Boolean.TRUE.equals(speculative)) {
                throw new IllegalStateException("official LiteRT-LM JVM engine init failed: "
                        + rootMessage(firstFailure), firstFailure);
            }
            log.warn("Official LiteRT-LM JVM engine failed with speculative decoding enabled; "
                    + "retrying without MTP: {}", rootMessage(firstFailure));
            try {
                initializeEngine(Boolean.FALSE);
            } catch (Exception secondFailure) {
                closeQuietly();
                throw new IllegalStateException("official LiteRT-LM JVM engine init failed: "
                        + rootMessage(secondFailure), secondFailure);
            }
        }
        initialized = true;
    }

    void generate(InferenceRequest request, Consumer<String> tokenCallback) {
        if (!initialized || engine == null) {
            throw new IllegalStateException("official LiteRT-LM JVM bridge is not initialized");
        }
        Object conversation = null;
        try {
            conversation = createConversation(request);
            streamGenerate(conversation,
                    extractPrompt(request),
                    Math.max(1, request.getMaxTokens()),
                    tokenCallback);
        } catch (InvocationTargetException e) {
            throw new IllegalStateException("official LiteRT-LM JVM bridge failed: "
                    + rootMessage(e.getTargetException()), e.getTargetException());
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("official LiteRT-LM JVM bridge failed: "
                    + rootMessage(e), e);
        } finally {
            closeAutoCloseable(conversation);
        }
    }

    private void initializeEngine(Boolean speculativeDecoding) throws Exception {
        Files.createDirectories(cacheDir());
        setNativeMinLogSeverityToError();
        setSpeculativeDecoding(speculativeDecoding);

        Class<?> engineConfigClass = litertLmClass("EngineConfig");
        Object backend = createBackend();
        Integer maxNumTokens = maxNumTokens(modelPath);
        Object config = constructorWithParameterCount(engineConfigClass, 7).newInstance(
                modelPath.toAbsolutePath().toString(),
                backend,
                null,
                null,
                maxNumTokens,
                null,
                cacheDir().toAbsolutePath().toString());
        Class<?> engineClass = litertLmClass("Engine");
        engine = constructorWithParameterCount(engineClass, 1).newInstance(config);
        engineClass.getMethod("initialize").invoke(engine);
        log.warn("Using official LiteRT-LM JVM engine for {} (backend={}, speculativeDecoding={}, maxNumTokens={}).",
                modelPath.getFileName(),
                backendName(useGpu),
                speculativeDecoding == null ? "model-default" : speculativeDecoding,
                maxNumTokens == null ? "model-default" : maxNumTokens);
    }

    private Object createBackend() throws ReflectiveOperationException {
        String backend = backendName(useGpu);
        return switch (backend) {
            case "GPU" -> constructorWithParameterCount(litertLmClass("Backend$GPU"), 0).newInstance();
            case "NPU" -> constructorWithParameterCount(litertLmClass("Backend$NPU"), 1).newInstance("");
            case "CPU" -> createCpuBackend();
            default -> {
                log.warn("Unsupported {} value '{}'; using {}",
                        BACKEND_PROPERTY, backend, useGpu ? "GPU" : "CPU");
                yield useGpu
                        ? constructorWithParameterCount(litertLmClass("Backend$GPU"), 0).newInstance()
                        : createCpuBackend();
            }
        };
    }

    private Object createCpuBackend() throws ReflectiveOperationException {
        Integer threads = numThreads > 0 ? numThreads : null;
        return constructorWithParameterCount(litertLmClass("Backend$CPU"), 1).newInstance(threads);
    }

    private Object createConversation(InferenceRequest request) throws ReflectiveOperationException {
        Class<?> conversationConfigClass = litertLmClass("ConversationConfig");
        Object samplerConfig = createSamplerConfig(request);
        Object conversationConfig = constructorWithParameterCount(conversationConfigClass, 7).newInstance(
                null,
                List.of(),
                List.of(),
                samplerConfig,
                true,
                null,
                Map.of());
        return engine.getClass()
                .getMethod("createConversation", conversationConfigClass)
                .invoke(engine, conversationConfig);
    }

    private Object createSamplerConfig(InferenceRequest request) {
        try {
            int topK = Math.max(1, request.getTopK());
            double topP = Math.max(0.0d, Math.min(1.0d, request.getTopP()));
            double temperature = Math.max(0.0d, request.getTemperature());
            return constructorWithParameterCount(litertLmClass("SamplerConfig"), 4)
                    .newInstance(topK, topP, temperature, 0);
        } catch (Exception e) {
            log.warn("Failed to create LiteRT-LM sampler config; using model defaults: {}",
                    rootMessage(e));
            return null;
        }
    }

    private void streamGenerate(Object conversation,
                                String prompt,
                                int maxTokens,
                                Consumer<String> tokenCallback)
            throws ReflectiveOperationException {
        Class<?> callbackClass = litertLmClass("MessageCallback");
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Throwable> error = new AtomicReference<>();
        AtomicBoolean cancelledForLimit = new AtomicBoolean(false);
        StringBuilder accumulated = new StringBuilder();
        StringBuilder emitted = new StringBuilder();
        ApproximateTokenStreamLimiter approximateLimiter = tokenizer == null
                ? new ApproximateTokenStreamLimiter(maxTokens)
                : null;

        Object callback = Proxy.newProxyInstance(
                callbackClass.getClassLoader(),
                new Class<?>[]{callbackClass},
                (proxy, method, args) -> {
                    try {
                        switch (method.getName()) {
                            case "onMessage" -> {
                                String chunk = args != null && args.length > 0 && args[0] != null
                                        ? args[0].toString()
                                        : "";
                                emitLimitedChunk(conversation, chunk, maxTokens, accumulated, emitted,
                                        approximateLimiter, cancelledForLimit, done, tokenCallback);
                            }
                            case "onDone" -> done.countDown();
                            case "onError" -> {
                                if (!cancelledForLimit.get() && args != null && args.length > 0
                                        && args[0] instanceof Throwable throwable) {
                                    error.compareAndSet(null, throwable);
                                }
                                done.countDown();
                            }
                            default -> {
                                if (method.getDeclaringClass() == Object.class) {
                                    return method.invoke(this, args);
                                }
                            }
                        }
                    } catch (Throwable callbackFailure) {
                        error.compareAndSet(null, callbackFailure);
                        cancelConversation(conversation);
                        done.countDown();
                    }
                    return null;
                });

        conversation.getClass()
                .getMethod("sendMessageAsync", String.class, callbackClass, Map.class)
                .invoke(conversation, promptForModel(prompt), callback, Map.of());

        try {
            boolean finished = done.await(timeout().toMillis(), TimeUnit.MILLISECONDS);
            if (!finished) {
                cancelConversation(conversation);
                throw new IllegalStateException("official LiteRT-LM JVM bridge timed out after "
                        + timeout().toSeconds() + "s");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            cancelConversation(conversation);
            throw new IllegalStateException("official LiteRT-LM JVM bridge interrupted", e);
        }
        Throwable throwable = error.get();
        if (throwable != null && !cancelledForLimit.get()) {
            throw new IllegalStateException(rootMessage(throwable), throwable);
        }
    }

    private void emitLimitedChunk(Object conversation,
                                  String chunk,
                                  int maxTokens,
                                  StringBuilder accumulated,
                                  StringBuilder emitted,
                                  ApproximateTokenStreamLimiter approximateLimiter,
                                  AtomicBoolean cancelledForLimit,
                                  CountDownLatch done,
                                  Consumer<String> tokenCallback) {
        if (chunk == null || chunk.isEmpty() || cancelledForLimit.get()) {
            return;
        }
        if (approximateLimiter != null) {
            String delta = approximateLimiter.offer(chunk);
            if (!delta.isEmpty()) {
                tokenCallback.accept(delta);
            }
            if (approximateLimiter.atLimit()) {
                cancelledForLimit.set(true);
                cancelConversation(conversation);
                done.countDown();
            }
            return;
        }
        accumulated.append(chunk);
        String limited = limitToMaxTokens(accumulated.toString(), maxTokens);
        if (limited.length() > emitted.length()) {
            String delta = limited.substring(emitted.length());
            if (!delta.isEmpty()) {
                tokenCallback.accept(delta);
            }
            emitted.setLength(0);
            emitted.append(limited);
        }
        if (atMaxTokens(accumulated.toString(), maxTokens)) {
            cancelledForLimit.set(true);
            cancelConversation(conversation);
            done.countDown();
        }
    }

    private String limitToMaxTokens(String text, int maxTokens) {
        if (text == null || text.isEmpty() || maxTokens <= 0) {
            return text == null ? "" : text;
        }
        if (tokenizer != null) {
            int[] tokens = tokenizer.encode(text);
            if (tokens.length <= maxTokens) {
                return text;
            }
            return tokenizer.decode(Arrays.copyOf(tokens, maxTokens));
        }
        return limitByApproximateTokens(text, maxTokens);
    }

    private boolean atMaxTokens(String text, int maxTokens) {
        if (text == null || text.isEmpty() || maxTokens <= 0) {
            return false;
        }
        if (tokenizer != null) {
            return tokenizer.encode(text).length >= maxTokens;
        }
        return approximateTokenCount(text) >= maxTokens;
    }

    private static int approximateTokenCount(String text) {
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

    private void cancelConversation(Object conversation) {
        if (conversation == null) {
            return;
        }
        try {
            conversation.getClass().getMethod("cancelProcess").invoke(conversation);
        } catch (Exception ignored) {
            // Best effort: some versions may not support cancellation.
        }
    }

    private void setNativeMinLogSeverityToError() {
        try {
            Class<?> engineClass = litertLmClass("Engine");
            Class<?> severityClass = litertLmClass("LogSeverity");
            @SuppressWarnings({"rawtypes", "unchecked"})
            Object error = Enum.valueOf((Class<? extends Enum>) severityClass.asSubclass(Enum.class), "ERROR");
            Object companion = engineClass.getField("Companion").get(null);
            companion.getClass().getMethod("setNativeMinLogSeverity", severityClass)
                    .invoke(companion, error);
        } catch (Exception e) {
            log.debug("Failed to set LiteRT-LM native log severity: {}", rootMessage(e));
        }
    }

    private void setSpeculativeDecoding(Boolean enabled) {
        try {
            Class<?> flagsClass = litertLmClass("ExperimentalFlags");
            Object instance = flagsClass.getField("INSTANCE").get(null);
            flagsClass.getMethod("setEnableSpeculativeDecoding", Boolean.class)
                    .invoke(instance, enabled);
        } catch (Exception e) {
            log.debug("Failed to set LiteRT-LM speculative decoding flag: {}", rootMessage(e));
        }
    }

    private static Constructor<?> constructorWithParameterCount(Class<?> type, int parameterCount)
            throws NoSuchMethodException {
        for (Constructor<?> constructor : type.getConstructors()) {
            if (constructor.getParameterCount() == parameterCount) {
                return constructor;
            }
        }
        throw new NoSuchMethodException(type.getName() + " constructor with "
                + parameterCount + " parameter(s)");
    }

    private static Class<?> litertLmClass(String simpleName) throws ClassNotFoundException {
        return classLoader().loadClass(PACKAGE_NAME + simpleName);
    }

    private static ClassLoader classLoader() {
        ClassLoader context = Thread.currentThread().getContextClassLoader();
        return context != null ? context : LiteRTLmJvmBridge.class.getClassLoader();
    }

    private static boolean looksLikeGemma4(Path path) {
        if (path == null || path.getFileName() == null) {
            return false;
        }
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.contains("gemma-4") || name.contains("gemma4");
    }

    private static String extractPrompt(InferenceRequest request) {
        String prompt = request.getPrompt();
        if (prompt != null && !prompt.isBlank()) {
            return prompt;
        }
        List<Message> messages = request.getMessages();
        if (messages == null || messages.isEmpty()) {
            return "";
        }
        List<String> parts = new ArrayList<>(messages.size());
        for (Message message : messages) {
            if (message.getContent() != null && !message.getContent().isBlank()) {
                parts.add(message.getRole().name().toLowerCase(Locale.ROOT) + ": " + message.getContent());
            }
        }
        return String.join("\n", parts);
    }

    private static String rootMessage(Throwable throwable) {
        Throwable current = throwable;
        while (current instanceof InvocationTargetException invocationTargetException
                && invocationTargetException.getTargetException() != null) {
            current = invocationTargetException.getTargetException();
        }
        String message = current.getMessage();
        return message == null || message.isBlank()
                ? current.getClass().getSimpleName()
                : message;
    }

    private static void closeAutoCloseable(Object object) {
        if (object == null) {
            return;
        }
        try {
            object.getClass().getMethod("close").invoke(object);
        } catch (InvocationTargetException e) {
            log.debug("Ignoring LiteRT-LM close failure: {}", rootMessage(e.getTargetException()));
        } catch (Exception e) {
            log.debug("Ignoring LiteRT-LM close failure: {}", rootMessage(e));
        }
    }

    private void closeQuietly() {
        try {
            close();
        } catch (Exception ignored) {
            // Best effort during initialization fallback.
        }
    }

    @Override
    public void close() {
        if (engine != null) {
            closeAutoCloseable(engine);
            engine = null;
            initialized = false;
        }
    }
}
