package tech.kayys.gollek.provider.litert;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Optional bridge to Google's official LiteRT-LM Python engine.
 *
 * <p>The manual {@link LiteRTGemmaNativeRunner} path drives the raw TFLite
 * signatures directly and must copy full KV-cache tensors through Java for each
 * token. The official LiteRT-LM engine owns that decode state and is the path
 * Google publishes performance numbers for on Apple Silicon. This bridge is
 * intentionally optional: when {@code litert_lm} is not installed, callers can
 * continue using the current native-signature fallback.</p>
 */
final class LiteRTLmPythonBridge {

    static final String ENABLE_PROPERTY = "gollek.litert.enable_official_litertlm_bridge";
    static final String PYTHON_PROPERTY = "gollek.litert.lm_bridge.python";
    static final String BACKEND_PROPERTY = "gollek.litert.lm_bridge.backend";
    static final String CACHE_DIR_PROPERTY = "gollek.litert.lm_bridge.cache_dir";
    static final String TIMEOUT_SECONDS_PROPERTY = "gollek.litert.lm_bridge.timeout_seconds";

    private static final org.slf4j.Logger log =
            org.slf4j.LoggerFactory.getLogger(LiteRTLmPythonBridge.class);
    private static final String RESPONSE_MARKER = "__GOLLEK_LITERTLM_RESPONSE_B64__=";
    private static final Map<String, Boolean> AVAILABILITY = new ConcurrentHashMap<>();
    private static final String PYTHON_DRIVER_TEMPLATE = """
            import base64
            import os
            import sys
            import traceback

            model_path = sys.argv[1]
            prompt = base64.b64decode(sys.argv[2]).decode("utf-8")
            backend_name = sys.argv[3].upper()
            cache_dir = sys.argv[4]

            try:
                import litert_lm

                try:
                    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
                except Exception:
                    pass

                backend = getattr(litert_lm.Backend, backend_name, litert_lm.Backend.CPU)

                def response_text(response):
                    if isinstance(response, str):
                        return response
                    if hasattr(response, "text"):
                        return str(response.text)
                    if hasattr(response, "content"):
                        content = response.content
                    elif isinstance(response, dict):
                        content = response.get("content", response)
                    else:
                        return str(response)
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        pieces = []
                        for item in content:
                            if isinstance(item, str):
                                pieces.append(item)
                            elif isinstance(item, dict) and "text" in item:
                                pieces.append(str(item["text"]))
                            elif hasattr(item, "text"):
                                pieces.append(str(item.text))
                        return "".join(pieces)
                    return str(content)

                os.makedirs(cache_dir, exist_ok=True)
                with litert_lm.Engine(model_path, backend=backend, cache_dir=cache_dir) as engine:
                    with engine.create_conversation() as conversation:
                        result = conversation.send_message(prompt)
                text = response_text(result)
                encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
                print("__GOLLEK_RESPONSE_MARKER__" + encoded, flush=True)
            except Exception:
                traceback.print_exc()
                sys.exit(42)
            """;
    private static final String PYTHON_DRIVER =
            PYTHON_DRIVER_TEMPLATE.replace("__GOLLEK_RESPONSE_MARKER__", RESPONSE_MARKER);

    private final Path modelPath;
    private final LiteRTTokenizer tokenizer;
    private final boolean useGpu;

    LiteRTLmPythonBridge(Path modelPath, LiteRTTokenizer tokenizer, boolean useGpu) {
        this.modelPath = modelPath;
        this.tokenizer = tokenizer;
        this.useGpu = useGpu;
    }

    static boolean enabled() {
        return !"false".equalsIgnoreCase(System.getProperty(ENABLE_PROPERTY, "true"));
    }

    static String pythonCommand() {
        return System.getProperty(PYTHON_PROPERTY, "python3").trim();
    }

    static String backendName(boolean useGpu) {
        String configured = System.getProperty(BACKEND_PROPERTY);
        if (configured == null || configured.isBlank()) {
            return useGpu ? "GPU" : "CPU";
        }
        String normalized = configured.trim().toUpperCase(Locale.ROOT);
        return normalized.equals("METAL") ? "GPU" : normalized;
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
        return Path.of(System.getProperty("user.home"), ".gollek", "cache", "litert-lm");
    }

    static boolean available() {
        if (!enabled()) {
            return false;
        }
        String python = pythonCommand();
        if (python.isBlank()) {
            return false;
        }
        return AVAILABILITY.computeIfAbsent(python, LiteRTLmPythonBridge::probePythonModule);
    }

    void generate(InferenceRequest request, Consumer<String> tokenCallback) {
        String text = runOfficialEngine(extractPrompt(request), Math.max(1, request.getMaxTokens()));
        if (!text.isEmpty()) {
            tokenCallback.accept(text);
        }
    }

    private String runOfficialEngine(String prompt, int maxTokens) {
        Path driver = null;
        try {
            Files.createDirectories(cacheDir());
            driver = Files.createTempFile("gollek-litert-lm-", ".py");
            Files.writeString(driver, PYTHON_DRIVER, StandardCharsets.UTF_8);

            String encodedPrompt = Base64.getEncoder()
                    .encodeToString(prompt.getBytes(StandardCharsets.UTF_8));
            Process process = new ProcessBuilder(
                    pythonCommand(),
                    driver.toAbsolutePath().toString(),
                    modelPath.toAbsolutePath().toString(),
                    encodedPrompt,
                    backendName(useGpu),
                    cacheDir().toAbsolutePath().toString())
                    .redirectErrorStream(true)
                    .start();
            boolean finished = process.waitFor(timeout().toMillis(), TimeUnit.MILLISECONDS);
            if (!finished) {
                process.destroyForcibly();
                throw new IllegalStateException("official LiteRT-LM bridge timed out after "
                        + timeout().toSeconds() + "s");
            }
            String output = readProcessOutput(process);
            if (process.exitValue() != 0) {
                throw new IllegalStateException("official LiteRT-LM bridge exited with "
                        + process.exitValue() + ": " + output.strip());
            }
            return limitToMaxTokens(parseResponse(output), maxTokens);
        } catch (IOException e) {
            throw new IllegalStateException("official LiteRT-LM bridge failed to start: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("official LiteRT-LM bridge interrupted", e);
        } finally {
            if (driver != null) {
                try {
                    Files.deleteIfExists(driver);
                } catch (IOException ignored) {
                    // Best-effort cleanup only.
                }
            }
        }
    }

    private String limitToMaxTokens(String text, int maxTokens) {
        if (text == null || text.isEmpty() || maxTokens <= 0) {
            return text == null ? "" : text;
        }
        int[] tokens = tokenizer.encode(text);
        if (tokens.length <= maxTokens) {
            return text;
        }
        return tokenizer.decode(Arrays.copyOf(tokens, maxTokens));
    }

    private static String parseResponse(String output) {
        int marker = output.lastIndexOf(RESPONSE_MARKER);
        if (marker < 0) {
            throw new IllegalStateException("official LiteRT-LM bridge did not emit a response marker: "
                    + output.strip());
        }
        int valueStart = marker + RESPONSE_MARKER.length();
        int valueEnd = output.indexOf('\n', valueStart);
        String encoded = (valueEnd >= 0 ? output.substring(valueStart, valueEnd) : output.substring(valueStart))
                .trim();
        return new String(Base64.getDecoder().decode(encoded), StandardCharsets.UTF_8);
    }

    private static String readProcessOutput(Process process) {
        try {
            return new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            return "<failed reading bridge output: " + e.getMessage() + ">";
        }
    }

    private static boolean probePythonModule(String python) {
        try {
            Process process = new ProcessBuilder(
                    python,
                    "-c",
                    "import litert_lm; print('ok')")
                    .redirectErrorStream(true)
                    .start();
            boolean finished = process.waitFor(5, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                return false;
            }
            boolean available = process.exitValue() == 0;
            if (!available) {
                log.debug("Official LiteRT-LM Python package is not available through '{}'", python);
            }
            return available;
        } catch (IOException e) {
            log.debug("Official LiteRT-LM Python probe failed for '{}': {}", python, e.getMessage());
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
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
}
