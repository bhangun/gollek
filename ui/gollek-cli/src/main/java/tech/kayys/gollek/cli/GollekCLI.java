package tech.kayys.gollek.cli;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.factory.GollekSdkFactory;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.session.ChatSession;
import tech.kayys.gollek.sdk.session.ChatSessionImpl;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;

@Command(name = "gollek", mixinStandardHelpOptions = true, version = "gollek 2.0-gradle", description = "Gollek Gradle-native local CLI", subcommands = {
        GollekCLI.ListCommand.class,
        GollekCLI.RunCommand.class,
        GollekCLI.ChatCommand.class
})
public class GollekCLI implements Callable<Integer> {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Path GOLLEK_HOME = Paths.get(System.getProperty("user.home"), ".gollek");
    private static final Path MODEL_INDEX = GOLLEK_HOME.resolve("models").resolve("manifest.json");
    private static final Path MODEL_MANIFESTS = GOLLEK_HOME.resolve("models").resolve("manifests");
    private static volatile GollekSdk sdk;

    @Override
    public Integer call() {
        CommandLine.usage(this, System.out);
        return 0;
    }

    public static void main(String[] args) {
        int exitCode = new CommandLine(new GollekCLI()).execute(args);
        System.exit(exitCode);
    }

    static synchronized GollekSdk getSdk() throws Exception {
        if (sdk == null) {
            sdk = GollekSdkFactory.createLocalSdk();
        }
        return sdk;
    }

    static List<ModelEntry> loadModels() throws IOException {
        if (Files.isDirectory(MODEL_MANIFESTS)) {
            List<ModelEntry> models = new ArrayList<>();
            try (var stream = Files.list(MODEL_MANIFESTS)) {
                stream.filter(path -> path.getFileName().toString().endsWith(".json"))
                        .sorted()
                        .forEach(path -> {
                            try {
                                Map<String, Object> raw = MAPPER.readValue(path.toFile(),
                                        new TypeReference<Map<String, Object>>() {
                                        });
                                models.add(ModelEntry.fromManifest(raw));
                            } catch (IOException ignored) {
                            }
                        });
            }
            if (!models.isEmpty()) {
                return models;
            }
        }

        if (Files.isRegularFile(MODEL_INDEX)) {
            return MAPPER.readValue(MODEL_INDEX.toFile(), new TypeReference<List<ModelEntry>>() {
            });
        }

        return List.of();
    }

    static List<ModelEntry> loadModelsFromSdk() {
        try {
            List<ModelInfo> infos = getSdk().listModels(0, 200);
            if (infos == null || infos.isEmpty()) {
                return List.of();
            }
            return infos.stream()
                    .map(ModelEntry::fromModelInfo)
                    .toList();
        } catch (Throwable ignored) {
            return List.of();
        }
    }

    static List<ModelEntry> loadModelsPreferSdk() throws IOException {
        List<ModelEntry> fromSdk = loadModelsFromSdk();
        if (!fromSdk.isEmpty()) {
            return fromSdk;
        }
        return loadModels();
    }

    static Optional<ModelEntry> resolveModel(String selector) throws IOException {
        String normalized = selector == null ? "" : selector.trim();
        if (normalized.isEmpty()) {
            return Optional.empty();
        }

        List<ModelEntry> models = new ArrayList<>(loadModels());
        models.sort(Comparator.comparing(ModelEntry::shortId, Comparator.nullsLast(String::compareTo)));

        return models.stream()
                .filter(model -> model.matches(normalized))
                .findFirst()
                .or(() -> models.stream()
                        .filter(model -> model.prefixMatches(normalized))
                        .findFirst());
    }

    static String defaultProvider(ModelEntry model) {
        if (model == null || model.format() == null) {
            return null;
        }
        return switch (model.format().trim().toLowerCase(Locale.ROOT)) {
            case "gguf" -> "gguf";
            case "litert", "tflite", "task" -> "litert";
            case "safetensor", "safetensors" -> "safetensor";
            case "onnx" -> "onnx";
            default -> null;
        };
    }

    static String logicalModelId(ModelEntry model) {
        if (model.modelId() != null && !model.modelId().isBlank()) {
            return model.modelId();
        }
        if (model.id() != null && !model.id().isBlank()) {
            return model.id();
        }
        if (model.blobPath() != null && !model.blobPath().isBlank()) {
            return model.blobPath();
        }
        return model.name();
    }

    static Map<String, Object> defaultParameters(ModelEntry model, double temperature, int maxTokens) {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("temperature", temperature);
        parameters.put("max_tokens", maxTokens);
        parameters.put("top_p", 0.9d);
        parameters.put("top_k", 40);
        parameters.put("repeat_penalty", 1.1d);
        parameters.put("repetition_penalty", 1.1d);
        if (model.blobPath() != null && !model.blobPath().isBlank()) {
            parameters.put("model_path", model.blobPath());
        }
        return parameters;
    }

    static InferenceRequest buildRequest(
            ModelEntry model,
            String prompt,
            String systemPrompt,
            String provider,
            double temperature,
            int maxTokens,
            boolean streaming) {
        InferenceRequest.Builder builder = InferenceRequest.builder()
                .requestId(UUID.randomUUID().toString())
                .model(logicalModelId(model))
                .temperature(temperature)
                .maxTokens(maxTokens)
                .topP(0.9d)
                .topK(40)
                .repeatPenalty(1.1d)
                .streaming(streaming);

        if (systemPrompt != null && !systemPrompt.isBlank()) {
            builder.message(Message.system(systemPrompt));
        }
        builder.message(Message.user(prompt));

        if (model.blobPath() != null && !model.blobPath().isBlank()) {
            builder.parameter("model_path", model.blobPath());
        }
        if (provider != null && !provider.isBlank()) {
            builder.preferredProvider(provider);
        }

        return builder.build();
    }

    static int runStreaming(Multi<StreamingInferenceChunk> stream) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        StringBuilder errorMessage = new StringBuilder();

        stream.subscribe().with(
                chunk -> {
                    String delta = chunk.getDelta();
                    if (delta != null && !delta.isEmpty()) {
                        System.out.print(delta);
                        System.out.flush();
                    }
                },
                error -> {
                    errorMessage.append(error.getMessage());
                    latch.countDown();
                },
                latch::countDown);

        latch.await();
        System.out.println();

        if (!errorMessage.isEmpty()) {
            System.err.println("Inference failed: " + errorMessage);
            return 1;
        }
        return 0;
    }

    @Command(name = "list", description = "List locally indexed models")
    static class ListCommand implements Callable<Integer> {
        @Option(names = { "--json" }, description = "Print JSON output")
        boolean json;

        @Override
        public Integer call() throws Exception {
            List<ModelEntry> models = loadModelsPreferSdk();
            if (models.isEmpty()) {
                System.out.println("No models found in " + MODEL_INDEX);
                return 0;
            }

            models = new ArrayList<>(models);
            models.sort(Comparator.comparing(ModelEntry::shortId, Comparator.nullsLast(String::compareTo)));

            if (json) {
                System.out.println(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(models));
                return 0;
            }

            System.out.printf("%-8s %-12s %-14s %s%n", "ID", "FORMAT", "SOURCE", "NAME");
            for (ModelEntry model : models) {
                System.out.printf(
                        "%-8s %-12s %-14s %s%n",
                        truncate(model.shortId(), 8),
                        truncate(model.format(), 12),
                        truncate(model.source(), 14),
                        model.name());
            }
            System.out.printf("%n%d model(s)%n", models.size());
            return 0;
        }
    }

    @Command(name = "run", description = "Run a prompt against a selected local model")
    static class RunCommand implements Callable<Integer> {
        @Option(names = { "-m", "--model" }, required = true, description = "Model short id, id, or name")
        String model;

        @Option(names = { "-p", "--prompt" }, required = true, description = "Prompt to send")
        String prompt;

        @Option(names = { "--system" }, description = "Optional system prompt")
        String systemPrompt;

        @Option(names = { "--provider" }, description = "Force provider id")
        String provider;

        @Option(names = { "-s", "--stream" }, description = "Stream output")
        boolean stream;

        @Option(names = { "--temperature" }, description = "Sampling temperature", defaultValue = "0.2")
        double temperature;

        @Option(names = { "--max-tokens" }, description = "Maximum tokens", defaultValue = "256")
        int maxTokens;

        @Option(names = { "--no-banner" }, description = "Suppress header and footer output")
        boolean noBanner;

        @Override
        public Integer call() throws Exception {
            Optional<ModelEntry> resolved = resolveModel(model);
            if (resolved.isEmpty()) {
                System.err.println("Model not found: " + model);
                return 1;
            }

            ModelEntry entry = resolved.get();
            String selectedProvider = provider != null && !provider.isBlank() ? provider : defaultProvider(entry);
            if (selectedProvider == null || selectedProvider.isBlank()) {
                System.err.println("No runnable provider could be inferred for model format: " + entry.format());
                return 1;
            }

            GollekSdk sdk = getSdk();
            sdk.setPreferredProvider(selectedProvider);

            InferenceRequest request = buildRequest(
                    entry,
                    prompt,
                    systemPrompt,
                    selectedProvider,
                    temperature,
                    maxTokens,
                    stream);

            Instant started = Instant.now();
            if (stream) {
                int status = runStreaming(sdk.streamCompletion(request));
                if (!noBanner) {
                    long elapsedMs = Math.max(1L, Duration.between(started, Instant.now()).toMillis());
                    System.out.printf("Completed in %d ms%n", elapsedMs);
                }
                return status;
            }

            InferenceResponse response = sdk.createCompletion(request);
            if (noBanner) {
                System.out.print(response.getContent());
            } else {
                System.out.println(response.getContent());
                long elapsedMs = Math.max(1L, Duration.between(started, Instant.now()).toMillis());
                System.out.printf("%nCompleted in %d ms%n", elapsedMs);
            }
            return 0;
        }
    }

    @Command(name = "chat", description = "Start an interactive local chat session")
    static class ChatCommand implements Callable<Integer> {
        @Option(names = { "-m", "--model" }, required = true, description = "Model short id, id, or name")
        String model;

        @Option(names = { "--system" }, description = "Optional system prompt")
        String systemPrompt;

        @Option(names = { "--provider" }, description = "Force provider id")
        String provider;

        @Option(names = { "--temperature" }, description = "Sampling temperature", defaultValue = "0.2")
        double temperature;

        @Option(names = { "--max-tokens" }, description = "Maximum tokens", defaultValue = "256")
        int maxTokens;

        @Option(names = { "--no-stream" }, description = "Disable streaming")
        boolean noStream;

        @Override
        public Integer call() throws Exception {
            Optional<ModelEntry> resolved = resolveModel(model);
            if (resolved.isEmpty()) {
                System.err.println("Model not found: " + model);
                return 1;
            }

            ModelEntry entry = resolved.get();
            String selectedProvider = provider != null && !provider.isBlank() ? provider : defaultProvider(entry);
            if (selectedProvider == null || selectedProvider.isBlank()) {
                System.err.println("No runnable provider could be inferred for model format: " + entry.format());
                return 1;
            }

            GollekSdk sdk = getSdk();
            sdk.setPreferredProvider(selectedProvider);

            ChatSession session = new ChatSessionImpl(sdk, logicalModelId(entry), selectedProvider);
            session.setAutoContinue(true);
            session.setDefaultParameters(defaultParameters(entry, temperature, maxTokens));
            if (systemPrompt != null && !systemPrompt.isBlank()) {
                session.setSystemPrompt(systemPrompt);
            }

            System.out.println("Chatting with " + entry.displayId() + " via " + selectedProvider);
            System.out.println("Type /exit to quit.");

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
                while (true) {
                    System.out.print("> ");
                    String line = reader.readLine();
                    if (line == null || "/exit".equalsIgnoreCase(line.trim())
                            || "/quit".equalsIgnoreCase(line.trim())) {
                        System.out.println("Bye.");
                        return 0;
                    }
                    if (line.trim().isEmpty()) {
                        continue;
                    }

                    if (noStream) {
                        InferenceResponse response = session.send(line);
                        System.out.println(response.getContent());
                    } else {
                        int status = runStreaming(session.stream(line));
                        if (status != 0) {
                            return status;
                        }
                    }
                }
            } finally {
                session.close();
            }
        }
    }

    static String truncate(String value, int max) {
        if (value == null || value.isBlank()) {
            return "-";
        }
        return value.length() <= max ? value : value.substring(0, Math.max(0, max - 3)) + "...";
    }

    record ModelEntry(
            String id,
            String modelId,
            String name,
            String format,
            String source,
            String shortId,
            String group,
            String origin,
            String blobPath) {

        static ModelEntry fromManifest(Map<String, Object> raw) {
            return new ModelEntry(
                    asText(raw.get("id")),
                    asText(raw.get("modelId")),
                    asText(raw.get("name")),
                    asText(raw.get("format")),
                    asText(raw.get("source")),
                    asText(raw.get("shortId")),
                    asText(raw.get("group")),
                    asText(raw.get("origin")),
                    asText(raw.get("blobPath")));
        }

        static ModelEntry fromModelInfo(ModelInfo info) {
            String path = null;
            if (info.getMetadata() != null) {
                Object rawPath = info.getMetadata().get("path");
                if (rawPath != null) {
                    path = String.valueOf(rawPath);
                }
            }
            return new ModelEntry(
                    info.getModelId(),
                    info.getModelId(),
                    info.getName() != null && !info.getName().isBlank() ? info.getName() : info.getModelId(),
                    info.getFormat(),
                    "sdk",
                    info.getShortId(),
                    null,
                    null,
                    path);
        }

        boolean matches(String selector) {
            return selector.equalsIgnoreCase(shortId)
                    || selector.equalsIgnoreCase(id)
                    || selector.equalsIgnoreCase(modelId)
                    || selector.equalsIgnoreCase(name);
        }

        boolean prefixMatches(String selector) {
            return startsWithIgnoreCase(shortId, selector)
                    || startsWithIgnoreCase(id, selector)
                    || startsWithIgnoreCase(modelId, selector)
                    || startsWithIgnoreCase(name, selector);
        }

        String displayId() {
            if (shortId != null && !shortId.isBlank()) {
                return shortId;
            }
            if (id != null && !id.isBlank()) {
                return id;
            }
            if (modelId != null && !modelId.isBlank()) {
                return modelId;
            }
            return name != null ? name : "-";
        }

        private static String asText(Object value) {
            if (value == null) {
                return null;
            }
            String text = String.valueOf(value).trim();
            return text.isEmpty() ? null : text;
        }

        private static boolean startsWithIgnoreCase(String value, String prefix) {
            return value != null && prefix != null
                    && value.toLowerCase(Locale.ROOT).startsWith(prefix.toLowerCase(Locale.ROOT));
        }
    }
}
