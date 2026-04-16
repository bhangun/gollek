package tech.kayys.gollek.onnx.runner;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.model.ArtifactLocation;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class OnnxHuggingFaceIntegrationTest {

    @Test
    void downloadsOnnxFromHuggingFaceAndGenerates() throws Exception {
        String repoId = envOrProp("GOLLEK_ONNX_HF_REPO", "gollek.onnx.hf.repo");
        String libraryPath = resolveLibraryPath();
        assumeTrue(repoId != null && !repoId.isBlank(), "Set GOLLEK_ONNX_HF_REPO to run this test");
        assumeTrue(libraryPath != null && !libraryPath.isBlank(), "Set GOLLEK_ONNX_LIBRARY_PATH to run this test");
        Path libPath = Path.of(libraryPath);
        assumeTrue(Files.exists(libPath) && Files.isRegularFile(libPath) && Files.isReadable(libPath),
                "Set GOLLEK_ONNX_LIBRARY_PATH to a valid ONNX Runtime library");
        preflightNetworkAccess();

        String modelId = envOrProp("GOLLEK_ONNX_HF_MODEL_ID", "gollek.onnx.hf.model-id");
        if (modelId == null || modelId.isBlank()) {
            modelId = "hf-onnx-test";
        }

        Path modelPath = downloadOnnxFromRepo(repoId);
        long size = Files.size(modelPath);

        ArtifactLocation artifact = new ArtifactLocation(modelPath.toString(), null, size, "application/octet-stream");
        ModelManifest manifest = ModelManifest.builder()
                .modelId(modelId)
                .name(modelId)
                .version("hf")
                .path(modelPath.toString())
                .apiKey("community")
                .requestId("hf-test-" + System.currentTimeMillis())
                .artifacts(Map.of(ModelFormat.ONNX, artifact))
                .metadata(Map.of("source", "huggingface", "repo", repoId))
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();

        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        setField(runner, "enabled", true);
        setField(runner, "libraryPath", libraryPath);
        setField(runner, "executionProvider",
                envOrProp("GOLLEK_ONNX_EXECUTION_PROVIDER", "gollek.onnx.execution-provider", "cpu"));
        setField(runner, "intraOpThreads",
                Integer.parseInt(envOrProp("GOLLEK_ONNX_INTRA_OP_THREADS", "gollek.onnx.intra-op-threads", "2")));
        setField(runner, "interOpThreads",
                Integer.parseInt(envOrProp("GOLLEK_ONNX_INTER_OP_THREADS", "gollek.onnx.inter-op-threads", "1")));
        setField(runner, "graphOptLevel",
                Integer.parseInt(envOrProp("GOLLEK_ONNX_GRAPH_OPT_LEVEL", "gollek.onnx.graph-opt-level", "99")));
        setField(runner, "usePastKvCache",
                Boolean.parseBoolean(envOrProp("GOLLEK_ONNX_USE_PAST_KV", "gollek.onnx.use-past-kv-cache", "false")));
        setField(runner, "vocabSize",
                Integer.parseInt(envOrProp("GOLLEK_ONNX_VOCAB_SIZE", "gollek.onnx.vocab-size", "50257")));

        RunnerConfiguration config = RunnerConfiguration.builder()
                .putParameter("execution_provider",
                        envOrProp("GOLLEK_ONNX_EXECUTION_PROVIDER", "gollek.onnx.execution-provider", "cpu"))
                .putParameter("vocab_size",
                        Integer.parseInt(envOrProp("GOLLEK_ONNX_VOCAB_SIZE", "gollek.onnx.vocab-size", "50257")))
                .putParameter("use_past_kv_cache",
                        Boolean.parseBoolean(
                                envOrProp("GOLLEK_ONNX_USE_PAST_KV", "gollek.onnx.use-past-kv-cache", "false")))
                .build();

        runner.initialize(manifest, config);

        InferenceRequest request = InferenceRequest.builder()
                .model(modelId)
                .message(Message.user("Hello"))
                .parameter("max_tokens", 8)
                .build();

        var response = runner.infer(request);
        assertNotNull(response);
        assertNotNull(response.getContent());
        assertTrue(!response.getContent().isBlank(), "Expected non-empty response content");
    }

    private static Path downloadOnnxFromRepo(String repoId) throws Exception {
        Path cacheDir = resolveCacheDir();
        Files.createDirectories(cacheDir);
        String fileName = resolveOnnxFilename(repoId);
        Path target = cacheDir.resolve(fileName);
        if (Files.exists(target) && Files.size(target) > 0) {
            return target;
        }
        tech.kayys.gollek.model.repo.hf.HuggingFaceClient client = createHfClient();
        boolean required = isRequired();
        try {
            client.downloadFile(repoId, fileName, target, null);
        } catch (java.io.IOException e) {
            if (isUnauthorized(e) && !hasToken() && !required) {
                assumeTrue(false, "HuggingFace returned 401 for repo " + repoId + ". Set GOLLEK_HF_TOKEN or HF_TOKEN.");
            }
            if (isNotFound(e) && !required) {
                assumeTrue(false, "HuggingFace repo or file not found: " + repoId + "/" + fileName);
            }
            if (isNetworkError(e) && !required) {
                assumeTrue(false, "Network unavailable for HuggingFace download: " + e.getClass().getSimpleName());
            }
            throw e;
        }
        return target;
    }

    private static String resolveOnnxFilename(String repoId) throws Exception {
        String configured = envOrProp("GOLLEK_ONNX_HF_FILENAME", "gollek.onnx.hf.filename");
        if (configured != null && !configured.isBlank()) {
            return configured.trim();
        }
        tech.kayys.gollek.model.repo.hf.HuggingFaceClient client = createHfClient();
        boolean required = isRequired();
        java.util.List<String> files;
        try {
            files = client.listFiles(repoId);
        } catch (java.io.IOException e) {
            if (isUnauthorized(e) && !hasToken() && !required) {
                assumeTrue(false, "HuggingFace returned 401 for repo " + repoId + ". Set GOLLEK_HF_TOKEN or HF_TOKEN.");
            }
            if (isNotFound(e) && !required) {
                assumeTrue(false, "HuggingFace repo not found: " + repoId);
            }
            if (isNetworkError(e) && !required) {
                assumeTrue(false, "Network unavailable for HuggingFace list files: " + e.getClass().getSimpleName());
            }
            throw e;
        }
        Optional<String> onnx = files.stream()
                .filter(name -> name.toLowerCase(Locale.ROOT).endsWith(".onnx"))
                .findFirst();
        assumeTrue(onnx.isPresent(), "No .onnx file found in repo " + repoId);
        return onnx.get();
    }

    private static Path resolveCacheDir() {
        String configured = envOrProp("GOLLEK_ONNX_HF_CACHE_DIR", "gollek.onnx.hf.cache-dir");
        if (configured != null && !configured.isBlank()) {
            return Path.of(configured);
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "cache", "onnx");
    }

    private static String resolveLibraryPath() {
        String configured = envOrProp("GOLLEK_ONNX_LIBRARY_PATH", "gollek.onnx.library-path");
        if (configured != null && !configured.isBlank()) {
            Path configuredPath = Path.of(configured);
            if (Files.exists(configuredPath) && Files.isReadable(configuredPath)) {
                return configured;
            }
        }
        String osName = System.getProperty("os.name", "").toLowerCase();
        boolean isMac = osName.contains("mac");
        String libName = isMac ? "libonnxruntime.dylib" : "libonnxruntime.so";
        Path fallback = Path.of(System.getProperty("user.home"), ".gollek", "libs", libName);
        if (Files.exists(fallback)) {
            return fallback.toString();
        }
        return configured;
    }

    private static tech.kayys.gollek.model.repo.hf.HuggingFaceClient createHfClient() throws Exception {
        tech.kayys.gollek.model.repo.hf.HuggingFaceClient client = new tech.kayys.gollek.model.repo.hf.HuggingFaceClient();
        setField(client, "config", new EnvHfConfig());
        setField(client, "objectMapper", new com.fasterxml.jackson.databind.ObjectMapper());
        return client;
    }

    private static boolean isUnauthorized(Exception e) {
        String message = e.getMessage();
        return message != null && message.contains("401");
    }

    private static boolean isNotFound(Exception e) {
        String message = e.getMessage();
        return message != null && message.contains("404");
    }

    private static boolean isNetworkError(Exception e) {
        Throwable current = e;
        while (current != null) {
            if (current instanceof java.net.ConnectException
                    || current instanceof java.net.UnknownHostException
                    || current instanceof java.nio.channels.UnresolvedAddressException) {
                return true;
            }
            String message = current.getMessage();
            if (message != null && message.contains("UnresolvedAddress")) {
                return true;
            }
            current = current.getCause();
        }
        return false;
    }

    private static boolean isRequired() {
        return Boolean.parseBoolean(envOrProp("GOLLEK_ONNX_HF_REQUIRED", "gollek.onnx.hf.required", "false"));
    }

    private static void preflightNetworkAccess() {
        if (!isRequired()) {
            return;
        }
        try {
            java.net.InetAddress.getByName("huggingface.co");
        } catch (Exception e) {
            throw new AssertionError("HuggingFace DNS resolution failed in strict mode. Check network/DNS.", e);
        }
    }

    private static boolean hasToken() {
        String token = envOrProp("GOLLEK_HF_TOKEN", "gollek.hf.token");
        if (token == null || token.isBlank()) {
            token = System.getenv("HF_TOKEN");
        }
        if (token == null || token.isBlank()) {
            token = System.getenv("HUGGINGFACE_TOKEN");
        }
        return token != null && !token.isBlank();
    }

    private static void setField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static String envOrProp(String envKey, String propKey) {
        return envOrProp(envKey, propKey, null);
    }

    private static String envOrProp(String envKey, String propKey, String fallback) {
        String env = System.getenv(envKey);
        if (env != null && !env.isBlank()) {
            return env.trim();
        }
        String prop = System.getProperty(propKey);
        if (prop != null && !prop.isBlank()) {
            return prop.trim();
        }
        return fallback;
    }

    private static final class EnvHfConfig implements tech.kayys.gollek.model.repo.hf.HuggingFaceConfig {
        @Override
        public String baseUrl() {
            return "https://huggingface.co";
        }

        @Override
        public Optional<String> token() {
            String token = envOrProp("GOLLEK_HF_TOKEN", "gollek.hf.token");
            if (token == null || token.isBlank()) {
                token = System.getenv("HF_TOKEN");
            }
            if (token == null || token.isBlank()) {
                token = System.getenv("HUGGINGFACE_TOKEN");
            }
            if (token == null || token.isBlank()) {
                return Optional.empty();
            }
            return Optional.of(token.trim());
        }

        @Override
        public int timeoutSeconds() {
            return 30;
        }

        @Override
        public int maxRetries() {
            return 3;
        }

        @Override
        public boolean parallelDownload() {
            return false;
        }

        @Override
        public int parallelChunks() {
            return 1;
        }

        @Override
        public int chunkSizeMB() {
            return 10;
        }

        @Override
        public String userAgent() {
            return "gollek-inference/1.0";
        }

        @Override
        public boolean autoDownload() {
            return true;
        }

        @Override
        public String revision() {
            return "main";
        }
    }
}
