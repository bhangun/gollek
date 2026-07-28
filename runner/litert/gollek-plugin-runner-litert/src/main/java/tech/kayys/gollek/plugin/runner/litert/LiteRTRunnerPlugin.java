/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.plugin.runner.litert;

import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.*;
import tech.kayys.gollek.provider.litert.LiteRTEngine;
import tech.kayys.gollek.provider.litert.LiteRTProviderConfig;
import tech.kayys.gollek.provider.litert.LiteRTRuntimeDiagnostics;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;


import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;

/**
 * LiteRT runner plugin backed by the real LiteRT provider implementation.
 *
 * @since 2.1.0
 * @version 2.0.0
 */
public class LiteRTRunnerPlugin implements RunnerPlugin {

    private static final Logger LOG = Logger.getLogger(LiteRTRunnerPlugin.class);

    public static final String ID = "litert-runner";

    private RunnerConfig config;
    private volatile boolean initialized = false;
    private volatile boolean healthy = false;

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "LiteRT Runner";
    }

    @Override
    public String version() {
        return "2.0.0";
    }

    @Override
    public String description() {
        return "LiteRT model support with CPU, GPU, NNAPI, and LiteRT-LM execution paths";
    }

    @Override
    public String format() {
        return "litert";
    }

    @Override
    public RunnerValidationResult validate() {
        LiteRTRuntimeDiagnostics.Snapshot diagnostics = LiteRTRuntimeDiagnostics.snapshot();
        if (!diagnostics.providerAvailable()) {
            return RunnerValidationResult.invalid("LiteRT provider classes are not available");
        }

        RunnerValidationResult.Builder builder = RunnerValidationResult.builder().valid(true);
        if (!diagnostics.hasExecutionRuntime()) {
            builder.warning("No LiteRT native library or official LiteRT-LM bridge detected yet; "
                    + "inference will require one at execution time");
        }
        if (diagnostics.gpuDelegateAvailable()) {
            builder.warning("GPU delegate available");
        }
        if (diagnostics.nnapiAvailable()) {
            builder.warning("NNAPI delegate available");
        }
        return builder.build();
    }

    @Override
    public void initialize(RunnerContext context) throws RunnerException {
        if (initialized) {
            throw new RunnerInitializationException("litert", "Already initialized");
        }

        try {
            RunnerContext effectiveContext = context != null ? context : RunnerContext.empty();
            this.config = effectiveContext.getConfig();
            LiteRTRuntimeDiagnostics.Snapshot diagnostics = LiteRTRuntimeDiagnostics.snapshot();
            if (!diagnostics.providerAvailable()) {
                throw new RunnerInitializationException("litert", "LiteRT provider classes are not available");
            }
            initialized = true;
            healthy = true;
            LOG.infof("LiteRT runner initialized; execution runtime detected=%s",
                    diagnostics.hasExecutionRuntime());
        } catch (Exception e) {
            initialized = false;
            healthy = false;
            throw new RunnerInitializationException("litert", "Failed to initialize LiteRT: " + e.getMessage(), e);
        }
    }

    @Override
    public boolean isAvailable() {
        return LiteRTRuntimeDiagnostics.snapshot().providerAvailable();
    }

    @Override
    public RunnerHealth health() {
        if (!initialized)
            return RunnerHealth.unhealthy("Not initialized");
        if (!healthy)
            return RunnerHealth.unhealthy("Runner unhealthy");

        Map<String, Object> details = new LinkedHashMap<>();
        details.put("initialized", initialized);
        details.putAll(LiteRTRuntimeDiagnostics.snapshot().asMap());
        return RunnerHealth.healthy(details);
    }

    @Override
    public Set<String> supportedFormats() {
        return LiteRTRuntimeDiagnostics.SUPPORTED_FORMATS;
    }

    @Override
    public Set<String> supportedArchitectures() {
        return Set.of("mobilenet", "efficientnet", "bert", "whisper", "yolo");
    }

    @Override
    public Set<RequestType> supportedRequestTypes() {
        return Set.of(RequestType.INFER, RequestType.CLASSIFY);
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request, RunnerContext context) throws RunnerException {
        if (!initialized) {
            throw new RunnerExecutionException("litert", request.getType().name(), "LiteRT not initialized");
        }
        if (!healthy) {
            throw new RunnerExecutionException("litert", request.getType().name(), "LiteRT unhealthy");
        }

        try {
            RunnerResult<T> result = switch (request.getType()) {
                case INFER -> executeInference(request, context);
                case CLASSIFY -> executeClassification(request, context);
                default -> throw new UnknownRequestException(request.getType());
            };
            return result;
        } catch (RunnerException e) {
            throw e;
        } catch (Exception e) {
            throw new RunnerExecutionException("litert", request.getType().name(),
                    "Execution failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Map<String, Object> metadata() {
        LiteRTRuntimeDiagnostics.Snapshot diagnostics = LiteRTRuntimeDiagnostics.snapshot();
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("format", "litert");
        metadata.put("version", version());
        metadata.put("backend", "LiteRT");
        metadata.put("quantization_support", true);
        metadata.put("runner_initialized", initialized);
        metadata.put("runner_healthy", healthy);
        metadata.put("provider", LiteRTEngine.class.getName());
        metadata.putAll(diagnostics.asMap());
        return metadata;
    }

    @Override
    public void shutdown() {
        if (!initialized)
            return;
        try {
            initialized = false;
            healthy = false;
            LOG.info("LiteRT runner shutdown complete");
        } catch (Exception e) {
            LOG.error("Error shutting down LiteRT runner", e);
        }
    }

    @SuppressWarnings("unchecked")
    private <T> RunnerResult<T> executeInference(RunnerRequest request, RunnerContext context) {
        RunnerContext effectiveContext = context != null ? context : RunnerContext.empty();
        InferenceRequest inferenceRequest;
        try {
            inferenceRequest = resolveInferenceRequest(request, effectiveContext);
        } catch (IllegalArgumentException e) {
            return failed(e.getMessage());
        }

        Optional<String> modelPath = firstString(request, effectiveContext, "model_path", "modelPath", "path");
        if (modelPath.isPresent() && !Files.exists(Path.of(modelPath.get()))) {
            return failed("LiteRT model_path does not exist: " + modelPath.get());
        }

        Map<String, Object> parameters = new LinkedHashMap<>(inferenceRequest.getParameters());
        modelPath.ifPresent(path -> parameters.putIfAbsent("model_path", path));

        Duration timeout = inferenceRequest.getTimeout()
                .or(() -> durationParam(request, effectiveContext, "timeout", "timeout_seconds"))
                .orElse(defaultTimeout(effectiveContext));

        InferenceRequest providerRequest = toProviderRequest(inferenceRequest, parameters, timeout, request);
        LiteRTEngine provider = new LiteRTEngine(providerConfig(request, effectiveContext, timeout));
        try {
            provider.initialize();
            InferenceResponse response = provider.infer(providerRequest).await().atMost(timeout);
            Map<String, Object> resultMetadata = new LinkedHashMap<>(metadata());
            resultMetadata.put("request_id", response.getRequestId());
            resultMetadata.put("model", response.getModel());
            resultMetadata.put("duration_ms", response.getDurationMs());
            return (RunnerResult<T>) RunnerResult.success(response, resultMetadata);
        } catch (Exception e) {
            return failed("LiteRT inference failed: " + rootMessage(e));
        } finally {
            provider.shutdown();
        }
    }

    @SuppressWarnings("unchecked")
    private <T> RunnerResult<T> executeClassification(RunnerRequest request, RunnerContext context) {
        return (RunnerResult<T>) failed("LiteRT classification requires a task-specific model head and is not "
                + "implemented by this runner facade yet");
    }

    private InferenceRequest resolveInferenceRequest(RunnerRequest request, RunnerContext context) {
        Optional<InferenceRequest> supplied = request.getInferenceRequest();
        if (supplied.isPresent()) {
            return supplied.get();
        }

        Optional<String> model = firstString(request, context, "model", "model_id", "modelId")
                .or(() -> firstString(request, context, "model_path", "modelPath", "path"));
        if (model.isEmpty()) {
            throw new IllegalArgumentException("LiteRT inference requires a model id or model_path");
        }

        Optional<String> prompt = firstString(request, context, "prompt", "input", "text");
        if (prompt.isEmpty()) {
            throw new IllegalArgumentException("LiteRT inference requires an InferenceRequest or prompt text");
        }

        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.putAll(context.getParameters());
        parameters.putAll(request.parameters() != null ? request.parameters() : Map.of());
        return InferenceRequest.builder()
                .model(model.get())
                .message(Message.user(prompt.get()))
                .parameters(parameters)
                .streaming(false)
                .build();
    }

    private InferenceRequest toProviderRequest(
            InferenceRequest inferenceRequest,
            Map<String, Object> parameters,
            Duration timeout,
            RunnerRequest runnerRequest) {
        var builder = inferenceRequest.toBuilder()
                .parameters(parameters)
                .timeout(timeout);

        if (runnerRequest.metadata() != null) {
            builder.metadata(runnerRequest.metadata());
        }
        return builder.build();
    }

    private LiteRTProviderConfig providerConfig(RunnerRequest request, RunnerContext context, Duration timeout) {
        RunnerConfig runnerConfig = context.getConfig() != null ? context.getConfig()
                : (config != null ? config : RunnerConfig.defaultConfig());
        String modelBasePath = firstString(request, context, "model_base_path", "modelBasePath")
                .orElse(Path.of(System.getProperty("user.home"), ".gollek", "models", "litert").toString());
        int threads = intParam(request, context, "threads")
                .orElse(Math.max(1, runnerConfig.threads()));
        boolean gpuEnabled = boolParam(request, context, "gpu_enabled", "use_gpu", "gpu")
                .orElse(false);
        boolean autoMetalEnabled = boolParam(request, context, "auto_metal", "gpu_auto_metal")
                .orElse(true);
        boolean npuEnabled = boolParam(request, context, "npu_enabled", "use_npu", "npu")
                .orElse(false);
        String gpuBackend = firstString(request, context, "gpu_backend", "gpuBackend").orElse("auto");
        String npuType = firstString(request, context, "npu_type", "npuType").orElse("auto");

        return new LiteRTProviderConfig() {
            @Override
            public boolean enabled() {
                return true;
            }

            @Override
            public String modelBasePath() {
                return modelBasePath;
            }

            @Override
            public int threads() {
                return threads;
            }

            @Override
            public boolean gpuEnabled() {
                return gpuEnabled;
            }

            @Override
            public boolean autoMetalEnabled() {
                return autoMetalEnabled;
            }

            @Override
            public boolean npuEnabled() {
                return npuEnabled;
            }

            @Override
            public String gpuBackend() {
                return gpuBackend;
            }

            @Override
            public String npuType() {
                return npuType;
            }

            @Override
            public Duration defaultTimeout() {
                return timeout;
            }

            @Override
            public SessionConfig session() {
                return new SessionConfig() {
                    @Override
                    public int maxPerTenant() {
                        return 2;
                    }

                    @Override
                    public int idleTimeoutSeconds() {
                        return 300;
                    }

                    @Override
                    public int maxTotal() {
                        return 8;
                    }
                };
            }
        };
    }

    private Optional<String> firstString(RunnerRequest request, RunnerContext context, String... keys) {
        for (String key : keys) {
            Optional<String> value = valueAsString(request.parameters() != null ? request.parameters().get(key) : null)
                    .or(() -> valueAsString(context.getParameters().get(key)));
            if (value.isPresent()) {
                return value;
            }
        }
        return Optional.empty();
    }

    private Optional<Integer> intParam(RunnerRequest request, RunnerContext context, String... keys) {
        for (String key : keys) {
            Optional<Integer> value = valueAsInteger(request.parameters() != null ? request.parameters().get(key) : null)
                    .or(() -> valueAsInteger(context.getParameters().get(key)));
            if (value.isPresent()) {
                return value;
            }
        }
        return Optional.empty();
    }

    private Optional<Boolean> boolParam(RunnerRequest request, RunnerContext context, String... keys) {
        for (String key : keys) {
            Optional<Boolean> value = valueAsBoolean(request.parameters() != null ? request.parameters().get(key) : null)
                    .or(() -> valueAsBoolean(context.getParameters().get(key)));
            if (value.isPresent()) {
                return value;
            }
        }
        return Optional.empty();
    }

    private Optional<Duration> durationParam(RunnerRequest request, RunnerContext context, String... keys) {
        for (String key : keys) {
            Optional<Duration> value = valueAsDuration(request.parameters() != null ? request.parameters().get(key) : null)
                    .or(() -> valueAsDuration(context.getParameters().get(key)));
            if (value.isPresent()) {
                return value;
            }
        }
        return Optional.empty();
    }

    private Duration defaultTimeout(RunnerContext context) {
        Object value = context.getParameters().get("timeout");
        return valueAsDuration(value).orElse(Duration.ofSeconds(30));
    }

    private Optional<String> valueAsString(Object value) {
        if (value == null) {
            return Optional.empty();
        }
        String text = value.toString().trim();
        return text.isEmpty() ? Optional.empty() : Optional.of(text);
    }

    private Optional<Integer> valueAsInteger(Object value) {
        if (value instanceof Number number) {
            return Optional.of(Math.max(1, number.intValue()));
        }
        return valueAsString(value).flatMap(text -> {
            try {
                return Optional.of(Math.max(1, Integer.parseInt(text)));
            } catch (NumberFormatException e) {
                return Optional.empty();
            }
        });
    }

    private Optional<Boolean> valueAsBoolean(Object value) {
        if (value instanceof Boolean bool) {
            return Optional.of(bool);
        }
        return valueAsString(value).map(Boolean::parseBoolean);
    }

    private Optional<Duration> valueAsDuration(Object value) {
        if (value instanceof Duration duration) {
            return Optional.of(duration);
        }
        if (value instanceof Number number) {
            return Optional.of(Duration.ofSeconds(Math.max(1, number.longValue())));
        }
        return valueAsString(value).flatMap(text -> {
            try {
                return Optional.of(Duration.parse(text));
            } catch (Exception ignored) {
                try {
                    return Optional.of(Duration.ofSeconds(Math.max(1, Long.parseLong(text))));
                } catch (NumberFormatException e) {
                    return Optional.empty();
                }
            }
        });
    }

    private <T> RunnerResult<T> failed(String message) {
        return RunnerResult.<T>builder()
                .status(RunnerResult.Status.FAILED)
                .errorMessage(message)
                .metadata(metadata())
                .build();
    }

    private String rootMessage(Throwable throwable) {
        Throwable cursor = throwable;
        while (cursor != null && cursor.getCause() != null && cursor.getCause() != cursor) {
            cursor = cursor.getCause();
        }
        return cursor == null || cursor.getMessage() == null ? throwable.getClass().getSimpleName() : cursor.getMessage();
    }
}
