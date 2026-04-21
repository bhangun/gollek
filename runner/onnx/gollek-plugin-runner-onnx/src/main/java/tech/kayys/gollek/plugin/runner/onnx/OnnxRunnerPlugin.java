/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.plugin.runner.onnx;

import org.jboss.logging.Logger;
import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;
import tech.kayys.gollek.plugin.runner.*;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Enhanced ONNX Runtime runner plugin with comprehensive lifecycle management,
 * validation, and error handling.
 */
public class OnnxRunnerPlugin implements RunnerPlugin {

    private static final Logger LOG = Logger.getLogger(OnnxRunnerPlugin.class);

    public static final String ID = "onnx-runner";

    private RunnerConfig config;
    private String executionProvider;
    private volatile boolean initialized = false;
    private volatile boolean healthy = false;
    private OnnxDevice device;
    
    private final Map<String, OnnxRunnerSession> sessionsByPath = new ConcurrentHashMap<>();
    private final Set<RunnerSession> activeSessions = ConcurrentHashMap.newKeySet();

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "ONNX Runtime Runner";
    }

    @Override
    public String version() {
        return "2.0.0";
    }

    @Override
    public String description() {
        return "ONNX format support using ONNX Runtime with multi-device execution providers";
    }

    @Override
    public String format() {
        return "onnx";
    }

    @Override
    public RunnerValidationResult validate() {
        LOG.info("Validating ONNX runner...");
        
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        if (!isOnnxRuntimeAvailable()) {
            errors.add("ONNX Runtime not available");
            return RunnerValidationResult.invalid(errors);
        }

        if ("CUDAExecutionProvider".equals(executionProvider)) {
            if (!isCudaAvailable()) {
                errors.add("CUDA not available for CUDAExecutionProvider");
            }
        } else if ("CoreMLExecutionProvider".equals(executionProvider)) {
            if (!isCoreMLAvailable()) {
                errors.add("CoreML not available for CoreMLExecutionProvider");
            }
        }

        return RunnerValidationResult.builder()
            .valid(errors.isEmpty())
            .errors(errors)
            .warnings(warnings)
            .build();
    }

    @Override
    public void initialize(RunnerContext context) throws RunnerException {
        if (initialized) return;

        LOG.info("Initializing ONNX runner...");

        try {
            this.config = context.getConfig();
            this.executionProvider = detectBestExecutionProvider();
            
            String libPath = System.getProperty("gollek.runners.onnx.library-path", "/usr/lib/libonnxruntime.so");
            if (!OnnxRuntimeBinding.initialize(java.nio.file.Path.of(libPath))) {
                OnnxRuntimeBinding.initializeFallback();
            }
            
            this.device = queryDevice();
            initialized = true;
            healthy = true;
            LOG.infof("ONNX runner initialized with %s on %s", executionProvider, device.name());
        } catch (Exception e) {
            initialized = false;
            healthy = false;
            throw new RunnerInitializationException("onnx", "Failed to initialize ONNX Runtime: " + e.getMessage(), e);
        }
    }

    @Override
    public RunnerSession createSession(String modelPath, Map<String, Object> config) {
        return sessionsByPath.computeIfAbsent(modelPath, path -> {
            int intraOp = this.config != null ? this.config.threads() : 4;
            int interOp = this.config != null ? this.config.threadsBatch() : 1;
            
            OnnxRunnerSession session = new OnnxRunnerSession(
                path, config, executionProvider, intraOp, interOp
            );
            activeSessions.add(session);
            return session;
        });
    }

    @Override
    public ModelHandle loadModel(ModelLoadRequest request, RunnerContext context) throws RunnerException {
        RunnerSession session = createSession(request.getModelPath(), request.getMetadata());
        return ModelHandle.of(
            request.getModelPath(),
            request.getFormat(),
            session.getModelInfo().getMetadata()
        );
    }

    @Override
    public boolean isAvailable() {
        return isOnnxRuntimeAvailable();
    }

    @Override
    public RunnerHealth health() {
        if (!initialized) return RunnerHealth.unhealthy("Not initialized");
        
        Map<String, Object> details = new HashMap<>();
        details.put("execution_provider", executionProvider);
        details.put("active_sessions", activeSessions.size());
        
        if (device != null) {
            details.put("device_name", device.name());
            details.put("cuda_available", device.cudaAvailable());
        }
        
        return healthy ? RunnerHealth.healthy(details) : RunnerHealth.unhealthy("Runner unhealthy", details);
    }

    @Override
    public Set<String> supportedFormats() {
        return Set.of(".onnx", ".onnxruntime");
    }

    @Override
    public Set<String> supportedArchitectures() {
        return Set.of("bert", "roberta", "llama", "mistral", "phi");
    }

    @Override
    public Set<RequestType> supportedRequestTypes() {
        return Set.of(RequestType.INFER, RequestType.EMBED);
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request, RunnerContext context) throws RunnerException {
        if (!initialized || !healthy) throw new RunnerExecutionException("onnx", request.getType().name(), "Runner not ready");

        return switch (request.getType()) {
            case INFER -> executeInference(request, context);
            default -> throw new UnknownRequestException(request.getType());
        };
    }

    @SuppressWarnings("unchecked")
    private <T> RunnerResult<T> executeInference(RunnerRequest request, RunnerContext context) throws RunnerException {
        var ir = request.inferenceRequest();
        if (ir == null) throw new RunnerException("Missing inference request");
        
        var session = sessionsByPath.get(ir.getModel());
        if (session == null) session = (OnnxRunnerSession) createSession(ir.getModel(), request.parameters());
        
        try {
            return (RunnerResult<T>) RunnerResult.success(session.infer(ir).await().indefinitely());
        } catch (Exception e) {
            throw new RunnerException("Inference failed", e);
        }
    }

    @Override
    public Map<String, Object> metadata() {
        Map<String, Object> meta = new HashMap<>();
        meta.put("format", "onnx");
        meta.put("version", version());
        meta.put("backend", "ONNX Runtime");
        meta.put("architectures", supportedArchitectures());
        
        if (device != null) {
            meta.put("device_name", device.name());
            meta.put("cuda_available", device.cudaAvailable());
        }
        
        meta.put("execution_provider", executionProvider);
        meta.put("active_sessions", activeSessions.size());
        
        return meta;
    }

    @Override
    public void shutdown() {
        if (!initialized) return;
        activeSessions.forEach(RunnerSession::close);
        activeSessions.clear();
        sessionsByPath.clear();
        initialized = false;
        healthy = false;
        LOG.info("ONNX runner shutdown complete");
    }

    private boolean isOnnxRuntimeAvailable() { return RunnerPlugin.isClassAvailable("ai.onnxruntime.OnnxTensor"); }
    private boolean isCudaAvailable() { return RunnerPlugin.isClassAvailable("org.bytedeco.cuda.cudart.CUDA"); }
    private boolean isCoreMLAvailable() { 
        String os = System.getProperty("os.name").toLowerCase();
        return os.contains("mac") && System.getProperty("os.arch").contains("64");
    }

    private String detectBestExecutionProvider() {
        if (isCudaAvailable()) return "CUDAExecutionProvider";
        if (isCoreMLAvailable()) return "CoreMLExecutionProvider";
        return "CPUExecutionProvider";
    }

    private OnnxDevice queryDevice() {
        return new OnnxDevice(detectBestExecutionProvider(), isCudaAvailable(), false);
    }

    private record OnnxDevice(String name, boolean cudaAvailable, boolean directMLAvailable) {}
}
