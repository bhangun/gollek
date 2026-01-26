package tech.kayys.golek.provider.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tech.kayys.golek.provider.spi.ModelRunner;
import tech.kayys.golek.provider.spi.exception.ModelLoadException;
import tech.kayys.golek.provider.spi.exception.InferenceException;
import tech.kayys.golek.provider.spi.model.ModelManifest;
import tech.kayys.golek.provider.spi.model.ModelFormat;
import tech.kayys.golek.provider.spi.model.RequestContext;
import tech.kayys.golek.provider.spi.model.InferenceRequest;
import tech.kayys.golek.provider.spi.model.InferenceResponse;
import tech.kayys.golek.provider.spi.health.HealthStatus;
import tech.kayys.golek.provider.spi.metrics.ResourceMetrics;
import tech.kayys.golek.provider.spi.model.RunnerMetadata;
import tech.kayys.golek.provider.spi.model.DeviceType;
import tech.kayys.golek.provider.spi.model.ExecutionMode;
import tech.kayys.golek.provider.spi.repository.ModelRepository;

import java.nio.LongBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.DoubleBuffer;
import java.nio.file.Path;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

/**
 * ONNX Runtime adapter with execution provider selection
 * Supports CPU, CUDA, DirectML, TensorRT, and OpenVINO
 */
@ApplicationScoped
public class OnnxRuntimeRunner implements ModelRunner {

    private static final Logger log = LoggerFactory.getLogger(OnnxRuntimeRunner.class);

    private volatile boolean initialized = false;
    private ModelManifest manifest;
    private OrtSession session;
    private OrtEnvironment environment;
    private ExecutionProviderSelector providerSelector;

    @Inject
    ModelRepository repository;

    @ConfigProperty(name = "inference.adapter.onnx.execution-provider")
    Optional<String> preferredProvider;

    @ConfigProperty(name = "inference.adapter.onnx.inter-op-threads")
    Optional<Integer> interOpThreads;

    @ConfigProperty(name = "inference.adapter.onnx.intra-op-threads")
    Optional<Integer> intraOpThreads;

    @Override
    public void initialize(
        ModelManifest manifest,
        Map<String, Object> config,
        RequestContext context
    ) throws ModelLoadException {
        try {
            // Validate inputs
            if (manifest == null) {
                throw new ModelLoadException("Model manifest cannot be null");
            }

            if (!manifest.supportedFormats().contains(ModelFormat.ONNX)) {
                throw new ModelLoadException(
                    "Model " + manifest.modelId() + " does not support ONNX format"
                );
            }

            this.manifest = manifest;

            // Download ONNX model
            Path modelPath = repository.downloadArtifact(
                manifest,
                ModelFormat.ONNX
            );

            if (modelPath == null || !modelPath.toFile().exists()) {
                throw new ModelLoadException("ONNX model file not found: " + modelPath);
            }

            // Create ONNX environment (shared)
            this.environment = OrtEnvironment.getEnvironment();

            // Select best execution provider
            this.providerSelector = new ExecutionProviderSelector();
            String provider = selectExecutionProvider(config);

            // Validate selected provider
            if (!providerSelector.isValidProvider(provider)) {
                log.warn("Selected provider {} is not available, falling back to CPU", provider);
                provider = "CPUExecutionProvider";
            }

            // Create session options
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            configureSessionOptions(options, provider, config);

            // Create session
            this.session = environment.createSession(
                modelPath.toString(),
                options
            );

            this.initialized = true;

            log.info("Initialized ONNX model {} with provider {}",
                manifest.modelId(), provider);

        } catch (OrtException e) {
            throw new ModelLoadException(
                "Failed to initialize ONNX Runtime: " + e.getMessage(), e
            );
        } catch (Exception e) {
            throw new ModelLoadException(
                "Unexpected error during ONNX initialization: " + e.getMessage(), e
            );
        }
    }

    @Override
    public InferenceResponse infer(
        InferenceRequest request,
        RequestContext context
    ) throws InferenceException {

        if (!initialized) {
            throw new IllegalStateException("Runner not initialized");
        }

        try {
            // Convert inputs to ONNX tensors
            Map<String, OnnxTensor> inputs = convertInputs(request);

            // Run inference
            OrtSession.Result result = session.run(inputs);

            // Convert outputs
            Map<String, Object> outputs = convertOutputs(result);

            // Cleanup
            inputs.values().forEach(tensor -> {
                try {
                    tensor.close();
                } catch (Exception e) {
                    log.warn("Error closing input tensor", e);
                }
            });
            result.close();

            return InferenceResponse.builder()
                .requestId(request.requestId())
                .modelId(manifest.modelId())
                .outputs(outputs)
                .metadata("runner", "onnx")
                .build();

        } catch (OrtException e) {
            throw new InferenceException("ONNX inference failed: " + e.getMessage(), e);
        } catch (Exception e) {
            throw new InferenceException("Unexpected error during inference: " + e.getMessage(), e);
        }
    }

    @Override
    public CompletionStage<InferenceResponse> inferAsync(
        InferenceRequest request,
        RequestContext context
    ) {
        return CompletableFuture.supplyAsync(
            () -> infer(request, context)
        );
    }

    @Override
    public HealthStatus health() {
        return initialized ?
            HealthStatus.builder().up().withDetail("status", "running").build() :
            HealthStatus.builder().down().withDetail("status", "not initialized").build();
    }

    @Override
    public ResourceMetrics getMetrics() {
        // ONNX Runtime doesn't expose direct memory metrics
        // Use JVM metrics or system monitoring
        return ResourceMetrics.builder()
            .memoryUsedMb(estimateMemoryUsage())
            .build();
    }

    @Override
    public RunnerMetadata metadata() {
        List<DeviceType> devices = new ArrayList<>();
        devices.add(DeviceType.CPU);

        if (providerSelector.isProviderAvailable("CUDAExecutionProvider")) {
            devices.add(DeviceType.GPU);
        }
        if (providerSelector.isProviderAvailable("TensorrtExecutionProvider")) {
            devices.add(DeviceType.GPU);
        }

        return RunnerMetadata.builder()
            .name("onnx")
            .version(OrtEnvironment.getVersion())
            .supportedFormats(List.of(ModelFormat.ONNX))
            .supportedDevices(devices)
            .executionMode(ExecutionMode.SYNCHRONOUS)
            .capabilities(Map.of(
                "execution_providers", providerSelector.getAvailableProviders()
            ))
            .build();
    }

    @Override
    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                log.error("Error closing ONNX session", e);
            }
        }
        initialized = false;
        log.info("ONNX Runtime runner closed");
    }

    private String selectExecutionProvider(Map<String, Object> config) {
        // Priority: config > hardware detection > default
        if (config.containsKey("execution_provider")) {
            return (String) config.get("execution_provider");
        }

        if (preferredProvider.isPresent()) {
            String provider = preferredProvider.get();
            if (providerSelector.isProviderAvailable(provider)) {
                return provider;
            }
        }

        // Auto-detect best available
        return providerSelector.selectBestProvider();
    }

    private void configureSessionOptions(
        OrtSession.SessionOptions options,
        String provider,
        Map<String, Object> config
    ) throws OrtException {

        // Set thread counts
        options.setInterOpNumThreads(
            interOpThreads.orElse(1)
        );
        options.setIntraOpNumThreads(
            intraOpThreads.orElse(0) // 0 means use all available processors
        );

        // Set execution provider
        switch (provider) {
            case "CUDAExecutionProvider":
                options.addCUDA(0); // GPU device 0
                break;
            case "TensorrtExecutionProvider":
                options.addTensorrt(0);
                break;
            case "OpenVINOExecutionProvider":
                options.addOpenVINO("");
                break;
            case "DirectMLExecutionProvider":
                options.addDirectML(0);
                break;
            default:
                // CPUExecutionProvider is always available
                break;
        }

        // Optimization level
        options.setOptimizationLevel(
            OrtSession.SessionOptions.OptLevel.ALL_OPT
        );

        // Memory optimization
        options.setMemoryPatternOptimization(true);
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
    }

    private Map<String, OnnxTensor> convertInputs(
        InferenceRequest request
    ) throws OrtException {
        // Convert request inputs to ONNX tensors
        // Implementation depends on model input schema
        Map<String, OnnxTensor> tensors = new HashMap<>();

        // Handle different input types based on the request
        for (Map.Entry<String, Object> entry : request.inputs().entrySet()) {
            String inputName = entry.getKey();
            Object inputValue = entry.getValue();

            if (inputValue instanceof long[]) {
                long[] longArray = (long[]) inputValue;
                OnnxTensor tensor = OnnxTensor.createTensor(
                    environment,
                    LongBuffer.wrap(longArray),
                    new long[]{1, longArray.length}
                );
                tensors.put(inputName, tensor);
            } else if (inputValue instanceof float[]) {
                float[] floatArray = (float[]) inputValue;
                OnnxTensor tensor = OnnxTensor.createTensor(
                    environment,
                    FloatBuffer.wrap(floatArray),
                    new long[]{1, floatArray.length}
                );
                tensors.put(inputName, tensor);
            } else if (inputValue instanceof int[]) {
                int[] intArray = (int[]) inputValue;
                OnnxTensor tensor = OnnxTensor.createTensor(
                    environment,
                    IntBuffer.wrap(intArray),
                    new long[]{1, intArray.length}
                );
                tensors.put(inputName, tensor);
            } else if (inputValue instanceof double[]) {
                double[] doubleArray = (double[]) inputValue;
                OnnxTensor tensor = OnnxTensor.createTensor(
                    environment,
                    DoubleBuffer.wrap(doubleArray),
                    new long[]{1, doubleArray.length}
                );
                tensors.put(inputName, tensor);
            } else if (inputValue instanceof List) {
                // Handle List inputs by converting to appropriate array type
                List<?> list = (List<?>) inputValue;
                if (!list.isEmpty()) {
                    Object firstElement = list.get(0);
                    if (firstElement instanceof Number) {
                        // Convert to float array by default for numeric lists
                        float[] floatArray = new float[list.size()];
                        for (int i = 0; i < list.size(); i++) {
                            floatArray[i] = ((Number) list.get(i)).floatValue();
                        }
                        OnnxTensor tensor = OnnxTensor.createTensor(
                            environment,
                            FloatBuffer.wrap(floatArray),
                            new long[]{1, floatArray.length}
                        );
                        tensors.put(inputName, tensor);
                    } else if (firstElement instanceof String) {
                        // Handle string inputs differently - this would require a different approach
                        log.warn("String list inputs not supported for ONNX: {}", inputName);
                    }
                }
            } else if (inputValue instanceof Number) {
                // Handle single number by wrapping in array
                float[] singleValue = new float[]{((Number) inputValue).floatValue()};
                OnnxTensor tensor = OnnxTensor.createTensor(
                    environment,
                    FloatBuffer.wrap(singleValue),
                    new long[]{1, 1}
                );
                tensors.put(inputName, tensor);
            } else {
                // For other types, try to convert to float array if possible
                log.warn("Unsupported input type for {}: {}", inputName, inputValue.getClass());
            }
        }

        return tensors;
    }

    private Map<String, Object> convertOutputs(
        OrtSession.Result result
    ) throws OrtException {
        Map<String, Object> outputs = new HashMap<>();

        for (Map.Entry<String, OnnxValue> entry : result) {
            String name = entry.getKey();
            OnnxValue value = entry.getValue();

            if (value instanceof OnnxTensor tensor) {
                // Convert tensor to appropriate Java type based on element type
                Object tensorValue = tensor.getValue();
                outputs.put(name, tensorValue);
            } else {
                // Handle other value types if needed
                outputs.put(name, value.toString());
            }
        }

        return outputs;
    }

    private long estimateMemoryUsage() {
        // Estimate based on model size and session state
        try {
            // This is a very rough estimation - in practice, you'd want to use
            // more sophisticated memory tracking
            return 512; // Placeholder value in MB
        } catch (Exception e) {
            return 0;
        }
    }
}