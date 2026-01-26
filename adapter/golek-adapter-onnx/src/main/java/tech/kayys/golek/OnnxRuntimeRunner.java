
/**
 * ONNX Runtime adapter with execution provider selection
 * Supports CPU, CUDA, DirectML, TensorRT, and OpenVINO
 */
@ApplicationScoped
@IfBuildProperty(
    name = "inference.adapter.onnx.enabled",
    stringValue = "true"
)
public class OnnxRuntimeRunner implements ModelRunner {
    
    private static final Logger log = Logger.getLogger(OnnxRuntimeRunner.class);
    
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
        TenantContext tenantContext
    ) throws ModelLoadException {
        try {
            this.manifest = manifest;
            
            // Download ONNX model
            Path modelPath = repository.downloadArtifact(
                manifest,
                ModelFormat.ONNX
            );
            
            // Create ONNX environment (shared)
            this.environment = OrtEnvironment.getEnvironment();
            
            // Select best execution provider
            this.providerSelector = new ExecutionProviderSelector();
            String provider = selectExecutionProvider(config);
            
            // Create session options
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            configureSessionOptions(options, provider, config);
            
            // Create session
            this.session = environment.createSession(
                modelPath.toString(),
                options
            );
            
            this.initialized = true;
            
            log.infof("Initialized ONNX model %s with provider %s", 
                manifest.modelId(), provider);
                
        } catch (OrtException e) {
            throw new ModelLoadException(
                "Failed to initialize ONNX Runtime", e
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
            inputs.values().forEach(OnnxTensor::close);
            result.close();
            
            return InferenceResponse.builder()
                .requestId(request.requestId())
                .modelId(manifest.modelId())
                .outputs(outputs)
                .metadata("runner", "onnx")
                .build();
                
        } catch (OrtException e) {
            throw new InferenceException("ONNX inference failed", e);
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
            HealthStatus.up() : 
            HealthStatus.down("Not initialized");
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
            devices.add(DeviceType.CUDA);
        }
        if (providerSelector.isProviderAvailable("TensorrtExecutionProvider")) {
            devices.add(DeviceType.CUDA);
        }
        
        return new RunnerMetadata(
            "onnx",
            OrtEnvironment.getVersion(),
            List.of(ModelFormat.ONNX),
            devices,
            ExecutionMode.SYNCHRONOUS,
            Map.of(
                "execution_providers", 
                    providerSelector.getAvailableProviders()
            )
        );
    }
    
    @Override
    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                log.errorf(e, "Error closing ONNX session");
            }
        }
        initialized = false;
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
            intraOpThreads.orElse(
                Runtime.getRuntime().availableProcessors()
            )
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
    }
    
    private Map<String, OnnxTensor> convertInputs(
        InferenceRequest request
    ) throws OrtException {
        // Convert request inputs to ONNX tensors
        // Implementation depends on model input schema
        Map<String, OnnxTensor> tensors = new HashMap<>();
        
        // Example for text input
        if (request.hasInput("input_ids")) {
            long[] inputIds = request.getInput("input_ids", long[].class);
            OnnxTensor tensor = OnnxTensor.createTensor(
                environment,
                LongBuffer.wrap(inputIds),
                new long[]{1, inputIds.length}
            );
            tensors.put("input_ids", tensor);
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
                // Convert tensor to appropriate Java type
                outputs.put(name, tensor.getValue());
            }
        }
        
        return outputs;
    }
    
    private long estimateMemoryUsage() {
        // Estimate based on model size and session state
        try {
            return session.getMetadata().getProducerName() != null ? 
                512 : 0; // Placeholder
        } catch (Exception e) {
            return 0;
        }
    }
}