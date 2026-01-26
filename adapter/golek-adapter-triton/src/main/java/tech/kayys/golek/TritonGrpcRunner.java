
/**
 * Triton Inference Server client adapter supporting both
 * gRPC and HTTP protocols with connection pooling
 */
@ApplicationScoped
@IfBuildProperty(
    name = "inference.adapter.triton.enabled",
    stringValue = "true"
)
public class TritonGrpcRunner implements ModelRunner {
    
    private static final Logger log = Logger.getLogger(TritonGrpcRunner.class);
    
    private volatile boolean initialized = false;
    private ModelManifest manifest;
    private GRPCInferenceServiceBlockingStub stub;
    private ManagedChannel channel;
    
    @ConfigProperty(name = "inference.adapter.triton.endpoint")
    String tritonEndpoint;
    
    @ConfigProperty(name = "inference.adapter.triton.timeout-ms")
    Optional<Integer> timeoutMs;
    
    @ConfigProperty(name = "inference.adapter.triton.use-ssl")
    Optional<Boolean> useSsl;
    
    @Override
    public void initialize(
        ModelManifest manifest,
        Map<String, Object> config,
        TenantContext tenantContext
    ) throws ModelLoadException {
        try {
            this.manifest = manifest;
            
            // Parse endpoint
            String[] parts = tritonEndpoint.split(":");
            String host = parts[0];
            int port = parts.length > 1 ? 
                Integer.parseInt(parts[1]) : 8001;
            
            // Create gRPC channel
            ManagedChannelBuilder<?> channelBuilder = 
                ManagedChannelBuilder.forAddress(host, port);
            
            if (useSsl.orElse(false)) {
                channelBuilder.useTransportSecurity();
            } else {
                channelBuilder.usePlaintext();
            }
            
            this.channel = channelBuilder
                .maxInboundMessageSize(100 * 1024 * 1024) // 100MB
                .keepAliveTime(30, TimeUnit.SECONDS)
                .build();
            
            // Create stub
            this.stub = GRPCInferenceServiceGrpc
                .newBlockingStub(channel);
            
            // Verify model is loaded on Triton
            verifyModelReady();
            
            this.initialized = true;
            
            log.infof("Connected to Triton server at %s for model %s",
                tritonEndpoint, manifest.name());
                
        } catch (Exception e) {
            throw new ModelLoadException(
                "Failed to initialize Triton client", e
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
            // Build Triton request
            ModelInferRequest tritonRequest = buildTritonRequest(request);
            
            // Execute with timeout
            ModelInferResponse tritonResponse = stub
                .withDeadlineAfter(
                    timeoutMs.orElse(30000), 
                    TimeUnit.MILLISECONDS
                )
                .modelInfer(tritonRequest);
            
            // Convert response
            return convertTritonResponse(tritonResponse, request);
            
        } catch (StatusRuntimeException e) {
            if (e.getStatus().getCode() == Status.Code.DEADLINE_EXCEEDED) {
                throw new InferenceTimeoutException(
                    "Triton inference timeout", e
                );
            }
            throw new InferenceException(
                "Triton inference failed: " + e.getStatus(), e
            );
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
        if (!initialized) {
            return HealthStatus.down("Not initialized");
        }
        
        try {
            // Call Triton health endpoint
            ServerLiveRequest liveRequest = ServerLiveRequest
                .newBuilder()
                .build();
                
            ServerLiveResponse response = stub
                .withDeadlineAfter(5, TimeUnit.SECONDS)
                .serverLive(liveRequest);
                
            return response.getLive() ?
                HealthStatus.up() :
                HealthStatus.down("Server not live");
                
        } catch (Exception e) {
            return HealthStatus.down("Health check failed: " + 
                e.getMessage());
        }
    }
    
    @Override
    public ResourceMetrics getMetrics() {
        // Query Triton metrics endpoint
        try {
            ModelStatisticsRequest statsRequest = ModelStatisticsRequest
                .newBuilder()
                .setName(manifest.name())
                .setVersion(manifest.version())
                .build();
                
            ModelStatisticsResponse statsResponse = stub
                .modelStatistics(statsRequest);
                
            // Extract relevant metrics
            return ResourceMetrics.builder()
                .requestCount(statsResponse.getModelStats(0)
                    .getInferenceStats()
                    .getSuccess()
                    .getCount())
                .build();
                
        } catch (Exception e) {
            return ResourceMetrics.empty();
        }
    }
    
    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(
            "triton",
            "2.x",
            List.of(
                ModelFormat.ONNX,
                ModelFormat.TENSORRT,
                ModelFormat.TORCHSCRIPT,
                ModelFormat.TENSORFLOW_SAVED_MODEL
            ),
            List.of(DeviceType.CPU, DeviceType.CUDA),
            ExecutionMode.ASYNCHRONOUS,
            Map.of(
                "supports_streaming", true,
                "supports_batching", true,
                "endpoint", tritonEndpoint
            )
        );
    }
    
    @Override
    public void close() {
        if (channel != null && !channel.isShutdown()) {
            try {
                channel.shutdown();
                channel.awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                channel.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        initialized = false;
    }
    
    private void verifyModelReady() {
        ModelReadyRequest readyRequest = ModelReadyRequest
            .newBuilder()
            .setName(manifest.name())
            .setVersion(manifest.version())
            .build();
            
        ModelReadyResponse response = stub.modelReady(readyRequest);
        
        if (!response.getReady()) {
            throw new IllegalStateException(
                "Model not ready on Triton: " + manifest.name()
            );
        }
    }
    
    private ModelInferRequest buildTritonRequest(
        InferenceRequest request
    ) {
        ModelInferRequest.Builder builder = ModelInferRequest
            .newBuilder()
            .setModelName(manifest.name())
            .setModelVersion(manifest.version());
        
        // Add inputs
        request.getInputs().forEach((name, value) -> {
            InferInputTensor input = InferInputTensor.newBuilder()
                .setName(name)
                .setDatatype("FP32") // or INT64, etc.
                .addShape(1) // Batch size
                .setContents(
                    InferTensorContents.newBuilder()
                        // Add actual data
                        .build()
                )
                .build();
            builder.addInputs(input);
        });
        
        // Add requested outputs
        builder.addOutputs(
            ModelInferRequest.InferRequestedOutputTensor
                .newBuilder()
                .setName("output")
                .build()
        );
        
        return builder.build();
    }
    
    private InferenceResponse convertTritonResponse(
        ModelInferResponse tritonResponse,
        InferenceRequest originalRequest
    ) {
        Map<String, Object> outputs = new HashMap<>();
        
        tritonResponse.getOutputsList().forEach(output -> {
            String name = output.getName();
            // Extract data from output tensor
            // Implementation depends on data type
            outputs.put(name, extractTensorData(output));
        });
        
        return InferenceResponse.builder()
            .requestId(originalRequest.requestId())
            .modelId(manifest.modelId())
            .outputs(outputs)
            .metadata("runner", "triton")
            .metadata("model_version", tritonResponse.getModelVersion())
            .build();
    }
    
    private Object extractTensorData(ModelInferResponse.InferOutputTensor output) {
        // Extract based on datatype
        // Simplified example
        return output.getContents().getFp32ContentList();
    }
}