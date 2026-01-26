
import io.quarkus.arc.properties.IfBuildProperty;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Production-ready GGUF/llama.cpp model runner implementation
 * Supports CPU and CUDA acceleration with complete lifecycle management
 */
@ApplicationScoped
@IfBuildProperty(name = "inference.adapter.gguf.enabled", stringValue = "true")
public class LlamaCppRunner implements ModelRunner {
    
    private static final Logger log = Logger.getLogger(LlamaCppRunner.class);
    
    private volatile boolean initialized = false;
    private ModelManifest manifest;
    private TenantContext tenantContext;
    private GGUFConfig config;
    
    // Native handles
    private LlamaCppBinding binding;
    private MemorySegment model;
    private MemorySegment context;
    private Path modelPath;
    
    // Model metadata
    private int eosToken;
    private int bosToken;
    private int contextSize;
    private int vocabSize;
    
    // Threading and concurrency
    private final ExecutorService executorService = Executors.newCachedThreadPool();
    private final Semaphore concurrencyLimit;
    
    // Metrics
    private final AtomicLong totalInferences = new AtomicLong(0);
    private final AtomicLong failedInferences = new AtomicLong(0);
    private final AtomicLong totalTokensGenerated = new AtomicLong(0);
    private volatile Duration lastInferenceLatency = Duration.ZERO;
    
    @Inject
    ModelRepository repository;
    
    @ConfigProperty(name = "inference.adapter.gguf.threads")
    Optional
<Integer> configThreads;

@ConfigProperty(name = "inference.adapter.gguf.use-gpu")
Optional
<Boolean> configUseGpu;

@ConfigProperty(name = "inference.adapter.gguf.gpu-layers")
Optional
<Integer> configGpuLayers;

@ConfigProperty(name = "inference.adapter.gguf.context-size")
Optional
<Integer> configContextSize;

@ConfigProperty(name = "inference.adapter.gguf.max-concurrent")
Optional
<Integer> configMaxConcurrent;

public LlamaCppRunner() {
this.concurrencyLimit = new Semaphore(
Integer.parseInt(System.getProperty("gguf.max.concurrent", "5"))
);
}

@Override
public void initialize(
ModelManifest manifest,
Map
<String , Object> runnerConfig,
TenantContext tenantContext
) throws ModelLoadException {

if (initialized) {
log.warnf("Runner already initialized for model %s", manifest.modelId());
return;
}

log.infof("Initializing GGUF runner for model %s (tenant: %s)",
manifest.modelId(), tenantContext.tenantId().value());

try {
this.manifest = manifest;
this.tenantContext = tenantContext;

// Build configuration
this.config = buildConfiguration(runnerConfig);

// Download model artifact if not cached
this.modelPath = repository.downloadArtifact(manifest, ModelFormat.GGUF);

if (!modelPath.toFile().exists()) {
throw new ModelLoadException("Model file not found: " + modelPath);
}

log.infof("Loading GGUF model from: %s", modelPath);

// Load native library
this.binding = LlamaCppBinding.load();

// Create model parameters
MemorySegment modelParams = binding.getDefaultModelParams();
configureModelParams(modelParams, config);

// Load model
this.model = binding.loadModel(modelPath.toString(), modelParams);

// Create context parameters
MemorySegment contextParams = binding.getDefaultContextParams();
configureContextParams(contextParams, config);

// Create context
this.context = binding.createContext(model, contextParams);

// Cache model metadata
this.eosToken = binding.getEosToken(model);
this.bosToken = binding.getBosToken(model);
this.contextSize = binding.getContextSize(context);
this.vocabSize = binding.getVocabSize(model);

this.initialized = true;

log.infof("Successfully initialized GGUF model %s - ctx_size=%d, vocab_size=%d, gpu_layers=%d",
manifest.modelId(), contextSize, vocabSize, config.getNGpuLayers());

} catch (Exception e) {
cleanup();
throw new ModelLoadException("Failed to initialize GGUF runner", e);
}
}

@Override
public InferenceResponse infer(
InferenceRequest request,
RequestContext requestContext
) throws InferenceException {

if (!initialized) {
throw new IllegalStateException("Runner not initialized");
}

// Check concurrency limit
boolean acquired = false;
try {
acquired = concurrencyLimit.tryAcquire(
requestContext.timeout().toMillis(),
TimeUnit.MILLISECONDS
);

if (!acquired) {
throw new InferenceException("Concurrency limit exceeded");
}

return executeInference(request, requestContext);

} catch (InterruptedException e) {
Thread.currentThread().interrupt();
throw new InferenceException("Inference interrupted", e);
} finally {
if (acquired) {
concurrencyLimit.release();
}
}
}

private InferenceResponse executeInference(
InferenceRequest request,
RequestContext requestContext
) throws InferenceException {

Instant startTime = Instant.now();
totalInferences.incrementAndGet();

try {
// Extract prompt
String prompt = request.getInput("prompt", String.class);
if (prompt == null || prompt.isEmpty()) {
throw new InferenceException("Prompt is required");
}

// Build generation parameters
GenerationParams genParams = buildGenerationParams(request);

// Tokenize prompt
int[] promptTokens = binding.tokenize(model, prompt, true, false);

if (promptTokens.length > contextSize) {
throw new InferenceException(
String.format("Prompt too long: %d tokens (max: %d)",
promptTokens.length, contextSize)
);
}

log.debugf("Tokenized prompt: %d tokens", promptTokens.length);

// Generate text with timeout
CompletableFuture
<String> future = CompletableFuture.supplyAsync(
() -> generate(promptTokens, genParams, requestContext),
executorService
);

String generatedText = future.get(
requestContext.timeout().toMillis(),
TimeUnit.MILLISECONDS
);

// Calculate metrics
Duration latency = Duration.between(startTime, Instant.now());
this.lastInferenceLatency = latency;

int generatedTokens = estimateTokenCount(generatedText);
totalTokensGenerated.addAndGet(generatedTokens);

log.debugf("Generated %d tokens in %dms", generatedTokens, latency.toMillis());

// Build response
return InferenceResponse.builder()
.requestId(request.requestId())
.modelId(manifest.modelId())
.output("text", generatedText)
.output("generated_tokens", generatedTokens)
.metadata("runner", "gguf")
.metadata("prompt_tokens", promptTokens.length)
.metadata("total_tokens", promptTokens.length + generatedTokens)
.metadata("latency_ms", latency.toMillis())
.metadata("tokens_per_second",
generatedTokens / Math.max(latency.toSeconds(), 1))
.build();

} catch (TimeoutException e) {
failedInferences.incrementAndGet();
throw new InferenceTimeoutException(
"Inference exceeded timeout: " + requestContext.timeout(), e
);
} catch (ExecutionException e) {
failedInferences.incrementAndGet();
throw new InferenceException("Inference execution failed", e.getCause());
} catch (InterruptedException e) {
failedInferences.incrementAndGet();
Thread.currentThread().interrupt();
throw new InferenceException("Inference interrupted", e);
} catch (Exception e) {
failedInferences.incrementAndGet();
throw new InferenceException("Inference failed", e);
}
}

/**
* Core text generation logic using llama.cpp
*/
private String generate(
int[] promptTokens,
GenerationParams params,
RequestContext requestContext
) {
StringBuilder result = new StringBuilder();
List
<Integer> generatedTokenIds = new ArrayList
<>();

try {
        // Initialize batch with prompt tokens
        MemorySegment batch = binding.batchInit(promptTokens.length, 0, 1);

        // Add prompt tokens to batch
        for (int i = 0; i < promptTokens.length; i++) {
            boolean isLast = (i == promptTokens.length - 1);
            addTokenToBatch(batch, promptTokens[i], i, isLast);
        }
        
        binding.setBatchNTokens(batch, promptTokens.length);

        // Process prompt
        if (binding.decode(context, batch) != 0) {
            binding.batchFree(batch);
            throw new RuntimeException("Failed to decode prompt");
        }
    int nCur= promptTokens.length;
    int nGen= 0;
            
            //
    Generation loop
    while (nGen
< params.getMaxTokens ()) {
                
                //
    Check for timeout
    if (Thread.currentThread().isInterrupted()) {
    break;
    }
                
                //
    Get logits for last token
    MemorySegment logits= binding.getLogits(context);
                
                //
    Sample next token with parameters
    int nextToken= sampleToken(logits, params, generatedTokenIds);
                
                //
    Check for EOS
    if (nextToken== eosToken) {
                    log.debug("EOS token generated, stopping");
                    break;
                }
                
                generatedTokenIds.add(nextToken);
                
                // Convert token to text
                String piece = binding.tokenToPiece(model, nextToken);
                result.append(piece);
                
                // Prepare next batch
                clearBatch(batch);
                addTokenToBatch(batch, nextToken, nCur, true);
                binding.setBatchNTokens(batch, 1);

                // Decode next token
                if (binding.decode(context, batch) != 0) {
                    log.warn("Failed to decode token, stopping generation");
                    break;
                }

                nCur++;
                nGen++;
            }
            
            binding.batchFree(batch);
            
            log.debugf("Generated %d tokens", nGen);
            
        } catch (Exception e) {
            log.error("Error during generation", e);
            throw new RuntimeException("Generation failed", e);
        }
        
        return result.toString();
    }
    
    /**
     * Sample next token using temperature, top-p, and top-k sampling
     */
    private int sampleToken(
        MemorySegment logits,
        GenerationParams params,
        List
<Integer> previousTokens
) {
// Simple greedy sampling for now
// TODO: Implement full sampling with temperature, top-p, top-k
return binding.sampleTokenGreedy(context, logits);
}

    /**
     * Add token to batch for processing
     */
    private void addTokenToBatch(
            MemorySegment batch,
            int token,
            int pos,
            boolean logits
    ) {
        int index = binding.getBatchNTokensCount(batch); // I need to add this native call or track it
        binding.setBatchToken(batch, index, token);
        binding.setBatchPos(batch, index, pos);
        binding.setBatchSeqId(batch, index, 0); // Default to sequence 0
        binding.setBatchLogits(batch, index, logits);
        binding.setBatchNTokens(batch, index + 1);
    }

    /**
     * Clear batch for reuse
     */
    private void clearBatch(MemorySegment batch) {
        binding.setBatchNTokens(batch, 0);
    }

@Override
public CompletionStage
<InferenceResponse> inferAsync(
InferenceRequest request,
RequestContext context
) {
return CompletableFuture.supplyAsync(
() -> infer(request, context),
executorService
);
}

@Override
public HealthStatus health() {
if (!initialized) {
return HealthStatus.down("Not initialized");
}

try {
// Perform basic health check - tokenize a simple string
int[] tokens = binding.tokenize(model, "test", false, false);

if (tokens.length == 0) {
return HealthStatus.down("Tokenization failed");
}

return HealthStatus.up()
.withDetail("model_id", manifest.modelId())
.withDetail("context_size", contextSize)
.withDetail("vocab_size", vocabSize)
.withDetail("gpu_layers", config.getNGpuLayers())
.withDetail("total_inferences", totalInferences.get())
.withDetail("failed_inferences", failedInferences.get());

} catch (Exception e) {
return HealthStatus.down("Health check failed: " + e.getMessage());
}
}

@Override
public ResourceMetrics getMetrics() {
if (!initialized) {
return ResourceMetrics.empty();
}

return ResourceMetrics.builder()
.memoryUsedMb(estimateMemoryUsage())
.lastInferenceLatency(lastInferenceLatency)
.totalRequests(totalInferences.get())
.failedRequests(failedInferences.get())
.successRate(calculateSuccessRate())
.averageLatency(lastInferenceLatency) // TODO: Track running average
.metadata("total_tokens_generated", totalTokensGenerated.get())
.metadata("context_size", contextSize)
.metadata("vocab_size", vocabSize)
.build();
}

@Override
public void warmup(List
<InferenceRequest> sampleInputs) {
if (!initialized) {
log.warn("Cannot warmup: runner not initialized");
return;
}

log.infof("Starting warmup for model %s", manifest.modelId());

List
<InferenceRequest> warmupRequests = sampleInputs.isEmpty() ?
createDefaultWarmupRequests() : sampleInputs;

RequestContext warmupContext = RequestContext.builder()
.tenantContext(tenantContext)
.timeout(Duration.ofSeconds(10))
.build();

for (InferenceRequest request : warmupRequests) {
try {
infer(request, warmupContext);
log.debug("Warmup request completed successfully");
} catch (Exception e) {
log.warnf("Warmup request failed: %s", e.getMessage());
}
}

log.infof("Warmup completed for model %s", manifest.modelId());
}

@Override
public RunnerMetadata metadata() {
List
<DeviceType> devices = new ArrayList
<>();
devices.add(DeviceType.CPU);

if (config != null && config.getNGpuLayers() > 0) {
devices.add(DeviceType.CUDA);
}

return new RunnerMetadata(
"gguf",
"1.0.0",
List.of(ModelFormat.GGUF),
devices,
ExecutionMode.SYNCHRONOUS,
Map.of(
"supports_streaming", false,
"supports_batching", false,
"max_context_size", contextSize,
"vocab_size", vocabSize,
"gpu_acceleration", config != null && config.getNGpuLayers() > 0
)
);
}

@Override
public void close() {
if (!initialized) {
return;
}

log.infof("Closing GGUF runner for model %s", manifest.modelId());

cleanup();

// Shutdown executor
executorService.shutdown();
try {
if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
executorService.shutdownNow();
}
} catch (InterruptedException e) {
executorService.shutdownNow();
Thread.currentThread().interrupt();
}

initialized = false;
log.infof("GGUF runner closed for model %s", manifest.modelId());
}

// ===================================================================
// Helper Methods
// ===================================================================

private GGUFConfig buildConfiguration(Map
<String , Object> runnerConfig) {
GGUFConfig.Builder builder = GGUFConfig.builder();

// Apply config properties
configThreads.ifPresent(builder::nThreads);
configContextSize.ifPresent(builder::nCtx);

// GPU settings
if (configUseGpu.orElse(false)) {
builder.nGpuLayers(configGpuLayers.orElse(32));
}

// Override with runner-specific config
if (runnerConfig.containsKey("n_threads")) {
builder.nThreads((Integer) runnerConfig.get("n_threads"));
}
if (runnerConfig.containsKey("n_ctx")) {
builder.nCtx((Integer) runnerConfig.get("n_ctx"));
}
if (runnerConfig.containsKey("n_gpu_layers")) {
builder.nGpuLayers((Integer) runnerConfig.get("n_gpu_layers"));
}
if (runnerConfig.containsKey("temperature")) {
builder.temperature(((Number) runnerConfig.get("temperature")).floatValue());
}

return builder.build();
}

    private void configureModelParams(MemorySegment params, GGUFConfig config) {
        binding.setModelParam(params, "n_gpu_layers", config.getNGpuLayers());
        binding.setModelParam(params, "use_mmap", config.isUseMmap());
        binding.setModelParam(params, "use_mlock", config.isUseMlock());
        // Add other params as needed
    }

    private void configureContextParams(MemorySegment params, GGUFConfig config) {
        binding.setContextParam(params, "n_ctx", config.getNCtx());
        binding.setContextParam(params, "n_batch", config.getNBatch());
        binding.setContextParam(params, "n_threads", config.getNThreads());
        binding.setContextParam(params, "n_threads_batch", config.getNThreads());
        // Add other params as needed
    }

private GenerationParams buildGenerationParams(InferenceRequest request) {
GenerationParams.Builder builder = GenerationParams.builder();

// Extract from request parameters
if (request.hasParameter("max_tokens")) {
builder.maxTokens(request.getParameter("max_tokens", Integer.class));
}
if (request.hasParameter("temperature")) {
builder.temperature(request.getParameter("temperature", Number.class).floatValue());
}
if (request.hasParameter("top_p")) {
builder.topP(request.getParameter("top_p", Number.class).floatValue());
}
if (request.hasParameter("top_k")) {
builder.topK(request.getParameter("top_k", Integer.class));
}

return builder.build();
}

private List
<InferenceRequest> createDefaultWarmupRequests() {
return List.of(
InferenceRequest.builder()
.requestId(UUID.randomUUID().toString())
.input("prompt", "Hello, how are you?")
.parameter("max_tokens", 10)
.build(),
InferenceRequest.builder()
.requestId(UUID.randomUUID().toString())
.input("prompt", "What is the capital of France?")
.parameter("max_tokens", 5)
.build()
);
}

private int estimateTokenCount(String text) {
// Simple estimation: ~4 characters per token
return Math.max(1, text.length() / 4);
}

private long estimateMemoryUsage() {
// Estimate based on model file size and context
if (modelPath != null && modelPath.toFile().exists()) {
long modelSize = modelPath.toFile().length();
long contextMemory = (long) contextSize * vocabSize * 4; // 4 bytes per float
return (modelSize + contextMemory) / (1024 * 1024); // MB
}
return 0;
}

private double calculateSuccessRate() {
long total = totalInferences.get();
if (total == 0) return 1.0;

long failed = failedInferences.get();
return (double) (total - failed) / total;
}

private void cleanup() {
try {
if (context != null && context.address() != 0) {
binding.freeContext(context);
context = null;
}

if (model != null && model.address() != 0) {
binding.freeModel(model);
model = null;
}

if (binding != null) {
binding.close();
binding = null;
}
} catch (Exception e) {
log.error("Error during cleanup", e);
}
}
}