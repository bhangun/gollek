package tech.kayys.gollek.sdk.litert.inference;

import tech.kayys.gollek.sdk.litert.LiteRTException;
import tech.kayys.gollek.sdk.litert.LiteRTMetrics;
import tech.kayys.gollek.sdk.litert.config.LiteRTConfig;
import tech.kayys.gollek.sdk.litert.model.LiteRTModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.provider.litert.LiteRTInferenceRunner;
import tech.kayys.gollek.provider.litert.LiteRTNativeBindings;
import tech.kayys.gollek.provider.litert.LiteRTTokenizer;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CompletableFuture;

/**
 * LiteRT Inference Engine - Internal implementation.
 * 
 * Wraps the LiteRT runner and provides high-level inference APIs.
 */
public class LiteRTInferenceEngine implements Closeable {

    private final LiteRTConfig config;
    private final LiteRTNativeBindings nativeBindings;
    private final Map<String, LiteRTInferenceRunner> runners = new ConcurrentHashMap<>();
    private final Map<String, LiteRTModelInfo> modelInfos = new ConcurrentHashMap<>();
    
    private volatile boolean closed = false;

    public LiteRTInferenceEngine(LiteRTConfig config) {
        this.config = config;
        this.nativeBindings = new LiteRTNativeBindings(resolveLibraryPath());
    }

    private Path resolveLibraryPath() {
        String home = System.getProperty("user.home");
        // Try the new official LiteRT name first
        Path libPath = Paths.get(home, ".gollek", "libs", "libLiteRt.dylib");
        if (Files.exists(libPath)) return libPath;

        // Fallback to the old name
        libPath = Paths.get(home, ".gollek", "libs", "libtensorflowlite_c.dylib");
        if (Files.exists(libPath)) return libPath;

        // Fallback for Linux
        Path linuxLib = Paths.get(home, ".gollek", "libs", "libLiteRt.so");
        if (Files.exists(linuxLib)) return linuxLib;
        
        linuxLib = Paths.get(home, ".gollek", "libs", "libtensorflowlite_c.so");
        if (Files.exists(linuxLib)) return linuxLib;

        return libPath;
    }

    /**
     * Load model from file path.
     */
    public void loadModel(String modelId, Path modelPath) {
        ensureNotClosed();
        try {
            LiteRTTokenizer tokenizer = LiteRTTokenizer.create(modelPath);
            LiteRTInferenceRunner runner = new LiteRTInferenceRunner(
                    nativeBindings, 
                    modelPath, 
                    tokenizer, 
                    config.getDelegate() == LiteRTConfig.Delegate.GPU, 
                    config.getNumThreads()
            );
            runner.initialize();
            runners.put(modelId, runner);
            
            modelInfos.put(modelId, LiteRTModelInfo.builder()
                    .modelId(modelId)
                    .modelPath(modelPath.toString())
                    .metadata(Map.of("format", "LiteRT"))
                    .build());
            
        } catch (Exception e) {
            throw new LiteRTException("Failed to load model: " + modelId, e);
        }
    }

    /**
     * Load model from byte array.
     */
    public void loadModel(String modelId, byte[] modelData) {
        throw new UnsupportedOperationException("Loading model from byte array not yet supported in SDK");
    }

    /**
     * Run synchronous inference.
     */
    public InferenceResponse infer(InferenceRequest request) {
        ensureNotClosed();
        String modelId = request.getModel();
        LiteRTInferenceRunner runner = runners.get(modelId);
        
        if (runner == null) {
            throw new LiteRTException("Model not loaded: " + modelId);
        }

        long start = System.currentTimeMillis();
        StringBuilder output = new StringBuilder();
        
        try {
            String prompt = request.getPrompt();
            runner.generate(prompt, output::append);
            
            long duration = System.currentTimeMillis() - start;
            
            return InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .model(modelId)
                    .content(output.toString())
                    .durationMs(duration)
                    .build();
        } catch (Exception e) {
            throw new LiteRTException("Inference failed for model: " + modelId, e);
        }
    }

    /**
     * Run asynchronous inference.
     */
    public CompletableFuture<InferenceResponse> inferAsync(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> infer(request));
    }

    /**
     * Run batch inference.
     */
    public List<InferenceResponse> inferBatch(List<InferenceRequest> requests) {
        return requests.stream()
                .map(req -> infer(req))
                .toList();
    }

    /**
     * Run batch inference asynchronously.
     */
    public CompletableFuture<List<InferenceResponse>> inferBatchAsync(List<InferenceRequest> requests) {
        return CompletableFuture.supplyAsync(() -> inferBatch(requests));
    }

    /**
     * Get model information.
     */
    public LiteRTModelInfo getModelInfo(String modelId) {
        return modelInfos.get(modelId);
    }

    /**
     * List all loaded models.
     */
    public List<String> listModels() {
        return new ArrayList<>(runners.keySet());
    }

    /**
     * Unload a model.
     */
    public void unloadModel(String modelId) {
        LiteRTInferenceRunner runner = runners.remove(modelId);
        if (runner != null) {
            runner.close();
        }
        modelInfos.remove(modelId);
    }

    /**
     * Get inference metrics.
     */
    public LiteRTMetrics getMetrics(String modelId) {
        // TODO: Implement actual metrics collection in runner
        return LiteRTMetrics.builder().build();
    }

    /**
     * Reset metrics.
     */
    public void resetMetrics() {
        // TODO: Implement
    }

    /**
     * Check if LiteRT is available.
     */
    public boolean isAvailable() {
        return Files.exists(resolveLibraryPath());
    }

    /**
     * Get LiteRT version.
     */
    public String getVersion() {
        return "0.1.0-SNAPSHOT";
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new LiteRTException("Inference engine is closed");
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        runners.values().forEach(LiteRTInferenceRunner::close);
        runners.clear();
        modelInfos.clear();
    }
}
