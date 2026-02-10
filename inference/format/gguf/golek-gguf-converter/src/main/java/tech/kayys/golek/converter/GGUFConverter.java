package tech.kayys.golek.converter;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.infrastructure.Infrastructure;
import tech.kayys.golek.converter.model.ConversionProgress;
import tech.kayys.golek.converter.model.ConversionResult;
import tech.kayys.golek.converter.model.GGUFConversionParams;
import tech.kayys.golek.converter.model.ModelInfo;
import tech.kayys.golek.converter.model.QuantizationType;
import tech.kayys.golek.spi.model.ModelFormat;
import jakarta.enterprise.context.ApplicationScoped;
import java.lang.foreign.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * High-level GGUF model converter service.
 * 
 * <p>
 * Provides enterprise-grade model conversion with:
 * <ul>
 * <li>Progress tracking and cancellation</li>
 * <li>Reactive/async API using Mutiny</li>
 * <li>Resource management and cleanup</li>
 * <li>Multi-tenancy support</li>
 * <li>Comprehensive error handling</li>
 * </ul>
 * 
 * @author Bhangun
 * @version 1.0.0
 */
@ApplicationScoped
public class GGUFConverter {

    private static final Logger log = LoggerFactory.getLogger(GGUFConverter.class);

    private final AtomicLong conversionIdCounter = new AtomicLong(0);
    private final ConcurrentHashMap<Long, ConversionContext> activeConversions = new ConcurrentHashMap<>();

    /**
     * Detect model format from path.
     *
     * @param modelPath path to model file or directory
     * @return detected format or UNKNOWN
     */
    public ModelFormat detectFormat(Path modelPath) {
        if (modelPath == null) {
            log.warn("Cannot detect format: modelPath is null");
            return ModelFormat.UNKNOWN;
        }
        
        try (Arena arena = Arena.ofConfined()) {
            String format = GGUFNative.detectFormat(arena, modelPath.toString());
            if (format != null) {
                ModelFormat modelFormat = ModelFormat.fromId(format);
                log.debug("Detected format for {}: {}", modelPath, modelFormat);
                return modelFormat;
            } else {
                log.warn("Could not detect format for: {}", modelPath);
                return ModelFormat.UNKNOWN;
            }
        } catch (Exception e) {
            log.warn("Failed to detect format for {}: {}", modelPath, e.getMessage(), e);
            return ModelFormat.UNKNOWN;
        }
    }

    /**
     * Extract model information without converting.
     *
     * @param modelPath path to model
     * @return model information
     * @throws GGUFException if validation fails
     */
    public ModelInfo getModelInfo(Path modelPath) {
        if (modelPath == null) {
            log.warn("Cannot get model info: modelPath is null");
            throw new GGUFException("Model path cannot be null");
        }
        
        if (!Files.exists(modelPath)) {
            log.warn("Cannot get model info: model path does not exist: {}", modelPath);
            throw new GGUFException("Model path does not exist: " + modelPath);
        }
        
        GGUFConversionParams params = GGUFConversionParams.builder()
                .inputPath(modelPath)
                .outputPath(Path.of("/tmp/dummy")) // Not used for validation
                .build();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment paramsSegment = createParamsSegment(arena, params, null, null, null);
            MemorySegment ctx = GGUFNative.createContext(paramsSegment);

            try {
                MemorySegment infoSegment = arena.allocate(GGUFNative.MODEL_INFO_LAYOUT);
                int result = GGUFNative.validateInput(ctx, infoSegment);

                if (result != 0) {
                    String errorMsg = GGUFNative.getLastError();
                    log.error("Model validation failed for {}: {}", modelPath, errorMsg);
                    throw new GGUFException("Validation failed: " + errorMsg);
                }

                ModelInfo modelInfo = extractModelInfo(infoSegment, modelPath);
                log.debug("Extracted model info for {}: {}", modelPath, modelInfo);
                return modelInfo;

            } finally {
                GGUFNative.freeContext(ctx);
            }
        } catch (GGUFException e) {
            // Re-throw GGUFExceptions as-is
            throw e;
        } catch (Exception e) {
            log.error("Failed to get model info for {}", modelPath, e);
            throw new GGUFException("Failed to extract model info: " + e.getMessage(), e);
        }
    }

    /**
     * Convert model synchronously with progress callback.
     *
     * @param params           conversion parameters
     * @param progressCallback optional progress callback
     * @return conversion result
     * @throws GGUFException if conversion fails
     */
    public ConversionResult convert(GGUFConversionParams params,
            Consumer<ConversionProgress> progressCallback) {
        params.validate();

        long conversionId = conversionIdCounter.incrementAndGet();
        ConversionContext context = new ConversionContext(conversionId, params);
        activeConversions.put(conversionId, context);

        try {
            log.info("Starting conversion {}: {} -> {}",
                    conversionId, params.getInputPath(), params.getOutputPath());

            try (Arena arena = Arena.ofConfined()) {
                // Create progress callback stub
                MemorySegment progressStub = createProgressCallback(arena, conversionId, progressCallback);

                // Create log callback stub
                MemorySegment logStub = createLogCallback(arena, conversionId);

                // Create parameters
                MemorySegment paramsSegment = createParamsSegment(
                        arena, params, progressStub, logStub, null);

                // Create conversion context
                MemorySegment nativeCtx = GGUFNative.createContext(paramsSegment);
                context.setNativeContext(nativeCtx);

                try {
                    // Validate input
                    MemorySegment infoSegment = arena.allocate(GGUFNative.MODEL_INFO_LAYOUT);
                    int result = GGUFNative.validateInput(nativeCtx, infoSegment);

                    if (result != 0) {
                        throw new GGUFException("Input validation failed: " +
                                GGUFNative.getLastError());
                    }

                    ModelInfo inputInfo = extractModelInfo(infoSegment, params.getInputPath());
                    log.info("Input model info: {}", inputInfo);

                    // Execute conversion with periodic progress checks
                    long startTime = System.currentTimeMillis();
                    
                    // Start a background thread to periodically check progress
                    Thread progressChecker = null;
                    if (progressCallback != null) {
                        progressChecker = new Thread(() -> {
                            try {
                                float lastProgress = 0.0f;
                                while (true) {
                                    float currentProgress = GGUFNative.getProgress(nativeCtx);
                                    
                                    if (currentProgress >= 0.0f && Math.abs(currentProgress - lastProgress) > 0.01f) {
                                        // Only report if progress has changed significantly
                                        ConversionProgress progress = new ConversionProgress.Builder()
                                                .conversionId(conversionId)
                                                .progress(currentProgress)
                                                .stage("Processing") // This will be updated by native callbacks
                                                .timestamp(System.currentTimeMillis())
                                                .build();
                                        
                                        progressCallback.accept(progress);
                                        lastProgress = currentProgress;
                                    }
                                    
                                    // Check if conversion is complete
                                    if (currentProgress >= 1.0f) {
                                        break;
                                    }
                                    
                                    Thread.sleep(500); // Check every 500ms
                                }
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                            } catch (Exception e) {
                                log.warn("Error in progress checker thread for conversion {}: {}", conversionId, e.getMessage());
                            }
                        });
                        
                        progressChecker.start();
                    }

                    result = GGUFNative.convert(nativeCtx);
                    long duration = System.currentTimeMillis() - startTime;

                    // Interrupt the progress checker if it's still running
                    if (progressChecker != null) {
                        progressChecker.interrupt();
                        try {
                            progressChecker.join(1000); // Wait up to 1 second for thread to finish
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    }

                    if (result != 0) {
                        if (result == -6) { // GGUF_ERROR_CANCELLED
                            throw new GGUFException("Conversion was cancelled");
                        }
                        throw new GGUFException("Conversion failed: " +
                                GGUFNative.getLastError());
                    }

                    // Get output file info
                    long outputSize = Files.size(params.getOutputPath());
                    double compressionRatio = (double) inputInfo.getFileSize() / outputSize;

                    log.info("Conversion {} completed in {}ms, compression ratio: {:.2f}x",
                            conversionId, duration, compressionRatio);

                    // Report final progress
                    if (progressCallback != null) {
                        ConversionProgress finalProgress = new ConversionProgress.Builder()
                                .conversionId(conversionId)
                                .progress(1.0f)
                                .stage("Complete")
                                .timestamp(System.currentTimeMillis())
                                .build();
                        progressCallback.accept(finalProgress);
                    }

                    return new ConversionResult.Builder()
                            .conversionId(conversionId)
                            .success(true)
                            .inputInfo(inputInfo)
                            .outputPath(params.getOutputPath())
                            .outputSize(outputSize)
                            .durationMs(duration)
                            .compressionRatio(compressionRatio)
                            .build();

                } finally {
                    GGUFNative.freeContext(nativeCtx);
                }
            }

        } catch (Exception e) {
            log.error("Conversion {} failed", conversionId, e);
            throw new GGUFException("Conversion failed", e);

        } finally {
            activeConversions.remove(conversionId);
        }
    }

    /**
     * Convert model asynchronously (reactive).
     * 
     * @param params conversion parameters
     * @return Uni emitting conversion result
     */
    public Uni<ConversionResult> convertAsync(GGUFConversionParams params) {
        return Uni.createFrom().item(() -> convert(params, null))
                .runSubscriptionOn(Infrastructure.getDefaultExecutor());
    }

    /**
     * Convert model with progress stream.
     * 
     * @param params conversion parameters
     * @return Multi emitting progress updates and final result
     */
    public Multi<Object> convertWithProgress(GGUFConversionParams params) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                ConversionResult result = convert(params, progress -> {
                    emitter.emit(progress);
                });
                emitter.emit(result);
                emitter.complete();
            } catch (Exception e) {
                emitter.fail(e);
            }
        });
    }

    /**
     * Cancel an active conversion.
     *
     * @param conversionId conversion ID
     * @return true if cancelled, false if not found
     */
    public boolean cancelConversion(long conversionId) {
        try {
            ConversionContext context = activeConversions.get(conversionId);
            if (context == null) {
                log.debug("Conversion {} not found for cancellation", conversionId);
                return false;
            }

            MemorySegment nativeCtx = context.getNativeContext();
            if (nativeCtx != null) {
                GGUFNative.cancel(nativeCtx);
                log.info("Cancellation requested for conversion {}", conversionId);
                return true;
            } else {
                log.debug("Native context not found for conversion {} during cancellation", conversionId);
                return false;
            }
        } catch (Exception e) {
            log.error("Error during cancellation of conversion {}", conversionId, e);
            return false;
        }
    }

    /**
     * Get available quantization types from native library.
     *
     * @return array of available quantization types
     */
    public QuantizationType[] getAvailableQuantizations() {
        try (Arena arena = Arena.ofConfined()) {
            String[] nativeQuantizations = GGUFNative.getAvailableQuantizations();
            return Arrays.stream(nativeQuantizations)
                    .map(QuantizationType::fromNativeName)
                    .filter(Objects::nonNull)
                    .toArray(QuantizationType[]::new);
        } catch (Exception e) {
            log.warn("Failed to get native quantization types, falling back to default", e);
            // Return default quantization types if native call fails
            return QuantizationType.values();
        }
    }
    
    /**
     * Check if a conversion is currently active.
     *
     * @param conversionId conversion ID
     * @return true if conversion is active, false otherwise
     */
    public boolean isConversionActive(long conversionId) {
        return activeConversions.containsKey(conversionId);
    }

    /**
     * Verify GGUF file integrity.
     *
     * @param ggufPath path to GGUF file
     * @return model info if valid
     * @throws GGUFException if file is invalid
     */
    public ModelInfo verifyGGUF(Path ggufPath) {
        if (ggufPath == null) {
            log.warn("Cannot verify GGUF: ggufPath is null");
            throw new GGUFException("GGUF path cannot be null");
        }
        
        if (!Files.exists(ggufPath)) {
            log.warn("Cannot verify GGUF: file does not exist: {}", ggufPath);
            throw new GGUFException("GGUF file does not exist: " + ggufPath);
        }
        
        if (!Files.isRegularFile(ggufPath)) {
            log.warn("Cannot verify GGUF: path is not a regular file: {}", ggufPath);
            throw new GGUFException("GGUF path is not a regular file: " + ggufPath);
        }
        
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment infoSegment = arena.allocate(GGUFNative.MODEL_INFO_LAYOUT);
            int result = GGUFNative.verifyFile(arena, ggufPath.toString(), infoSegment);

            if (result != 0) {
                String errorMsg = GGUFNative.getLastError();
                log.error("GGUF verification failed for {}: {}", ggufPath, errorMsg);
                throw new GGUFException("GGUF verification failed: " + errorMsg);
            }

            ModelInfo modelInfo = extractModelInfo(infoSegment, ggufPath);
            log.debug("Verified GGUF file {}: {}", ggufPath, modelInfo);
            return modelInfo;

        } catch (GGUFException e) {
            // Re-throw GGUFExceptions as-is
            throw e;
        } catch (Exception e) {
            log.error("Failed to verify GGUF file {}", ggufPath, e);
            throw new GGUFException("GGUF verification failed: " + e.getMessage(), e);
        }
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    private MemorySegment createParamsSegment(Arena arena,
            GGUFConversionParams params,
            MemorySegment progressCallback,
            MemorySegment logCallback,
            MemorySegment userData) {
        MemorySegment segment = GGUFNative.defaultParams(arena);

        // Set string parameters
        segment.set(ValueLayout.ADDRESS, 0,
                arena.allocateFrom(params.getInputPath().toString()));
        segment.set(ValueLayout.ADDRESS, 8,
                arena.allocateFrom(params.getOutputPath().toString()));

        if (params.getModelType() != null) {
            segment.set(ValueLayout.ADDRESS, 16,
                    arena.allocateFrom(params.getModelType()));
        }

        segment.set(ValueLayout.ADDRESS, 24,
                arena.allocateFrom(params.getQuantization().getNativeName()));

        // Set integer parameters
        segment.set(ValueLayout.JAVA_INT, 32, params.isVocabOnly() ? 1 : 0);
        segment.set(ValueLayout.JAVA_INT, 36, params.isUseMmap() ? 1 : 0);
        segment.set(ValueLayout.JAVA_INT, 40, params.getNumThreads());

        if (params.getVocabType() != null) {
            segment.set(ValueLayout.ADDRESS, 48,
                    arena.allocateFrom(params.getVocabType()));
        }

        segment.set(ValueLayout.JAVA_INT, 56, params.getPadVocab());

        // Set callbacks
        if (progressCallback != null) {
            segment.set(ValueLayout.ADDRESS, 72, progressCallback);
        }
        if (logCallback != null) {
            segment.set(ValueLayout.ADDRESS, 80, logCallback);
        }
        if (userData != null) {
            segment.set(ValueLayout.ADDRESS, 88, userData);
        }

        return segment;
    }

    private MemorySegment createProgressCallback(Arena arena, long conversionId,
            Consumer<ConversionProgress> callback) {
        if (callback == null) {
            return MemorySegment.NULL;
        }

        return Linker.nativeLinker().upcallStub(
                (float progress, MemorySegment stagePtr, MemorySegment userDataPtr) -> {
                    String stage = stagePtr.address() != 0 ? stagePtr.reinterpret(Long.MAX_VALUE).getString(0) : "";

                    ConversionProgress prog = ConversionProgress.builder()
                            .conversionId(conversionId)
                            .progress(progress)
                            .stage(stage)
                            .timestamp(System.currentTimeMillis())
                            .build();

                    callback.accept(prog);
                },
                GGUFNative.PROGRESS_CALLBACK_DESC,
                arena);
    }

    private MemorySegment createLogCallback(Arena arena, long conversionId) {
        return Linker.nativeLinker().upcallStub(
                (int level, MemorySegment messagePtr, MemorySegment userDataPtr) -> {
                    if (messagePtr.address() == 0)
                        return;

                    String message = messagePtr.reinterpret(Long.MAX_VALUE).getString(0);

                    switch (level) {
                        case 0 -> log.debug("[Conversion {}] {}", conversionId, message);
                        case 1 -> log.info("[Conversion {}] {}", conversionId, message);
                        case 2 -> log.warn("[Conversion {}] {}", conversionId, message);
                        case 3 -> log.error("[Conversion {}] {}", conversionId, message);
                    }
                },
                GGUFNative.LOG_CALLBACK_DESC,
                arena);
    }

    private ModelInfo extractModelInfo(MemorySegment infoSegment, Path sourcePath) {
        // Extract string fields
        String modelType = extractString(infoSegment, 0, 64);
        String architecture = extractString(infoSegment, 64, 64);
        String quantization = extractString(infoSegment, 200, 32);

        // Extract numeric fields
        long paramCount = infoSegment.get(ValueLayout.JAVA_LONG, 128);
        int numLayers = infoSegment.get(ValueLayout.JAVA_INT, 136);
        int hiddenSize = infoSegment.get(ValueLayout.JAVA_INT, 140);
        int vocabSize = infoSegment.get(ValueLayout.JAVA_INT, 144);
        int contextLength = infoSegment.get(ValueLayout.JAVA_INT, 148);
        long fileSize = infoSegment.get(ValueLayout.JAVA_LONG, 232);

        ModelFormat format = detectFormat(sourcePath);

        return ModelInfo.builder()
                .modelType(modelType.isEmpty() ? null : modelType)
                .architecture(architecture.isEmpty() ? null : architecture)
                .parameterCount(paramCount)
                .numLayers(numLayers)
                .hiddenSize(hiddenSize)
                .vocabSize(vocabSize)
                .contextLength(contextLength)
                .quantization(quantization.isEmpty() ? null : quantization)
                .fileSize(fileSize)
                .format(format)
                .build();
    }

    private String extractString(MemorySegment segment, long offset, int maxLength) {
        for (int i = 0; i < maxLength; i++) {
            if (segment.get(ValueLayout.JAVA_BYTE, offset + i) == 0) {
                if (i == 0)
                    return "";
                byte[] bytes = new byte[i];
                MemorySegment.copy(segment, ValueLayout.JAVA_BYTE, offset,
                        bytes, 0, i);
                return new String(bytes);
            }
        }
        byte[] bytes = new byte[maxLength];
        MemorySegment.copy(segment, ValueLayout.JAVA_BYTE, offset,
                bytes, 0, maxLength);
        return new String(bytes).trim();
    }

    /**
     * Context for tracking active conversions.
     */
    private static class ConversionContext {
        private final long id;
        private final GGUFConversionParams params;
        private volatile MemorySegment nativeContext;

        ConversionContext(long id, GGUFConversionParams params) {
            this.id = id;
            this.params = params;
        }

        void setNativeContext(MemorySegment nativeContext) {
            this.nativeContext = nativeContext;
        }

        MemorySegment getNativeContext() {
            return nativeContext;
        }
    }
}
