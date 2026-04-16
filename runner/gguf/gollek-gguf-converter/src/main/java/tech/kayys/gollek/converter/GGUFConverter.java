package tech.kayys.gollek.converter;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.infrastructure.Infrastructure;
import tech.kayys.gollek.converter.gguf.SafetensorToGgufConverter;
import tech.kayys.gollek.converter.gguf.GgmlType;
import tech.kayys.gollek.converter.model.ConversionProgress;
import tech.kayys.gollek.converter.model.ConversionResult;
import tech.kayys.gollek.converter.model.GGUFConversionParams;
import tech.kayys.gollek.converter.model.ModelMetadata;
import tech.kayys.gollek.converter.model.QuantizationType;
import tech.kayys.gollek.spi.model.ModelFormat;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
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
 * <li>Native Safetensors support via pure Java FFM implementation</li>
 * </ul>
 *
 * <p>
 * <b>Conversion Strategy:</b>
 * <ul>
 * <li><b>Safetensors input:</b> Uses {@link SafetensorToGgufConverter} for pure Java-based conversion</li>
 * <li><b>PyTorch/TensorFlow input:</b> Uses native GGUF bridge (gguf_bridge) via FFM</li>
 * <li><b>GGUF input:</b> Direct quantization via llama.cpp FFM bindings</li>
 * </ul>
 *
 * @author Bhangun
 * @version 1.0.0
 */
@ApplicationScoped
public class GGUFConverter {

    private static final Logger log = LoggerFactory.getLogger(GGUFConverter.class);
    private static final String DEFAULT_CONVERTER_BASE =
            System.getProperty("user.home") + "/.gollek/conversions";
    private static final String DEFAULT_MODEL_BASE =
            System.getProperty("user.home") + "/.gollek/models";

    private final AtomicLong conversionIdCounter = new AtomicLong(0);
    private final ConcurrentHashMap<Long, ConversionContext> activeConversions = new ConcurrentHashMap<>();

    /**
     * Get the native GGUF bridge version.
     *
     * @return version string
     */
    public String getVersion() {
        if (!isNativeAvailable()) {
            return "1.0.0-fallback";
        }
        try {
            return GGUFNative.getVersion();
        } catch (Exception e) {
            log.warn("Failed to get native version, using fallback: {}", e.getMessage());
            return "1.0.0-fallback";
        }
    }

    /**
     * Returns true when GGUF native bridge is available in current runtime.
     */
    public boolean isNativeAvailable() {
        return GGUFNative.isAvailable();
    }

    /**
     * Returns the reason when native bridge is unavailable.
     */
    public String getNativeUnavailableReason() {
        return GGUFNative.getUnavailableReason();
    }

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

        Path resolvedPath = resolveInputPath(modelPath);
        if (resolvedPath != null) {
            modelPath = resolvedPath;
        }

        log.info("Detecting format for: " + modelPath);
        log.info("Native available: " + isNativeAvailable());

        if (!isNativeAvailable()) {
            log.info("Native not available, using fallback");
            return detectFormatFallback(modelPath);
        }

        try (Arena arena = Arena.ofConfined()) {
            String format = GGUFNative.detectFormat(arena, modelPath.toString());
            if (format != null) {
                ModelFormat modelFormat = ModelFormat.fromId(format);
                log.debug("Detected format for {}: {}", modelPath, modelFormat);
                // If native detection returns UNKNOWN, try fallback detection
                if (modelFormat == ModelFormat.UNKNOWN) {
                    log.info("Native detectFormat returned UNKNOWN, trying fallback");
                    return detectFormatFallback(modelPath);
                }
                return modelFormat;
            } else {
                log.info("Native detectFormat returned null, using fallback");
                return detectFormatFallback(modelPath);
            }
        } catch (Exception e) {
            log.warn("Failed to detect format for {}: {}", modelPath, e.getMessage());
            log.info("Using fallback detection");
            return detectFormatFallback(modelPath);
        }
    }

    private ModelFormat detectFormatFallback(Path modelPath) {
        // Handle URI paths (file:/...) by converting to regular Path
        String pathStr = modelPath.toString();
        if (pathStr.startsWith("file:")) {
            try {
                modelPath = Path.of(new java.net.URI(pathStr));
            } catch (Exception e) {
                log.warn("Failed to parse URI path: " + e.getMessage());
            }
        }
        
        log.info("Fallback format detection for: " + modelPath);
        log.info("Is directory: " + Files.isDirectory(modelPath));
        log.info("Is regular file: " + Files.isRegularFile(modelPath));
        
        if (Files.isRegularFile(modelPath)) {
            String name = modelPath.getFileName().toString().toLowerCase();
            log.info("File extension: " + name.substring(name.lastIndexOf(".")));
            return ModelFormat.fromExtension(name.substring(name.lastIndexOf(".")));
        } else if (Files.isDirectory(modelPath)) {
            // Check SAFETENSORS first to avoid false positive with PYTORCH
            // (both have model.safetensors as marker, but SAFETENSORS is more specific)
            ModelFormat[] checkOrder = {
                ModelFormat.SAFETENSORS,
                ModelFormat.PYTORCH,
                ModelFormat.TENSORFLOW,
                ModelFormat.FLAX,
                ModelFormat.GGUF,
                ModelFormat.LITERT,
                ModelFormat.ONNX,
                ModelFormat.TENSORRT,
                ModelFormat.TORCHSCRIPT,
                ModelFormat.TENSORFLOW_SAVED_MODEL,
                ModelFormat.UNKNOWN
            };

            for (ModelFormat format : checkOrder) {
                if (format.getMarkerFiles() != null && !format.getMarkerFiles().isEmpty()) {
                    for (String mf : format.getMarkerFiles()) {
                        Path markerPath = modelPath.resolve(mf);
                        boolean exists = Files.exists(markerPath);
                        log.info("Checking marker " + mf + " for " + format.getId() + ": " + exists);
                        if (exists) {
                            log.info("Found marker " + mf + " - detected as " + format.getId());
                            return format;
                        }
                    }
                }
            }
        }
        log.warn("No format detected, returning UNKNOWN");
        return ModelFormat.UNKNOWN;
    }

    /**
     * Extract model information without converting.
     *
     * @param modelPath path to model
     * @return model information
     * @throws GGUFException if validation fails
     */
    public ModelMetadata getModelInfo(Path modelPath) {
        if (modelPath == null) {
            log.warn("Cannot get model info: modelPath is null");
            throw new GGUFException("Model path cannot be null");
        }

        Path resolvedPath = resolveInputPath(modelPath);
        if (resolvedPath != null) {
            modelPath = resolvedPath;
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

                ModelMetadata modelInfo = extractModelInfo(infoSegment, modelPath);
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
        params = resolveParams(params);
        params.validate();

        // Detect input format and delegate to appropriate converter
        ModelFormat format = detectFormat(params.getInputPath());
        log.info("Detected model format: " + format);
        log.info("Format ID: " + format.getId());
        log.info("Is SAFETENSORS: " + (format == ModelFormat.SAFETENSORS));

        if (format == ModelFormat.SAFETENSORS) {
            // Use pure Java Safetensors converter
            log.info("Detected Safetensors format, using Java FFM converter");
            return convertSafetensors(params, progressCallback);
        }

        log.info("Using native GGUF bridge for conversion (format: " + format + ")");
        // Use native GGUF bridge for other formats
        return convertWithNativeBridge(params, progressCallback);
    }

    /**
     * Convert Safetensors model using pure Java FFM implementation (streaming).
     */
    private ConversionResult convertSafetensors(
            GGUFConversionParams params,
            Consumer<ConversionProgress> progressCallback) {

        long conversionId = conversionIdCounter.incrementAndGet();
        ConversionContext context = new ConversionContext(conversionId, params);
        activeConversions.put(conversionId, context);

        try {
            Path inputPath = params.getInputPath();

            // Determine input directory
            Path inputDir;
            if (Files.isDirectory(inputPath)) {
                inputDir = inputPath;
            } else if (inputPath.toString().endsWith(".safetensors")) {
                inputDir = inputPath.getParent();
            } else {
                inputDir = inputPath;
            }

            log.info("Starting Safetensors to GGUF conversion {}: {} -> {} [{}]",
                    conversionId, inputPath, params.getOutputPath(), params.getQuantization());

            // Convert quantization type to GgmlType
            GgmlType ggmlType = switch (params.getQuantization()) {
                case F32 -> GgmlType.F32;
                case F16, BF16 -> GgmlType.F16;
                case Q8_0 -> GgmlType.Q8_0;
                case Q4_0 -> GgmlType.Q4_0;
                case Q4_1 -> GgmlType.Q4_1;
                case Q5_0 -> GgmlType.Q5_0;
                case Q5_1 -> GgmlType.Q5_1;
                case Q4_K_M, Q4_K_S -> GgmlType.Q4_K;
                case Q5_K_M, Q5_K_S -> GgmlType.Q5_K;
                case Q6_K -> GgmlType.Q6_K;
                default -> GgmlType.F16;
            };

            // Build options for streaming converter
            var opts = new SafetensorToGgufConverter.Options.Builder()
                    .inputDir(inputDir)
                    .outputFile(params.getOutputPath())
                    .quantType(ggmlType)
                    .verbose(true)
                    .onProgress((done, total) -> {
                        if (progressCallback != null) {
                            float progress = (float) done / total;
                            progressCallback.accept(new ConversionProgress.Builder()
                                    .conversionId(conversionId)
                                    .progress(progress)
                                    .stage("Converting tensors")
                                    .timestamp(System.currentTimeMillis())
                                    .build());
                        }
                    })
                    .build();

            // Execute streaming conversion
            SafetensorToGgufConverter.convert(opts);

            // Build result
            long outputSize = Files.size(params.getOutputPath());
            ConversionResult result = new ConversionResult.Builder()
                    .conversionId(conversionId)
                    .success(true)
                    .outputPath(params.getOutputPath())
                    .outputSize(outputSize)
                    .build();

            log.info("Safetensors conversion {} completed successfully", conversionId);
            return result;

        } catch (Exception e) {
            log.error("Safetensors conversion {} failed", conversionId, e);
            log.error("Exception type: " + e.getClass().getName());
            log.error("Exception message: " + e.getMessage());
            if (e.getCause() != null) {
                log.error("Caused by: " + e.getCause().getClass().getName() + ": " + e.getCause().getMessage());
            }
            throw new GGUFException("Safetensors conversion failed: " + e.getMessage(), e);
        } finally {
            activeConversions.remove(conversionId);
        }
    }

    /**
     * Convert model using native GGUF bridge (for PyTorch, TensorFlow, etc.).
     */
    private ConversionResult convertWithNativeBridge(
            GGUFConversionParams params,
            Consumer<ConversionProgress> progressCallback) {
        
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

                    ModelMetadata inputInfo = extractModelInfo(infoSegment, params.getInputPath());
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
                                log.warn("Error in progress checker thread for conversion {}: {}", conversionId,
                                        e.getMessage());
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
     * Resolve conversion parameters without executing a conversion.
     *
     * @param params conversion parameters
     * @return normalized parameters with resolved paths
     */
    public GGUFConversionParams resolveParams(GGUFConversionParams params) {
        return normalizeParams(params);
    }

    public Path resolveModelBasePath() {
        return resolveBasePath("GOLLEK_MODEL_BASE", "gollek.model.base", DEFAULT_MODEL_BASE);
    }

    public Path resolveConverterBasePath() {
        return resolveBasePath("GOLLEK_CONVERTER_BASE", "gollek.converter.base", DEFAULT_CONVERTER_BASE);
    }

    public String deriveOutputName(Path inputPath, QuantizationType quantization) {
        return buildOutputName(inputPath, quantization);
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
        if (!isNativeAvailable()) {
            return QuantizationType.values();
        }
        try (Arena arena = Arena.ofConfined()) {
            String[] nativeQuantizations = GGUFNative.getAvailableQuantizations();
            if (nativeQuantizations == null || nativeQuantizations.length == 0) {
                return QuantizationType.values();
            }
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
    public ModelMetadata verifyGGUF(Path ggufPath) {
        if (ggufPath == null) {
            log.warn("Cannot verify GGUF: ggufPath is null");
            throw new GGUFException("GGUF path cannot be null");
        }

        Path resolvedPath = resolveInputPath(ggufPath);
        if (resolvedPath != null) {
            ggufPath = resolvedPath;
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

            ModelMetadata modelInfo = extractModelInfo(infoSegment, ggufPath);
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

        ProgressCallback target = (float progress, MemorySegment stagePtr, MemorySegment userDataPtr) -> {
            String stage = stagePtr.address() != 0 ? stagePtr.reinterpret(Long.MAX_VALUE).getString(0L) : "";

            ConversionProgress prog = ConversionProgress.builder()
                    .conversionId(conversionId)
                    .progress(progress)
                    .stage(stage)
                    .timestamp(System.currentTimeMillis())
                    .build();

            callback.accept(prog);
        };

        try {
            MethodHandle handle = MethodHandles.lookup()
                    .findVirtual(ProgressCallback.class, "run",
                            MethodType.methodType(void.class, float.class, MemorySegment.class, MemorySegment.class))
                    .bindTo(target);

            return Linker.nativeLinker().upcallStub(
                    handle,
                    GGUFNative.PROGRESS_CALLBACK_DESC,
                    arena);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to create progress upcall stub", t);
        }
    }

    private MemorySegment createLogCallback(Arena arena, long conversionId) {
        LogCallback target = (int level, MemorySegment messagePtr, MemorySegment userDataPtr) -> {
            if (messagePtr.address() == 0)
                return;

            String message = messagePtr.reinterpret(Long.MAX_VALUE).getString(0L);

            switch (level) {
                case 0 -> log.debug("[Conversion {}] {}", conversionId, message);
                case 1 -> log.info("[Conversion {}] {}", conversionId, message);
                case 2 -> log.warn("[Conversion {}] {}", conversionId, message);
                case 3 -> log.error("[Conversion {}] {}", conversionId, message);
            }
        };

        try {
            MethodHandle handle = MethodHandles.lookup()
                    .findVirtual(LogCallback.class, "run",
                            MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class))
                    .bindTo(target);

            return Linker.nativeLinker().upcallStub(
                    handle,
                    GGUFNative.LOG_CALLBACK_DESC,
                    arena);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to create log upcall stub", t);
        }
    }

    @FunctionalInterface
    private interface ProgressCallback {
        void run(float progress, MemorySegment stagePtr, MemorySegment userDataPtr);
    }

    @FunctionalInterface
    private interface LogCallback {
        void run(int level, MemorySegment messagePtr, MemorySegment userDataPtr);
    }

    private ModelMetadata extractModelInfo(MemorySegment infoSegment, Path sourcePath) {
        // Extract string fields
        String modelType = extractString(infoSegment, 0, 64);
        String architecture = extractString(infoSegment, 64, 64);
        String quantization = extractString(infoSegment, 152, 32);

        // Extract numeric fields
        long paramCount = infoSegment.get(ValueLayout.JAVA_LONG, 128);
        int numLayers = infoSegment.get(ValueLayout.JAVA_INT, 136);
        int hiddenSize = infoSegment.get(ValueLayout.JAVA_INT, 140);
        int vocabSize = infoSegment.get(ValueLayout.JAVA_INT, 144);
        int contextLength = infoSegment.get(ValueLayout.JAVA_INT, 148);
        long fileSize = infoSegment.get(ValueLayout.JAVA_LONG, 184);

        ModelFormat format = detectFormat(sourcePath);

        return ModelMetadata.builder()
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

    private GGUFConversionParams normalizeParams(GGUFConversionParams params) {
        if (params == null) {
            throw new IllegalArgumentException("Conversion params cannot be null");
        }
        Path inputPath = resolveInputPath(params.getInputPath());
        if (inputPath == null) {
            inputPath = params.getInputPath();
        }
        Path outputPath = resolveOutputPath(params.getOutputPath(), inputPath, params.getQuantization());
        if (inputPath != null && outputPath != null && inputPath.normalize().equals(outputPath.normalize())) {
            throw new GGUFException("Output path must be different from input path");
        }
        if (outputPath != null && Files.exists(outputPath) && !params.isOverwriteExisting()) {
            throw new GGUFException("Output path already exists: " + outputPath
                    + " (set overwriteExisting=true to replace)");
        }

        return params.toBuilder()
                .inputPath(inputPath)
                .outputPath(outputPath)
                .build();
    }

    private Path resolveInputPath(Path inputPath) {
        if (inputPath == null) {
            return null;
        }
        if (inputPath.isAbsolute()) {
            return inputPath;
        }
        if (Files.exists(inputPath)) {
            return inputPath.toAbsolutePath().normalize();
        }
        Path modelBase = resolveModelBasePath();
        Path candidate = modelBase.resolve(inputPath).normalize();
        if (Files.exists(candidate)) {
            return candidate;
        }
        return inputPath;
    }

    private Path resolveOutputPath(Path outputPath, Path inputPath, QuantizationType quantization) {
        if (outputPath == null) {
            return null;
        }
        Path resolved = outputPath;
        if (!outputPath.isAbsolute()) {
            Path base = resolveConverterBasePath();
            resolved = base.resolve(outputPath).normalize();
        }

        if (Files.isDirectory(resolved) || outputPath.toString().endsWith("/") || outputPath.toString().endsWith("\\")) {
            String name = buildOutputName(inputPath, quantization);
            resolved = resolved.resolve(name);
        } else if (!resolved.getFileName().toString().toLowerCase().endsWith(".gguf")) {
            String name = resolved.getFileName().toString() + ".gguf";
            resolved = resolved.getParent() == null ? Path.of(name) : resolved.getParent().resolve(name);
        }

        try {
            Path parent = resolved.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
        } catch (Exception e) {
            throw new GGUFException("Failed to create output directory: " + e.getMessage(), e);
        }

        return resolved;
    }

    private Path resolveBasePath(String envKey, String propKey, String fallback) {
        String override = System.getenv(envKey);
        if (override == null || override.isBlank()) {
            override = System.getProperty(propKey);
        }
        String path = (override == null || override.isBlank()) ? fallback : override;
        return Path.of(path).toAbsolutePath().normalize();
    }

    private String buildOutputName(Path inputPath, QuantizationType quantization) {
        String baseName = "model";
        if (inputPath != null) {
            Path name = inputPath.getFileName();
            if (name != null) {
                baseName = name.toString();
                int dot = baseName.lastIndexOf('.');
                if (dot > 0) {
                    baseName = baseName.substring(0, dot);
                }
            }
        }
        String quant = quantization != null ? quantization.getNativeName() : "unknown";
        return baseName + "-" + quant + ".gguf";
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
