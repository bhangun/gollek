package tech.kayys.gollek.converter.java;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.infrastructure.Infrastructure;
import tech.kayys.gollek.converter.java.gguf.SafetensorToGgufConverter;
import tech.kayys.gollek.converter.java.gguf.GgmlType;
import tech.kayys.gollek.converter.model.ConversionProgress;
import tech.kayys.gollek.converter.model.ConversionResult;
import tech.kayys.gollek.converter.model.GGUFConversionParams;
import tech.kayys.gollek.converter.model.ModelMetadata;
import tech.kayys.gollek.converter.model.QuantizationType;
import tech.kayys.gollek.spi.model.ModelFormat;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Pure Java GGUF model converter service.
 * Supports only Safetensors as input format.
 */
@ApplicationScoped
public class JavaGGUFConverter {

    private static final Logger log = LoggerFactory.getLogger(JavaGGUFConverter.class);
    
    private final AtomicLong conversionIdCounter = new AtomicLong(0);
    private final ConcurrentHashMap<Long, GGUFConversionParams> activeConversions = new ConcurrentHashMap<>();

    public String getVersion() {
        return "1.0.0-java";
    }

    public ModelFormat detectFormat(Path modelPath) {
        if (modelPath == null) return ModelFormat.UNKNOWN;
        
        if (Files.isRegularFile(modelPath)) {
            String name = modelPath.getFileName().toString().toLowerCase();
            if (name.endsWith(".safetensors")) return ModelFormat.SAFETENSORS;
            if (name.endsWith(".gguf")) return ModelFormat.GGUF;
        } else if (Files.isDirectory(modelPath)) {
            if (Files.exists(modelPath.resolve("model.safetensors.index.json")) ||
                Files.exists(modelPath.resolve("model.safetensors"))) {
                return ModelFormat.SAFETENSORS;
            }
        }
        return ModelFormat.UNKNOWN;
    }

    public ConversionResult convert(GGUFConversionParams params,
            Consumer<ConversionProgress> progressCallback) {
        
        long conversionId = conversionIdCounter.incrementAndGet();
        activeConversions.put(conversionId, params);

        try {
            Path inputPath = params.getInputPath();
            Path inputDir = Files.isDirectory(inputPath) ? inputPath : inputPath.getParent();

            log.info("Starting Pure Java Safetensors to GGUF conversion {}: {} -> {}",
                    conversionId, inputPath, params.getOutputPath());

            // Convert quantization type to GgmlType
            GgmlType ggmlType = switch (params.getQuantization()) {
                case F32 -> GgmlType.F32;
                case F16, BF16 -> GgmlType.F16;
                case Q8_0 -> GgmlType.Q8_0;
                case Q4_0 -> GgmlType.Q4_0;
                case Q2_K -> GgmlType.Q2_K;
                case Q4_K_M, Q4_K_S -> GgmlType.Q4_K;
                case Q5_K_M, Q5_K_S -> GgmlType.Q5_K;
                case Q6_K -> GgmlType.Q6_K;
                default -> GgmlType.F16;
            };

            var opts = new SafetensorToGgufConverter.Options.Builder()
                    .inputDir(inputDir)
                    .outputFile(params.getOutputPath())
                    .quantType(ggmlType)
                    .verbose(true)
                    .onProgress((done, total) -> {
                        if (progressCallback != null) {
                            progressCallback.accept(new ConversionProgress.Builder()
                                    .conversionId(conversionId)
                                    .progress((float) done / total)
                                    .stage("Converting tensors")
                                    .timestamp(System.currentTimeMillis())
                                    .build());
                        }
                    })
                    .build();

            SafetensorToGgufConverter.convert(opts);

            long outputSize = Files.size(params.getOutputPath());
            return new ConversionResult.Builder()
                    .conversionId(conversionId)
                    .success(true)
                    .outputPath(params.getOutputPath())
                    .outputSize(outputSize)
                    .build();

        } catch (Exception e) {
            log.error("Java GGUF conversion failed", e);
            return new ConversionResult.Builder()
                    .conversionId(conversionId)
                    .success(false)
                    .errorMessage(e.getMessage())
                    .build();
        } finally {
            activeConversions.remove(conversionId);
        }
    }

    public Uni<ConversionResult> convertAsync(GGUFConversionParams params) {
        return Uni.createFrom().item(() -> convert(params, null))
                .runSubscriptionOn(Infrastructure.getDefaultExecutor());
    }
}
