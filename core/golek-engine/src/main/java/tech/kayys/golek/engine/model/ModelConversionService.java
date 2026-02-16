package tech.kayys.golek.engine.model;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.golek.engine.model.ModelRegistryService.ConversionJob;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Service for converting models between different formats.
 *
 * <p>
 * Supported conversions:
 * <ul>
 * <li>PyTorch → ONNX → LiteRT</li>
 * <li>TensorFlow → LiteRT</li>
 * <li>ONNX → LiteRT</li>
 * <li>TensorFlow → ONNX</li>
 * </ul>
 *
 * <p>
 * Conversion happens asynchronously in background workers.
 *
 * @author Bhangun
 * @since 1.0.0
 */
@ApplicationScoped
public class ModelConversionService {

    private static final Logger log = LoggerFactory.getLogger(ModelConversionService.class);

    private final Map<String, ConversionJob> jobs = new ConcurrentHashMap<>();

    /**
     * Submit model conversion job.
     */
    public Uni<ConversionJob> submitConversion(
            String requestId,
            String modelId,
            String targetFormat) {

        return Uni.createFrom().item(() -> {
            String jobId = UUID.randomUUID().toString();

            ConversionJob job = new ConversionJob(
                    jobId,
                    modelId,
                    "unknown", // Source format detected from model
                    targetFormat,
                    "PENDING");

            jobs.put(jobId, job);

            log.info("Conversion job submitted: jobId={}, modelId={}, targetFormat={}",
                    jobId, modelId, targetFormat);

            // In production, this would trigger actual conversion worker
            // For now, just store the job

            return job;
        });
    }

    /**
     * Get conversion job status.
     */
    public ConversionJob getJobStatus(String jobId) {
        return jobs.get(jobId);
    }
}
