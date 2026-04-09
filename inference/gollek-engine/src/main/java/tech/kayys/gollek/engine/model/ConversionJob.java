package tech.kayys.gollek.engine.model;

/**
 * Conversion job record.
 */
public record ConversionJob(
        String jobId,
        String modelId,
        String sourceFormat,
        String targetFormat,
        String status) {
}
