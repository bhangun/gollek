package tech.kayys.gollek.cli.audit;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;

/**
 * Structured audit record capturing all profiling data for a quantization operation.
 * <p>
 * Used for benchmarking, analysis, and troubleshooting. Persisted as JSON and CSV
 * to {@code ~/.gollek/audit/}.
 */
public record QuantAuditRecord(
        /** Unique audit ID */
        String id,

        /** Timestamp of the quantization operation */
        Instant timestamp,

        /** Source model identifier (HuggingFace ID or local path) */
        String modelId,

        /** Quantized model output path */
        String outputPath,

        /** Quantization strategy used (bnb, turbo, awq, gptq, autoround) */
        String strategy,

        /** Bit width (2, 3, 4, 8) */
        int bits,

        /** Group size for block-based quantization */
        int groupSize,

        /** Hardware accelerator description (e.g., "Metal (Apple M4)", "CUDA", "CPU") */
        String accelerator,

        /** Java version string */
        String javaVersion,

        /** Whether SIMD Vector API was used */
        boolean simdEnabled,

        /** Original model size in bytes */
        long originalSizeBytes,

        /** Quantized model size in bytes */
        long quantizedSizeBytes,

        /** Compression ratio (original / quantized) */
        double compressionRatio,

        /** Total number of tensors processed */
        int tensorCount,

        /** Total parameter count */
        long parameterCount,

        /** Wall-clock duration of quantization */
        Duration quantizationDuration,

        /** Quantization throughput (parameters per second) */
        double paramsPerSecond,

        /** Inference speed if measured (tokens/s), -1 if not measured */
        double tokensPerSecond,

        /** Mode: "offline" (gollek quantize) or "jit" (--quantize on run/chat) */
        String mode,

        /** Whether this model was registered in the gollek manifest */
        boolean registeredInManifest,

        /** Extensible metadata (e.g., MSE, calibration info) */
        Map<String, Object> extra
) {

    /** Create a builder pre-populated with defaults. */
    public static Builder builder() {
        return new Builder();
    }

    /** CSV header line for export. */
    public static String csvHeader() {
        return "id,timestamp,model_id,output_path,strategy,bits,group_size,accelerator,"
                + "java_version,simd_enabled,original_size_bytes,quantized_size_bytes,"
                + "compression_ratio,tensor_count,parameter_count,duration_ms,"
                + "params_per_second,tokens_per_second,mode,registered_in_manifest";
    }

    /** CSV data line for export. */
    public String toCsvLine() {
        return String.join(",",
                quote(id),
                quote(timestamp.toString()),
                quote(modelId),
                quote(outputPath != null ? outputPath : ""),
                quote(strategy),
                String.valueOf(bits),
                String.valueOf(groupSize),
                quote(accelerator),
                quote(javaVersion),
                String.valueOf(simdEnabled),
                String.valueOf(originalSizeBytes),
                String.valueOf(quantizedSizeBytes),
                String.format("%.4f", compressionRatio),
                String.valueOf(tensorCount),
                String.valueOf(parameterCount),
                String.valueOf(quantizationDuration.toMillis()),
                String.format("%.2f", paramsPerSecond),
                String.format("%.2f", tokensPerSecond),
                quote(mode),
                String.valueOf(registeredInManifest)
        );
    }

    private static String quote(String s) {
        if (s == null) return "\"\"";
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    public static class Builder {
        private String id = UUID.randomUUID().toString();
        private Instant timestamp = Instant.now();
        private String modelId = "";
        private String outputPath;
        private String strategy = "bnb";
        private int bits = 4;
        private int groupSize = 128;
        private String accelerator = "CPU";
        private String javaVersion = System.getProperty("java.version", "unknown");
        private boolean simdEnabled = true;
        private long originalSizeBytes;
        private long quantizedSizeBytes;
        private double compressionRatio;
        private int tensorCount;
        private long parameterCount;
        private Duration quantizationDuration = Duration.ZERO;
        private double paramsPerSecond;
        private double tokensPerSecond = -1;
        private String mode = "offline";
        private boolean registeredInManifest = false;
        private Map<String, Object> extra = Map.of();

        public Builder id(String id) { this.id = id; return this; }
        public Builder timestamp(Instant ts) { this.timestamp = ts; return this; }
        public Builder modelId(String m) { this.modelId = m; return this; }
        public Builder outputPath(String p) { this.outputPath = p; return this; }
        public Builder strategy(String s) { this.strategy = s; return this; }
        public Builder bits(int b) { this.bits = b; return this; }
        public Builder groupSize(int g) { this.groupSize = g; return this; }
        public Builder accelerator(String a) { this.accelerator = a; return this; }
        public Builder javaVersion(String v) { this.javaVersion = v; return this; }
        public Builder simdEnabled(boolean s) { this.simdEnabled = s; return this; }
        public Builder originalSizeBytes(long s) { this.originalSizeBytes = s; return this; }
        public Builder quantizedSizeBytes(long s) { this.quantizedSizeBytes = s; return this; }
        public Builder compressionRatio(double r) { this.compressionRatio = r; return this; }
        public Builder tensorCount(int c) { this.tensorCount = c; return this; }
        public Builder parameterCount(long c) { this.parameterCount = c; return this; }
        public Builder quantizationDuration(Duration d) { this.quantizationDuration = d; return this; }
        public Builder paramsPerSecond(double p) { this.paramsPerSecond = p; return this; }
        public Builder tokensPerSecond(double t) { this.tokensPerSecond = t; return this; }
        public Builder mode(String m) { this.mode = m; return this; }
        public Builder registeredInManifest(boolean r) { this.registeredInManifest = r; return this; }
        public Builder extra(Map<String, Object> e) { this.extra = e; return this; }

        public QuantAuditRecord build() {
            if (compressionRatio <= 0 && quantizedSizeBytes > 0) {
                compressionRatio = (double) originalSizeBytes / quantizedSizeBytes;
            }
            if (paramsPerSecond <= 0 && parameterCount > 0 && quantizationDuration.toMillis() > 0) {
                paramsPerSecond = parameterCount / (quantizationDuration.toMillis() / 1000.0);
            }
            return new QuantAuditRecord(id, timestamp, modelId, outputPath, strategy, bits,
                    groupSize, accelerator, javaVersion, simdEnabled, originalSizeBytes,
                    quantizedSizeBytes, compressionRatio, tensorCount, parameterCount,
                    quantizationDuration, paramsPerSecond, tokensPerSecond, mode,
                    registeredInManifest, extra);
        }
    }
}
