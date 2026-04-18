package tech.kayys.gollek.quantizer.gptq;

import java.nio.file.Path;

/**
 * Result of a quantization operation.
 */
public record QuantizationResult(
        int inputTensors,
        int outputTensors,
        long inputSizeBytes,
        long outputSizeBytes,
        long elapsedMs,
        GPTQConfig config,
        Path outputPath) {
    
    public double compressionRatio() {
        return outputSizeBytes > 0 ? (double) inputSizeBytes / outputSizeBytes : 0;
    }

    public double throughputMBps() {
        return elapsedMs > 0 ? (outputSizeBytes / 1e6) / (elapsedMs / 1000.0) : 0;
    }

    @Override
    public String toString() {
        return "QuantizationResult{inputTensors=%d, outputTensors=%d, inputSize=%.2f MB, outputSize=%.2f MB, compression=%.2fx, elapsed=%dms, throughput=%.1f MB/s}"
                .formatted(inputTensors, outputTensors, 
                        inputSizeBytes / 1e6, outputSizeBytes / 1e6,
                        compressionRatio(), elapsedMs, throughputMBps());
    }
}
