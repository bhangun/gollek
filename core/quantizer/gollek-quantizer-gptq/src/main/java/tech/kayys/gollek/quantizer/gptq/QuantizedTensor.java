package tech.kayys.gollek.quantizer.gptq;

/**
 * Quantized tensor metadata.
 */
public record QuantizedTensor(
        String name,
        long[] shape,
        String dtype,
        long sizeBytes) {
}
