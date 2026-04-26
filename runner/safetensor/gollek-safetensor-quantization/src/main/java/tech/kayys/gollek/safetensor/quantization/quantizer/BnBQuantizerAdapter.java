package tech.kayys.gollek.safetensor.quantization.quantizer;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.quantizer.turboquant.BnBDequantizer;
import org.jboss.logging.Logger;

/**
 * Adapter for BitsAndBytes (BnB) quantization formats (NF4, INT8).
 */
public class BnBQuantizerAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(BnBQuantizerAdapter.class);
    private final BnBDequantizer engine = new BnBDequantizer();

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        if (tensor == null) throw new IllegalArgumentException("Tensor cannot be null");
        
        int numElements = (int) tensor.numel();
        int blockSize = 64; // NF4 default
        int numBlocks = (numElements + blockSize - 1) / blockSize;
        
        float[] weights = tensor.toFloatArray();
        byte[] packedOut = new byte[(numElements + 1) / 2];
        float[] absmaxOut = new float[numBlocks];
        
        // Execute the quantization side of BnB engine
        engine.quantizeNF4(weights, packedOut, absmaxOut);
        
        // Create quantized AccelTensor with metadata
        AccelTensor quantized = AccelTensor.fromByteArray(packedOut, tensor.shape());
        
        // Attach scales (absmax) as a secondary MemorySegment
        java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofAuto();
        java.lang.foreign.MemorySegment scaleSeg = arena.allocateFrom(
            java.lang.foreign.ValueLayout.JAVA_FLOAT, absmaxOut);
            
        quantized.withQuantization(
            AccelTensor.QuantType.INT4, 
            scaleSeg, 
            null, // no zeros for NF4
            blockSize
        );
        
        log.debugf("BnB: quantized %d elements to NF4 (blockSize=%d)", numElements, blockSize);
        return quantized;
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        log.debugf("BnB: dequantizing NF4/INT8 tensor %s", java.util.Arrays.toString(quantizedTensor.shape()));
        
        // Extract raw data and metadata from AccelTensor
        java.lang.foreign.MemorySegment dataSeg = quantizedTensor.dataSegment();
        java.lang.foreign.MemorySegment scalesSeg = quantizedTensor.scales();
        
        if (scalesSeg == null) {
            log.warn("BnB dequantization requested but no scales found on tensor");
            return quantizedTensor;
        }

        int numElements = (int) quantizedTensor.numel();
        float[] output = new float[numElements];
        
        // Map MemorySegment to byte array (copy for now, optimized engine can use segments)
        byte[] packedBytes = dataSeg.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
        float[] absmax = scalesSeg.toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);

        // Run the optimized SIMD dequantization from core
        engine.dequantNF4(packedBytes, absmax, numElements, output);
        
        return AccelTensor.fromFloatArray(output, quantizedTensor.shape());
    }

    @Override
    public String getName() {
        return "BitsAndBytes";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return config.getStrategy() == tech.kayys.gollek.safetensor.quantization.QuantizationEngine.QuantStrategy.INT8 
            || config.getBits() == 4; // NF4
    }
}
