package tech.kayys.gollek.safetensor.quantization.quantizer;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.quantizer.awq.AWQDequantizer;
import tech.kayys.gollek.quantizer.awq.AWQConfig;
import org.jboss.logging.Logger;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Adapter for Activation-aware Weight Quantization (AWQ).
 */
public class AWQQuantizerAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(AWQQuantizerAdapter.class);

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        if (tensor == null) throw new IllegalArgumentException("Tensor cannot be null");
        
        int inF = (int) tensor.size(0);
        int outF = (int) tensor.size(1);
        int bits = config.getBits();
        int groupSize = 128; // Default AWQ group size
        
        log.infof("AWQ: quantizing %dx%d tensor to %d bits (groupSize=%d)", inF, outF, bits, groupSize);

        // Map to AWQConfig
        AWQConfig awqConfig = new AWQConfig(bits, groupSize, AWQConfig.KernelFormat.GEMM, true, "float32", false, 128, 2048, true, null);
        float[] weights = tensor.toFloatArray();
        
        int numGroups = (inF + groupSize - 1) / groupSize;
        int packFactor = 32 / bits;
        
        int[] qweight = new int[(inF / packFactor) * outF];
        float[] scales = new float[numGroups * outF];
        float[] zeros = new float[numGroups * outF];
        
        // Simple Min-Max quantization (Note: Actual AWQ requires calibration for scales)
        for (int g = 0; g < numGroups; g++) {
            int iStart = g * groupSize;
            int iEnd = Math.min(iStart + groupSize, inF);
            
            for (int j = 0; j < outF; j++) {
                float min = Float.MAX_VALUE;
                float max = -Float.MAX_VALUE;
                
                for (int i = iStart; i < iEnd; i++) {
                    float v = weights[i * outF + j];
                    min = Math.min(min, v);
                    max = Math.max(max, v);
                }
                
                float scale = (max - min) / ((1 << bits) - 1);
                float zero = -min / (scale == 0 ? 1 : scale);
                
                scales[g * outF + j] = scale;
                zeros[g * outF + j] = Math.round(zero);
                
                float invScale = scale == 0 ? 0 : 1.0f / scale;
                
                for (int i = iStart; i < iEnd; i++) {
                    float v = weights[i * outF + j];
                    int q = Math.max(0, Math.min((1 << bits) - 1, Math.round((v - min) * invScale)));
                    
                    int pi = i / packFactor;
                    int b = i % packFactor;
                    qweight[pi * outF + j] |= (q << (b * bits));
                }
            }
        }

        // Store in AccelTensor
        byte[] packedBytes = new byte[qweight.length * 4];
        MemorySegment.ofArray(packedBytes).copyFrom(MemorySegment.ofArray(qweight));
        
        AccelTensor quantized = AccelTensor.fromByteArray(packedBytes, tensor.shape());
        
        Arena arena = Arena.ofShared();
        MemorySegment scaleSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, scales);
        MemorySegment zeroSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, zeros);
        
        quantized.withQuantization(
            bits == 4 ? AccelTensor.QuantType.INT4 : AccelTensor.QuantType.INT8,
            scaleSeg,
            zeroSeg,
            groupSize
        );
        
        return quantized;
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        int inF = (int) quantizedTensor.size(0);
        int outF = (int) quantizedTensor.size(1);
        int bits = config.getBits();
        int groupSize = quantizedTensor.groupSize();
        
        AWQConfig awqConfig = new AWQConfig(bits, groupSize, AWQConfig.KernelFormat.GEMM, true, "float32", false, 128, 2048, true, null);
        AWQDequantizer dequantizer = new AWQDequantizer(awqConfig);
        
        MemorySegment dataSeg = quantizedTensor.dataSegment();
        int[] qweightInts = new int[(int) (dataSeg.byteSize() / 4)];
        for (int i = 0; i < qweightInts.length; i++) {
            qweightInts[i] = dataSeg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), (long) i * 4);
        }
        
        float[] scales = quantizedTensor.scales().toArray(ValueLayout.JAVA_FLOAT);
        float[] zeros = quantizedTensor.zeros().toArray(ValueLayout.JAVA_FLOAT);
        
        // AutoAWQ uses (q - zero) * scale, core expects FP16 scales and packed zeros.
        // We'll use a simplified dequantization loop if the core dequantizer doesn't match our storage exactly.
        // But the AWQDequantizer.dequantize is SIMD optimized.
        
        float[] output = new float[inF * outF];
        
        // We need to provide what AWQDequantizer expects: 
        // qzerosInts is packed, scalesShorts is FP16.
        // For simplicity in this adapter, we'll implement the loop directly here using SIMD if possible,
        // or just use the core one by converting back.
        
        int numGroups = (inF + groupSize - 1) / groupSize;
        short[] scalesShorts = new short[scales.length];
        for (int i = 0; i < scales.length; i++) scalesShorts[i] = floatToFp16(scales[i]);
        
        // implicit zero handling or convert zeros back to packed
        int[] qzerosInts = packZeros(zeros, numGroups, outF, bits);
        
        dequantizer.dequantize(qweightInts, qzerosInts, scalesShorts, inF, outF, output);
        
        return AccelTensor.fromFloatArray(output, quantizedTensor.shape());
    }

    private short floatToFp16(float v) {
        return (short) Float.floatToFloat16(v);
    }

    private int[] packZeros(float[] zeros, int numGroups, int outF, int bits) {
        int packFactor = 32 / bits;
        int[] packed = new int[numGroups * (outF / packFactor)];
        for (int g = 0; g < numGroups; g++) {
            for (int j = 0; j < outF; j++) {
                int qz = (int) zeros[g * outF + j] - 1; // AutoAWQ +1 offset
                int pj = j / packFactor;
                int b = j % packFactor;
                packed[g * (outF / packFactor) + pj] |= (qz << (b * bits));
            }
        }
        return packed;
    }

    @Override
    public String getName() {
        return "AWQ";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return config.getStrategy() == tech.kayys.gollek.safetensor.quantization.QuantizationEngine.QuantStrategy.AWQ;
    }
}
