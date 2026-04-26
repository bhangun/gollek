package tech.kayys.gollek.safetensor.quantization.quantizer;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.quantizer.autoround.AutoRoundDequantizer;
import tech.kayys.gollek.quantizer.autoround.AutoRoundConfig;
import org.jboss.logging.Logger;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Adapter for AutoRound quantization.
 */
public class AutoRoundQuantizerAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(AutoRoundQuantizerAdapter.class);

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        if (tensor == null) throw new IllegalArgumentException("Tensor cannot be null");
        
        int inF = (int) tensor.size(1); // AutoRound uses [outF, inF]
        int outF = (int) tensor.size(0);
        int bits = config.getBits();
        int groupSize = 128;
        
        log.infof("AutoRound: quantizing %dx%d tensor to %d bits", outF, inF, bits);
        
        AutoRoundConfig arConfig = new AutoRoundConfig(bits, groupSize, true, AutoRoundConfig.ScaleDtype.FLOAT32, AutoRoundConfig.PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, "exllamav2", null);
        float[] weights = tensor.toFloatArray();
        
        int numGroups = (inF + groupSize - 1) / groupSize;
        int packFactor = 32 / bits;
        
        int[] qweight = new int[(outF / packFactor) * inF];
        float[] scales = new float[outF * numGroups];
        int[] zp = new int[outF * numGroups];
        
        // Naive min-max quantization per group
        for (int j = 0; j < outF; j++) {
            for (int g = 0; g < numGroups; g++) {
                int iStart = g * groupSize;
                int iEnd = Math.min(iStart + groupSize, inF);
                
                float min = Float.MAX_VALUE;
                float max = -Float.MAX_VALUE;
                for (int i = iStart; i < iEnd; i++) {
                    float v = weights[j * inF + i];
                    min = Math.min(min, v);
                    max = Math.max(max, v);
                }
                
                float scale = (max - min) / ((1 << bits) - 1);
                int zero = Math.round(-min / (scale == 0 ? 1 : scale));
                
                scales[j * numGroups + g] = scale;
                zp[j * numGroups + g] = zero;
                
                float invScale = scale == 0 ? 0 : 1.0f / scale;
                for (int i = iStart; i < iEnd; i++) {
                    float v = weights[j * inF + i];
                    int q = Math.max(0, Math.min((1 << bits) - 1, Math.round((v - min) * invScale)));
                    
                    int pr = j / packFactor;
                    int b = j % packFactor;
                    qweight[pr * inF + i] |= (q << (b * bits));
                }
            }
        }

        byte[] packedBytes = new byte[qweight.length * 4];
        MemorySegment.ofArray(packedBytes).copyFrom(MemorySegment.ofArray(qweight));
        
        AccelTensor quantized = AccelTensor.fromByteArray(packedBytes, tensor.shape());
        
        Arena arena = Arena.ofAuto();
        MemorySegment scaleSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, scales);
        MemorySegment zeroSeg = arena.allocateFrom(ValueLayout.JAVA_INT, zp);
        
        quantized.withQuantization(
            AccelTensor.QuantType.INT4,
            scaleSeg,
            zeroSeg,
            groupSize
        );
        
        return quantized;
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        int outF = (int) quantizedTensor.size(0);
        int inF = (int) quantizedTensor.size(1);
        int bits = config.getBits();
        int groupSize = quantizedTensor.groupSize();
        
        AutoRoundConfig arConfig = new AutoRoundConfig(bits, groupSize, true, AutoRoundConfig.ScaleDtype.FLOAT32, AutoRoundConfig.PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, "exllamav2", null);
        AutoRoundDequantizer dequantizer = new AutoRoundDequantizer(arConfig);
        
        MemorySegment dataSeg = quantizedTensor.dataSegment();
        int[] qweightInts = new int[(int) (dataSeg.byteSize() / 4)];
        for (int i = 0; i < qweightInts.length; i++) {
            qweightInts[i] = dataSeg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), (long) i * 4);
        }
        
        float[] scales = quantizedTensor.scales().toArray(ValueLayout.JAVA_FLOAT);
        int[] zp = quantizedTensor.zeros().toArray(ValueLayout.JAVA_INT);
        
        float[] output = new float[outF * inF];
        dequantizer.dequantize(qweightInts, scales, zp, inF, outF, output);
        
        return AccelTensor.fromFloatArray(output, quantizedTensor.shape());
    }

    @Override
    public String getName() {
        return "AutoRound";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return config.getStrategy() == tech.kayys.gollek.safetensor.quantization.QuantizationEngine.QuantStrategy.INT4;
    }
}
