package tech.kayys.gollek.safetensor.quantization.quantizer;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.quantizer.turboquant.TurboQuantEngine;
import tech.kayys.gollek.quantizer.turboquant.TurboQuantConfig;
import org.jboss.logging.Logger;

import java.util.Arrays;

/**
 * Adapter that bridges the optimized TurboQuant core engine with the Safetensor runner.
 */
public class TurboQuantAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(TurboQuantAdapter.class);

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        if (tensor == null) throw new IllegalArgumentException("Tensor cannot be null");
        
        log.infof("TurboQuant: quantizing tensor %s (bits=%d)", 
                 Arrays.toString(tensor.shape()), config.getBits());

        // Map runner QuantConfig to TurboQuantConfig
        int dim = (int) (tensor.numel());
        TurboQuantConfig tqConfig = new TurboQuantConfig(
            config.getBits(),
            dim,
            TurboQuantConfig.Variant.MSE,
            TurboQuantConfig.RotationStrategy.HADAMARD,
            false, // splitOutliers
            0,     // outlierChannels
            42L    // seed
        );

        TurboQuantEngine engine = new TurboQuantEngine(tqConfig);
        float[] data = tensor.toFloatArray();
        int[] indices = new int[dim];
        
        // Execute Algorithms 1: TurboQuant_mse
        engine.quantizeMse(data, indices);
        
        // Wrap indices in AccelTensor.
        // We pack indices into a byte[] if bits <= 8
        byte[] packed = new byte[dim];
        for (int i = 0; i < dim; i++) {
            packed[i] = (byte) indices[i];
        }
        
        AccelTensor quantized = AccelTensor.fromByteArray(packed, tensor.shape());
        quantized.withQuantization(AccelTensor.QuantType.INT8, null, null, -1);
        
        return quantized;
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        log.infof("TurboQuant: dequantizing tensor %s", Arrays.toString(quantizedTensor.shape()));
        
        int dim = (int) quantizedTensor.numel();
        byte[] packed = quantizedTensor.dataSegment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
        int[] indices = new int[dim];
        for (int i = 0; i < dim; i++) {
            indices[i] = packed[i] & 0xFF;
        }
        
        TurboQuantConfig tqConfig = new TurboQuantConfig(
            config.getBits(),
            dim,
            TurboQuantConfig.Variant.MSE,
            TurboQuantConfig.RotationStrategy.HADAMARD,
            false,
            0,
            42L
        );
        
        TurboQuantEngine engine = new TurboQuantEngine(tqConfig);
        float[] output = new float[dim];
        engine.dequantizeMse(indices, output);
        
        return AccelTensor.fromFloatArray(output, quantizedTensor.shape());
    }

    @Override
    public String getName() {
        return "TurboQuant";
    }

    @Override
    public boolean supports(QuantConfig config) {
        // TurboQuant specialized in 2-4 bit vector quantization
        return config.getBits() <= 4;
    }
}
