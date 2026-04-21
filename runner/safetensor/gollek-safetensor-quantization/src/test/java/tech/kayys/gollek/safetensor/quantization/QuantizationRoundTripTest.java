package tech.kayys.gollek.safetensor.quantization;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.quantization.quantizer.*;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;

/**
 * Verifies rounding and numerical stability for all quantization adapters.
 * Round-trip: FP32 -> Quantized -> Dequantized -> FP32' (MSE verification)
 */
public class QuantizationRoundTripTest {

    @Test
    @DisplayName("TurboQuant Round-Trip (MSE 4-bit)")
    void testTurboQuantRoundTrip() {
        float[] original = { 0.1f, -0.5f, 0.8f, -0.2f, 1.0f, -1.0f, 0.0f, 0.5f };
        long[] shape = { 8 };
        AccelTensor tensor = AccelTensor.fromFloatArray(original, shape);

        TurboQuantAdapter adapter = new TurboQuantAdapter();
        QuantConfig config = QuantConfig.int4Gptq(); // Using 4-bit for TurboQuant test
        
        AccelTensor quantized = adapter.quantizeTensor(tensor, config);
        AccelTensor dequantized = adapter.dequantizeTensor(quantized, config);

        float[] result = dequantized.toFloatArray();
        System.out.println("TurboQuant Round-trip result: " + Arrays.toString(result));

        // Verify MSE is within bounds (TurboQuant 4-bit is very accurate)
        double mse = calculateMSE(original, result);
        assertTrue(mse < 0.05, "TurboQuant MSE too high: " + mse);
    }

    @Test
    @DisplayName("BnB NF4 Round-Trip")
    void testBnBRoundTrip() {
        float[] original = { 0.1f, -0.5f, 0.8f, -0.2f, 1.0f, -1.0f, 0.0f, 0.5f };
        long[] shape = { 8 };
        AccelTensor tensor = AccelTensor.fromFloatArray(original, shape);

        BnBQuantizerAdapter adapter = new BnBQuantizerAdapter();
        QuantConfig config = new QuantConfig(); // Defaults to 4-bit NF4
        
        AccelTensor quantized = adapter.quantizeTensor(tensor, config);
        AccelTensor dequantized = adapter.dequantizeTensor(quantized, config);

        float[] result = dequantized.toFloatArray();
        System.out.println("BnB NF4 Round-trip result: " + Arrays.toString(result));

        // NF4 is highly optimized for distributions, MSE should be low
        double mse = calculateMSE(original, result);
        assertTrue(mse < 0.1, "BnB NF4 MSE too high: " + mse);
    }
    
    @Test
    @DisplayName("AWQ Round-Trip (Naive)")
    void testAWQRoundTrip() {
        // AWQ requires larger groups for the adapter logic to work correctly (groupSize=128)
        int size = 128;
        float[] original = new float[size * 2];
        for(int i=0; i<original.length; i++) original[i] = (float)Math.sin(i * 0.1);
        
        long[] shape = { 2, size };
        AccelTensor tensor = AccelTensor.fromFloatArray(original, shape);

        AWQQuantizerAdapter adapter = new AWQQuantizerAdapter();
        QuantConfig config = QuantConfig.awq(4, 128);
        
        AccelTensor quantized = adapter.quantizeTensor(tensor, config);
        AccelTensor dequantized = adapter.dequantizeTensor(quantized, config);

        float[] result = dequantized.toFloatArray();
        
        double mse = calculateMSE(original, result);
        assertTrue(mse < 0.1, "AWQ MSE too high: " + mse);
    }

    @Test
    @DisplayName("AutoRound Round-Trip")
    void testAutoRoundRoundTrip() {
        int outF = 16;
        int inF = 128; // groupSize=128
        float[] original = new float[outF * inF];
        for(int i=0; i<original.length; i++) original[i] = (float)Math.cos(i * 0.05);

        long[] shape = { outF, inF };
        AccelTensor tensor = AccelTensor.fromFloatArray(original, shape);

        AutoRoundQuantizerAdapter adapter = new AutoRoundQuantizerAdapter();
        QuantConfig config = QuantConfig.autoRound();
        
        AccelTensor quantized = adapter.quantizeTensor(tensor, config);
        AccelTensor dequantized = adapter.dequantizeTensor(quantized, config);

        float[] result = dequantized.toFloatArray();
        
        double mse = calculateMSE(original, result);
        assertTrue(mse < 0.1, "AutoRound MSE too high: " + mse);
    }

    private double calculateMSE(float[] expected, float[] actual) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            double diff = expected[i] - actual[i];
            sum += diff * diff;
        }
        return sum / expected.length;
    }
}
