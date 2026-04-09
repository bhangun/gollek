package tech.kayys.gollek.sdk.cnn;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.sdk.cnn.layers.Conv1d;
import tech.kayys.gollek.sdk.cnn.layers.Conv3d;
import tech.kayys.gollek.sdk.cnn.layers.ConvTranspose2d;
import tech.kayys.gollek.sdk.cnn.layers.Upsample;

/**
 * Comprehensive tests for CNN layers.
 */
public class CNNLayersTest {

    @Test
    public void testConv1dForward() {
        Conv1d conv = new Conv1d(3, 16, 3, 1, 1);
        GradTensor input = GradTensor.randn(2, 3, 100);
        GradTensor output = conv.forward(input);
        
        assertNotNull(output);
        assertEquals(4, output.shape().length);
        assertEquals(2, output.shape()[0]);  // batch size
        assertEquals(16, output.shape()[1]); // output channels
        assertEquals(100, output.shape()[2]); // sequence length (with padding)
    }

    @Test
    public void testConv1dOutputSize() {
        Conv1d conv = new Conv1d(3, 32, 5, 2, 2);
        GradTensor input = GradTensor.randn(4, 3, 50);
        GradTensor output = conv.forward(input);
        
        assertEquals(4, output.shape()[0]);
        assertEquals(32, output.shape()[1]);
        // Output length: (50 + 2*2 - 5) / 2 + 1 = 25
        assertEquals(25, output.shape()[2]);
    }

    @Test
    public void testConv1dGradients() {
        Conv1d conv = new Conv1d(3, 16, 3, 1, 1);
        GradTensor input = GradTensor.randn(2, 3, 100);
        input.requiresGrad(true);
        
        GradTensor output = conv.forward(input);
        assertNotNull(output);
        
        GradTensor weight = conv.getWeight();
        assertNotNull(weight);
        assertTrue(weight.requiresGrad());
        
        GradTensor bias = conv.getBias();
        assertNotNull(bias);
        assertTrue(bias.requiresGrad());
    }

    @Test
    public void testConv3dForward() {
        Conv3d conv = new Conv3d(1, 32, 3, 1, 1);
        GradTensor input = GradTensor.randn(2, 1, 10, 28, 28);
        GradTensor output = conv.forward(input);
        
        assertNotNull(output);
        assertEquals(5, output.shape().length);
        assertEquals(2, output.shape()[0]);   // batch size
        assertEquals(32, output.shape()[1]);  // output channels
        assertEquals(10, output.shape()[2]);  // depth
        assertEquals(28, output.shape()[3]);  // height
        assertEquals(28, output.shape()[4]);  // width
    }

    @Test
    public void testConv3dOutputSize() {
        Conv3d conv = new Conv3d(3, 64, 3, 2, 1);
        GradTensor input = GradTensor.randn(2, 3, 16, 32, 32);
        GradTensor output = conv.forward(input);
        
        assertEquals(2, output.shape()[0]);
        assertEquals(64, output.shape()[1]);
        // Output: (16 + 2*1 - 3) / 2 + 1 = 8
        assertEquals(8, output.shape()[2]);
        // Output: (32 + 2*1 - 3) / 2 + 1 = 16
        assertEquals(16, output.shape()[3]);
        assertEquals(16, output.shape()[4]);
    }

    @Test
    public void testConvTranspose2dForward() {
        ConvTranspose2d transpose = new ConvTranspose2d(64, 32, 4, 2, 1);
        GradTensor input = GradTensor.randn(2, 64, 16, 16);
        GradTensor output = transpose.forward(input);
        
        assertNotNull(output);
        assertEquals(4, output.shape().length);
        assertEquals(2, output.shape()[0]);   // batch size
        assertEquals(32, output.shape()[1]);  // output channels
        // Output: (16 - 1) * 2 - 2*1 + 4 + 0 = 32
        assertEquals(32, output.shape()[2]);
        assertEquals(32, output.shape()[3]);
    }

    @Test
    public void testConvTranspose2dWithOutputPadding() {
        ConvTranspose2d transpose = new ConvTranspose2d(32, 16, 3, 2, 1, 1, 1, true);
        GradTensor input = GradTensor.randn(1, 32, 8, 8);
        GradTensor output = transpose.forward(input);
        
        assertEquals(1, output.shape()[0]);
        assertEquals(16, output.shape()[1]);
        // Output: (8 - 1) * 2 - 2*1 + 3 + 1 = 17
        assertEquals(17, output.shape()[2]);
        assertEquals(17, output.shape()[3]);
    }

    @Test
    public void testUpsampleNearest2d() {
        Upsample up = new Upsample(2.0, "nearest");
        GradTensor input = GradTensor.randn(2, 3, 28, 28);
        GradTensor output = up.forward(input);
        
        assertNotNull(output);
        assertEquals(4, output.shape().length);
        assertEquals(2, output.shape()[0]);   // batch size unchanged
        assertEquals(3, output.shape()[1]);   // channels unchanged
        assertEquals(56, output.shape()[2]);  // height upsampled
        assertEquals(56, output.shape()[3]);  // width upsampled
    }

    @Test
    public void testUpsampleBilinear2d() {
        Upsample up = new Upsample(2.5, "bilinear");
        GradTensor input = GradTensor.randn(1, 3, 16, 16);
        GradTensor output = up.forward(input);
        
        assertEquals(1, output.shape()[0]);
        assertEquals(3, output.shape()[1]);
        assertEquals(40, output.shape()[2]);  // 16 * 2.5 = 40
        assertEquals(40, output.shape()[3]);
    }

    @Test
    public void testUpsample3dNearest() {
        Upsample up = new Upsample(2.0, "nearest");
        GradTensor input = GradTensor.randn(1, 8, 4, 8, 8);
        GradTensor output = up.forward(input);
        
        assertEquals(1, output.shape()[0]);
        assertEquals(8, output.shape()[1]);
        assertEquals(8, output.shape()[2]);   // depth upsampled
        assertEquals(16, output.shape()[3]);  // height upsampled
        assertEquals(16, output.shape()[4]);  // width upsampled
    }

    @Test
    public void testUpsample3dTrilinear() {
        Upsample up = new Upsample(2.0, "trilinear");
        GradTensor input = GradTensor.randn(2, 16, 4, 8, 8);
        GradTensor output = up.forward(input);
        
        assertEquals(2, output.shape()[0]);
        assertEquals(16, output.shape()[1]);
        assertEquals(8, output.shape()[2]);   // 4 * 2
        assertEquals(16, output.shape()[3]);  // 8 * 2
        assertEquals(16, output.shape()[4]);  // 8 * 2
    }

    @Test
    public void testUpsampleInvalidScaleFactor() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Upsample(0.5, "nearest");
        });
    }

    @Test
    public void testUpsampleInvalidMode() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Upsample(2.0, "cubic");
        });
    }

    @Test
    public void testConv1dChaining() {
        Conv1d conv1 = new Conv1d(3, 16, 3, 1, 1);
        Conv1d conv2 = new Conv1d(16, 32, 3, 1, 1);
        
        GradTensor input = GradTensor.randn(2, 3, 100);
        GradTensor hidden = conv1.forward(input);
        GradTensor output = conv2.forward(hidden);
        
        assertEquals(2, output.shape()[0]);
        assertEquals(32, output.shape()[1]);
        assertEquals(100, output.shape()[2]);
    }

    @Test
    public void testConv3dWithStride() {
        Conv3d conv = new Conv3d(1, 32, 3, 2, 1);
        GradTensor input = GradTensor.randn(1, 1, 32, 64, 64);
        GradTensor output = conv.forward(input);
        
        assertEquals(1, output.shape()[0]);
        assertEquals(32, output.shape()[1]);
        // (32 + 2 - 3) / 2 + 1 = 16
        assertEquals(16, output.shape()[2]);
        assertEquals(32, output.shape()[3]);
        assertEquals(32, output.shape()[4]);
    }

    @Test
    public void testConvTranspose2dChaining() {
        ConvTranspose2d t1 = new ConvTranspose2d(64, 32, 4, 2, 1);
        ConvTranspose2d t2 = new ConvTranspose2d(32, 16, 4, 2, 1);
        
        GradTensor input = GradTensor.randn(1, 64, 8, 8);
        GradTensor hidden = t1.forward(input);
        GradTensor output = t2.forward(hidden);
        
        assertEquals(1, output.shape()[0]);
        assertEquals(16, output.shape()[1]);
        // First transpose: (8-1)*2 - 2 + 4 = 16
        // Second transpose: (16-1)*2 - 2 + 4 = 32
        assertEquals(32, output.shape()[2]);
        assertEquals(32, output.shape()[3]);
    }

    @Test
    public void testUpsampleVerySmall() {
        Upsample up = new Upsample(3.0, "nearest");
        GradTensor input = GradTensor.randn(1, 1, 2, 2);
        GradTensor output = up.forward(input);
        
        assertEquals(1, output.shape()[0]);
        assertEquals(1, output.shape()[1]);
        assertEquals(6, output.shape()[2]);
        assertEquals(6, output.shape()[3]);
    }

    @Test
    public void testConv1dToString() {
        Conv1d conv = new Conv1d(3, 16, 5, 2, 1, 1, true);
        String str = conv.toString();
        
        assertTrue(str.contains("Conv1d"));
        assertTrue(str.contains("3"));   // in_channels
        assertTrue(str.contains("16"));  // out_channels
        assertTrue(str.contains("5"));   // kernel_size
    }

    @Test
    public void testConv3dToString() {
        Conv3d conv = new Conv3d(1, 32, 3, 1, 1);
        String str = conv.toString();
        
        assertTrue(str.contains("Conv3d"));
        assertTrue(str.contains("1"));   // in_channels
        assertTrue(str.contains("32"));  // out_channels
    }

    @Test
    public void testConvTranspose2dToString() {
        ConvTranspose2d transpose = new ConvTranspose2d(64, 32, 4, 2, 1);
        String str = transpose.toString();
        
        assertTrue(str.contains("ConvTranspose2d"));
        assertTrue(str.contains("64"));  // in_channels
        assertTrue(str.contains("32"));  // out_channels
    }

    @Test
    public void testUpsampleToString() {
        Upsample up = new Upsample(2.0, "bilinear");
        String str = up.toString();
        
        assertTrue(str.contains("Upsample"));
        assertTrue(str.contains("2"));         // scale_factor
        assertTrue(str.contains("bilinear")); // mode
    }
}
