package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import tech.kayys.gollek.ml.autograd.GradTensor;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for neural network architectures.
 * Tests combining multiple modules in realistic scenarios.
 */
public class IntegrationTest {

    private NNModule model;

    @BeforeEach
    public void setUp() {
        // Build a simple 3-layer feedforward network
        model = new Sequential(
            new Linear(10, 64),
            new ReLU(),
            new Dropout(0.5f),
            new Linear(64, 32),
            new ReLU(),
            new Dropout(0.5f),
            new Linear(32, 2)
        );
    }

    /**
     * Test forward pass through entire network.
     */
    @Test
    public void testFeedforwardForward() {
        float[] input = new float[10];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) i / 10;
        }

        // Should not throw
        var output = model.forward(GradTensor.of(input, new long[]{1, 10}));
        assertNotNull(output);
        assertEquals(2, output.shape()[1]);  // Output should be 2 classes
    }

    /**
     * Test residual block integration.
     */
    @Test
    public void testResidualBlockIntegration() {
        var residualModel = new Sequential(
            new Linear(32, 32),
            new ResidualBlock(
                new Sequential(
                    new Linear(32, 32),
                    new ReLU(),
                    new Linear(32, 32)
                )
            ),
            new ReLU(),
            new Linear(32, 10)
        );

        float[] input = new float[32];
        var tensor = GradTensor.of(input, new long[]{1, 32});
        var output = residualModel.forward(tensor);

        assertNotNull(output);
        assertEquals(10, output.shape()[1]);
    }

    /**
     * Test multi-head attention in isolation.
     */
    @Test
    public void testMultiHeadAttention() {
        int embedDim = 64;
        int numHeads = 4;
        int seqLen = 8;
        int batchSize = 2;

        var attention = new MultiHeadAttention(embedDim, numHeads);

        // Create sequence tensor: [batch, seq_len, embed_dim]
        long[] shape = {batchSize, seqLen, embedDim};
        float[] data = new float[(int)(batchSize * seqLen * embedDim)];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.sin(i * 0.1);
        }
        var input = GradTensor.of(data, shape);

        var output = attention.forward(input);
        assertNotNull(output);
        assertArrayEquals(shape, output.shape());
    }

    /**
     * Test transformer layer.
     */
    @Test
    public void testTransformerEncoderLayer() {
        int embedDim = 64;
        int ffDim = 128;
        int numHeads = 4;

        var layer = new TransformerEncoderLayer(embedDim, numHeads, ffDim, 0.1f);

        long[] shape = {2, 4, embedDim};  // [batch, seq, embed]
        float[] data = new float[(int)(2L * 4 * embedDim)];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) i / 1000;
        }

        var input = GradTensor.of(data, shape);
        var output = layer.forward(input);

        assertNotNull(output);
        assertArrayEquals(shape, output.shape());
    }

    /**
     * Test batch normalization in a model.
     */
    @Test
    public void testBatchNormalizationIntegration() {
        var model = new Sequential(
            new Linear(10, 32),
            new BatchNorm1d(32),
            new ReLU(),
            new Linear(32, 10)
        );

        float[] input = new float[10];
        var tensor = GradTensor.of(input, new long[]{2, 10});  // batch size 2
        var output = model.forward(tensor);

        assertNotNull(output);
        assertEquals(10, output.shape()[1]);
    }

    /**
     * Test learning rate scheduling.
     */
    @Test
    public void testLRScheduler() {
        var params = model.parameters();
        var optimizer = new tech.kayys.gollek.ml.nn.optim.Adam(params, 0.001f);
        var scheduler = new tech.kayys.gollek.ml.nn.optim.StepLR(optimizer, 5, 0.5f);

        float initialLr = optimizer.learningRate();
        for (int step = 0; step < 10; step++) {
            scheduler.step();
        }

        // After 10 steps with stepSize=5: should have decayed twice
        float finalLr = optimizer.learningRate();
        assertTrue(finalLr < initialLr);
    }

    /**
     * Test early stopping.
     */
    @Test
    public void testEarlyStopping() {
        var earlyStopping = new EarlyStopping(3, true, 0, "min");

        assertFalse(earlyStopping.check(1.0f));  // First value
        assertFalse(earlyStopping.check(0.9f));  // Improvement
        assertFalse(earlyStopping.check(0.95f)); // No improvement (counter=1)
        assertFalse(earlyStopping.check(0.96f)); // No improvement (counter=2)
        assertTrue(earlyStopping.check(0.97f));  // No improvement (counter=3) -> stop

        assertEquals(0.9f, earlyStopping.getBestValue());
    }

    /**
     * Test accuracy metric.
     */
    @Test
    public void testAccuracyMetric() {
        var accuracy = new tech.kayys.gollek.ml.nn.metrics.Accuracy();

        float[] predictions = {0f, 1f, 1f, 0f};
        float[] targets = {0f, 1f, 1f, 0f};

        accuracy.update(predictions, targets);
        assertEquals(1.0f, accuracy.compute());  // 100% accuracy

        accuracy.reset();
        float[] predictions2 = {0f, 1f, 0f, 0f};
        float[] targets2 = {1f, 1f, 1f, 0f};
        accuracy.update(predictions2, targets2);
        assertEquals(0.5f, accuracy.compute());  // 50% accuracy
    }
}
