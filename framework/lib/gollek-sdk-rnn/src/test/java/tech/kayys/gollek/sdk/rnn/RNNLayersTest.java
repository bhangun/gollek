package tech.kayys.gollek.sdk.rnn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.sdk.rnn.cells.GRUCell;
import tech.kayys.gollek.sdk.rnn.cells.LSTMCell;
import tech.kayys.gollek.sdk.rnn.cells.LSTMCell.LSTMOutput;
import tech.kayys.gollek.sdk.rnn.layers.Bidirectional;
import tech.kayys.gollek.sdk.rnn.layers.GRULayer;
import tech.kayys.gollek.sdk.rnn.layers.LSTMLayer;

/**
 * Comprehensive tests for RNN layers and cells.
 */
public class RNNLayersTest {

    @Test
    public void testLSTMCellForward() {
        LSTMCell cell = new LSTMCell(100, 128);
        GradTensor input = GradTensor.randn(32, 100);
        GradTensor hidden = cell.initHidden(32);
        GradTensor cellState = cell.initCell(32);

        LSTMOutput output = cell.forward(input, hidden, cellState);

        assertNotNull(output);
        assertNotNull(output.hidden);
        assertNotNull(output.cell);
        assertEquals(2, output.hidden.shape().length);
        assertEquals(32, output.hidden.shape()[0]);
        assertEquals(128, output.hidden.shape()[1]);
    }

    @Test
    public void testLSTMCellParameters() {
        LSTMCell cell = new LSTMCell(100, 128);
        GradTensor[] params = cell.getParameters();

        assertNotNull(params);
        assertEquals(4, params.length);  // weightIh, weightHh, biasIh, biasHh

        // Check weight dimensions: 4*hidden_size for gates
        assertEquals(4 * 128, params[0].shape()[0]);  // weightIh
        assertEquals(100, params[0].shape()[1]);

        assertEquals(4 * 128, params[1].shape()[0]);  // weightHh
        assertEquals(128, params[1].shape()[1]);
    }

    @Test
    public void testLSTMCellInitialization() {
        LSTMCell cell = new LSTMCell(50, 64);
        
        GradTensor hidden = cell.initHidden(16);
        assertEquals(2, hidden.shape().length);
        assertEquals(16, hidden.shape()[0]);
        assertEquals(64, hidden.shape()[1]);

        GradTensor cellState = cell.initCell(16);
        assertEquals(16, cellState.shape()[0]);
        assertEquals(64, cellState.shape()[1]);
    }

    @Test
    public void testGRUCellForward() {
        GRUCell cell = new GRUCell(100, 128);
        GradTensor input = GradTensor.randn(32, 100);
        GradTensor hidden = cell.initHidden(32);

        GradTensor output = cell.forward(input, hidden);

        assertNotNull(output);
        assertEquals(2, output.shape().length);
        assertEquals(32, output.shape()[0]);
        assertEquals(128, output.shape()[1]);
    }

    @Test
    public void testGRUCellParameters() {
        GRUCell cell = new GRUCell(100, 128);
        GradTensor[] params = cell.getParameters();

        assertNotNull(params);
        assertEquals(4, params.length);  // weightIh, weightHh, biasIh, biasHh

        // Check weight dimensions: 3*hidden_size for gates
        assertEquals(3 * 128, params[0].shape()[0]);  // weightIh
        assertEquals(100, params[0].shape()[1]);

        assertEquals(3 * 128, params[1].shape()[0]);  // weightHh
        assertEquals(128, params[1].shape()[1]);
    }

    @Test
    public void testLSTMLayerReturnSequences() {
        LSTMLayer lstm = new LSTMLayer(100, 128, true);
        GradTensor input = GradTensor.randn(32, 50, 100);

        GradTensor output = lstm.forward(input);

        assertNotNull(output);
        assertEquals(3, output.shape().length);
        assertEquals(32, output.shape()[0]);   // batch size
        assertEquals(50, output.shape()[1]);   // sequence length
        assertEquals(128, output.shape()[2]);  // hidden size
    }

    @Test
    public void testLSTMLayerReturnLast() {
        LSTMLayer lstm = new LSTMLayer(100, 128, false);
        GradTensor input = GradTensor.randn(32, 50, 100);

        GradTensor output = lstm.forward(input);

        assertNotNull(output);
        assertEquals(2, output.shape().length);
        assertEquals(32, output.shape()[0]);   // batch size
        assertEquals(128, output.shape()[1]);  // hidden size (only last)
    }

    @Test
    public void testGRULayerReturnSequences() {
        GRULayer gru = new GRULayer(100, 128, true);
        GradTensor input = GradTensor.randn(32, 50, 100);

        GradTensor output = gru.forward(input);

        assertNotNull(output);
        assertEquals(3, output.shape().length);
        assertEquals(32, output.shape()[0]);
        assertEquals(50, output.shape()[1]);
        assertEquals(128, output.shape()[2]);
    }

    @Test
    public void testGRULayerReturnLast() {
        GRULayer gru = new GRULayer(100, 128, false);
        GradTensor input = GradTensor.randn(32, 50, 100);

        GradTensor output = gru.forward(input);

        assertNotNull(output);
        assertEquals(2, output.shape().length);
        assertEquals(32, output.shape()[0]);
        assertEquals(128, output.shape()[1]);
    }

    @Test
    public void testBidirectionalLSTM() {
        LSTMLayer forward = new LSTMLayer(100, 128, true);
        LSTMLayer backward = new LSTMLayer(100, 128, true);
        Bidirectional biLSTM = new Bidirectional(forward, backward);

        GradTensor input = GradTensor.randn(32, 50, 100);
        GradTensor output = biLSTM.forward(input);

        assertNotNull(output);
        assertEquals(3, output.shape().length);
        assertEquals(32, output.shape()[0]);   // batch size
        assertEquals(50, output.shape()[1]);   // sequence length
        assertEquals(256, output.shape()[2]);  // 2 * 128 (bidirectional)
    }

    @Test
    public void testBidirectionalGRU() {
        GRULayer forward = new GRULayer(100, 128, true);
        GRULayer backward = new GRULayer(100, 128, true);
        Bidirectional biGRU = new Bidirectional(forward, backward);

        GradTensor input = GradTensor.randn(32, 50, 100);
        GradTensor output = biGRU.forward(input);

        assertNotNull(output);
        assertEquals(256, output.shape()[2]);  // 2 * 128
    }

    @Test
    public void testBidirectionalMismatchedHiddenSize() {
        LSTMLayer forward = new LSTMLayer(100, 128, true);
        LSTMLayer backward = new LSTMLayer(100, 256, true);

        assertThrows(IllegalArgumentException.class, () -> {
            new Bidirectional(forward, backward);
        });
    }

    @Test
    public void testLSTMLayerProperties() {
        LSTMLayer lstm = new LSTMLayer(100, 256, true);

        assertEquals(256, lstm.getHiddenSize());
        assertTrue(lstm.isReturnSequences());
        assertNotNull(lstm.getCell());
    }

    @Test
    public void testGRULayerProperties() {
        GRULayer gru = new GRULayer(100, 256, false);

        assertEquals(256, gru.getHiddenSize());
        assertFalse(gru.isReturnSequences());
        assertNotNull(gru.getCell());
    }

    @Test
    public void testBidirectionalProperties() {
        LSTMLayer forward = new LSTMLayer(100, 128, true);
        LSTMLayer backward = new LSTMLayer(100, 128, true);
        Bidirectional bi = new Bidirectional(forward, backward);

        assertEquals(128, bi.getHiddenSize());
        assertEquals(256, bi.getOutputSize());
    }

    @Test
    public void testLSTMCellGradients() {
        LSTMCell cell = new LSTMCell(100, 128);
        GradTensor[] params = cell.getParameters();

        for (GradTensor param : params) {
            assertTrue(param.requiresGrad(), "All parameters should require gradients");
        }
    }

    @Test
    public void testGRUCellGradients() {
        GRUCell cell = new GRUCell(100, 128);
        GradTensor[] params = cell.getParameters();

        for (GradTensor param : params) {
            assertTrue(param.requiresGrad(), "All parameters should require gradients");
        }
    }

    @Test
    public void testLSTMLayerDifferentBatchSizes() {
        LSTMLayer lstm = new LSTMLayer(100, 128, false);

        GradTensor input1 = GradTensor.randn(1, 50, 100);
        GradTensor output1 = lstm.forward(input1);
        assertEquals(1, output1.shape()[0]);

        GradTensor input2 = GradTensor.randn(128, 50, 100);
        GradTensor output2 = lstm.forward(input2);
        assertEquals(128, output2.shape()[0]);
    }

    @Test
    public void testGRULayerDifferentSequenceLengths() {
        GRULayer gru = new GRULayer(100, 128, true);

        GradTensor input1 = GradTensor.randn(32, 10, 100);
        GradTensor output1 = gru.forward(input1);
        assertEquals(10, output1.shape()[1]);

        GradTensor input2 = GradTensor.randn(32, 100, 100);
        GradTensor output2 = gru.forward(input2);
        assertEquals(100, output2.shape()[1]);
    }

    @Test
    public void testLSTMLayerVeryLongSequence() {
        LSTMLayer lstm = new LSTMLayer(100, 256, false);
        GradTensor input = GradTensor.randn(4, 1000, 100);  // Very long sequence
        
        GradTensor output = lstm.forward(input);
        assertEquals(4, output.shape()[0]);
        assertEquals(256, output.shape()[1]);
    }

    @Test
    public void testStackedLSTMLayers() {
        LSTMLayer lstm1 = new LSTMLayer(100, 128, true);
        LSTMLayer lstm2 = new LSTMLayer(128, 64, false);

        GradTensor input = GradTensor.randn(32, 50, 100);
        GradTensor hidden1 = lstm1.forward(input);  // [32, 50, 128]
        GradTensor output = lstm2.forward(hidden1);  // [32, 64]

        assertEquals(2, output.shape().length);
        assertEquals(32, output.shape()[0]);
        assertEquals(64, output.shape()[1]);
    }

    @Test
    public void testLSTMCellToString() {
        LSTMCell cell = new LSTMCell(100, 256);
        String str = cell.toString();

        assertTrue(str.contains("LSTMCell"));
        assertTrue(str.contains("100"));
        assertTrue(str.contains("256"));
    }

    @Test
    public void testGRUCellToString() {
        GRUCell cell = new GRUCell(100, 256);
        String str = cell.toString();

        assertTrue(str.contains("GRUCell"));
        assertTrue(str.contains("100"));
        assertTrue(str.contains("256"));
    }

    @Test
    public void testLSTMLayerToString() {
        LSTMLayer lstm = new LSTMLayer(100, 256, true);
        String str = lstm.toString();

        assertTrue(str.contains("LSTMLayer"));
        assertTrue(str.contains("true"));
    }

    @Test
    public void testBidirectionalToString() {
        LSTMLayer forward = new LSTMLayer(100, 128, true);
        LSTMLayer backward = new LSTMLayer(100, 128, true);
        Bidirectional bi = new Bidirectional(forward, backward);
        
        String str = bi.toString();
        assertTrue(str.contains("Bidirectional"));
        assertTrue(str.contains("256"));
    }
}
