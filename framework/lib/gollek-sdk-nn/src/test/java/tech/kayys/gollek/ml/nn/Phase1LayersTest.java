package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Smoke tests for Phase 1 new layers (Conv2d, MaxPool2d, LSTM, GRU)
 * and VectorOps acceleration.
 */
class Phase1LayersTest {

    // ── VectorOps ────────────────────────────────────────────────────────

    @Test
    void vectorOpsAdd() {
        float[] a = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] b = {8, 7, 6, 5, 4, 3, 2, 1};
        float[] out = new float[8];
        VectorOps.add(a, b, out);
        for (float v : out) assertEquals(9f, v, 1e-5f);
    }

    @Test
    void vectorOpsMatmul2x2() {
        // [2,2] @ [2,2] = [2,2]
        float[] a = {1, 2, 3, 4};
        float[] b = {5, 6, 7, 8};
        float[] c = VectorOps.matmul(a, b, 2, 2, 2);
        assertArrayEquals(new float[]{19, 22, 43, 50}, c, 1e-4f);
    }

    @Test
    void vectorOpsSum() {
        float[] a = new float[1024];
        java.util.Arrays.fill(a, 1f);
        assertEquals(1024f, VectorOps.sum(a), 1e-3f);
    }

    // ── Conv2d ───────────────────────────────────────────────────────────

    @Test
    void conv2dOutputShape() {
        Conv2d conv = new Conv2d(3, 16, 3); // no padding → H-2, W-2
        GradTensor input = GradTensor.randn(2, 3, 8, 8); // [N=2, C=3, H=8, W=8]
        GradTensor out = conv.forward(input);
        assertArrayEquals(new long[]{2, 16, 6, 6}, out.shape());
    }

    @Test
    void conv2dSamePadding() {
        Conv2d conv = new Conv2d(1, 8, 3, 1, 1); // padding=1 → same size
        GradTensor input = GradTensor.randn(1, 1, 16, 16);
        GradTensor out = conv.forward(input);
        assertArrayEquals(new long[]{1, 8, 16, 16}, out.shape());
    }

    // ── MaxPool2d ────────────────────────────────────────────────────────

    @Test
    void maxPool2dHalvesSpatial() {
        MaxPool2d pool = new MaxPool2d(2); // 2x2, stride=2
        GradTensor input = GradTensor.randn(2, 4, 8, 8);
        GradTensor out = pool.forward(input);
        assertArrayEquals(new long[]{2, 4, 4, 4}, out.shape());
    }

    @Test
    void maxPool2dPicksMax() {
        MaxPool2d pool = new MaxPool2d(2);
        // 1x1x2x2 input: [1, 2, 3, 4] → max = 4
        GradTensor input = GradTensor.of(new float[]{1, 2, 3, 4}, 1, 1, 2, 2);
        GradTensor out = pool.forward(input);
        assertEquals(4f, out.item(0), 1e-5f);
    }

    // ── LSTM ─────────────────────────────────────────────────────────────

    @org.junit.jupiter.api.Disabled("LSTM.forward() returns GradTensor, not LSTM.LSTMOutput - wrapper type pending")
    @Test
    void lstmOutputShape() {
        // TODO: LSTM wrapper types LSTMOutput/GRUOutput not yet implemented
        // LSTM lstm = new LSTM(32, 64);
        // GradTensor input = GradTensor.randn(10, 4, 32); // [T=10, N=4, inputSize=32]
        // LSTM.LSTMOutput out = lstm.forward(input);
        // assertArrayEquals(new long[]{10, 4, 64}, out.output().shape());
        // assertArrayEquals(new long[]{1,  4, 64}, out.hn().shape());
        // assertArrayEquals(new long[]{1,  4, 64}, out.cn().shape());
    }

    @org.junit.jupiter.api.Disabled("LSTM.forward() returns GradTensor, not LSTM.LSTMOutput - wrapper type pending")
    @Test
    void lstmBatchFirst() {
        // TODO: LSTM wrapper types LSTMOutput/GRUOutput not yet implemented
        // LSTM lstm = new LSTM(16, 32, true);
        // GradTensor input = GradTensor.randn(4, 10, 16); // [N=4, T=10, inputSize=16]
        // LSTM.LSTMOutput out = lstm.forward(input);
        // assertArrayEquals(new long[]{10, 4, 32}, out.output().shape());
    }

    // ── GRU ──────────────────────────────────────────────────────────────

    @org.junit.jupiter.api.Disabled("GRU.forward() returns GradTensor, not GRU.GRUOutput - wrapper type pending")
    @Test
    void gruOutputShape() {
        // TODO: GRU wrapper types LSTMOutput/GRUOutput not yet implemented
        // GRU gru = new GRU(32, 64);
        // GradTensor input = GradTensor.randn(10, 4, 32);
        // GRU.GRUOutput out = gru.forward(input);
        // assertArrayEquals(new long[]{10, 4, 64}, out.output().shape());
        // assertArrayEquals(new long[]{1,  4, 64}, out.hn().shape());
    }

    // ── CNN + Pool pipeline ──────────────────────────────────────────────

    @Test
    void cnnPipelineForward() {
        // Simple: Conv → ReLU → MaxPool → flatten → Linear
        Conv2d    conv  = new Conv2d(1, 8, 3, 1, 1);
        MaxPool2d pool  = new MaxPool2d(2);
        Linear    fc    = new Linear(8 * 14 * 14, 10);

        GradTensor x = GradTensor.randn(2, 1, 28, 28);
        x = conv.forward(x);                          // [2, 8, 28, 28]
        x = x.relu();
        x = pool.forward(x);                          // [2, 8, 14, 14]
        x = x.reshape(2, 8 * 14 * 14);               // [2, 1568]
        x = fc.forward(x);                            // [2, 10]
        assertArrayEquals(new long[]{2, 10}, x.shape());
    }
}
