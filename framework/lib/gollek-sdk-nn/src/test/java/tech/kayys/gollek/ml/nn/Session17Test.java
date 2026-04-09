package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.TensorOps;
import tech.kayys.gollek.ml.models.T5;
import tech.kayys.gollek.ml.models.VGG;
import tech.kayys.gollek.ml.nn.loss.HuberLoss;
import tech.kayys.gollek.ml.nn.loss.IoULoss;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Session 17 tests: IoULoss, HuberLoss, VGG, T5, TensorOps (permute/repeat/chunk).
 */
class Session17Test {

    // ── IoULoss ───────────────────────────────────────────────────────────

    @Test
    void iouLossPerfectOverlap() {
        IoULoss loss = new IoULoss();
        GradTensor pred   = GradTensor.of(new float[]{0,0,2,2}, 1, 4);
        GradTensor target = GradTensor.of(new float[]{0,0,2,2}, 1, 4);
        assertEquals(0f, loss.forward(pred, target).item(), 1e-4f);
    }

    @Test
    void iouLossNoOverlap() {
        IoULoss loss = new IoULoss();
        GradTensor pred   = GradTensor.of(new float[]{0,0,1,1}, 1, 4);
        GradTensor target = GradTensor.of(new float[]{2,2,3,3}, 1, 4);
        assertEquals(1f, loss.forward(pred, target).item(), 1e-4f);
    }

    @Test
    void iouLossPartialOverlap() {
        IoULoss loss = new IoULoss();
        GradTensor pred   = GradTensor.of(new float[]{0,0,2,2}, 1, 4);
        GradTensor target = GradTensor.of(new float[]{1,1,3,3}, 1, 4);
        float l = loss.forward(pred, target).item();
        assertTrue(l > 0f && l < 1f);
    }

    // ── HuberLoss ─────────────────────────────────────────────────────────

    @Test
    void huberLossSmallError() {
        HuberLoss loss = new HuberLoss(1.0f);
        // |error| = 0.5 < delta → quadratic: 0.5 * 0.5^2 = 0.125
        GradTensor pred   = GradTensor.of(new float[]{1.5f}, 1);
        GradTensor target = GradTensor.of(new float[]{1.0f}, 1);
        assertEquals(0.125f, loss.forward(pred, target).item(), 1e-5f);
    }

    @Test
    void huberLossLargeError() {
        HuberLoss loss = new HuberLoss(1.0f);
        // |error| = 2.0 > delta → linear: 1.0*(2.0 - 0.5) = 1.5
        GradTensor pred   = GradTensor.of(new float[]{3.0f}, 1);
        GradTensor target = GradTensor.of(new float[]{1.0f}, 1);
        assertEquals(1.5f, loss.forward(pred, target).item(), 1e-5f);
    }

    @Test
    void huberLossZeroError() {
        HuberLoss loss = new HuberLoss();
        GradTensor x = GradTensor.of(new float[]{1f, 2f, 3f}, 3);
        assertEquals(0f, loss.forward(x, x).item(), 1e-6f);
    }

    // ── VGG ───────────────────────────────────────────────────────────────

    @Test
    void vgg11OutputShape() {
        NNModule model = VGG.vgg11(10);
        GradTensor x = GradTensor.randn(1, 3, 32, 32);
        assertArrayEquals(new long[]{1, 10}, model.forward(x).shape());
    }

    @Test
    void vgg16OutputShape() {
        NNModule model = VGG.vgg16(5);
        GradTensor x = GradTensor.randn(1, 3, 32, 32);
        assertArrayEquals(new long[]{1, 5}, model.forward(x).shape());
    }

    @Test
    void vgg16HasParameters() {
        NNModule model = VGG.vgg16(1000);
        assertTrue(model.parameterCount() > 1_000_000L);
    }

    // ── T5 ────────────────────────────────────────────────────────────────

    @Test
    void t5TinyOutputShape() {
        T5.T5Model t5 = T5.t5Tiny(100);
        GradTensor enc = GradTensor.randn(2, 8, 128);
        GradTensor dec = GradTensor.randn(2, 6, 128);
        GradTensor out = t5.forward(enc, dec);
        assertArrayEquals(new long[]{2, 6, 100}, out.shape());
    }

    @Test
    void t5TinyHasParameters() {
        T5.T5Model t5 = T5.t5Tiny(100);
        assertTrue(t5.parameterCount() > 10_000L);
    }

    // ── TensorOps: permute ────────────────────────────────────────────────

    @Test
    void permuteTranspose() {
        GradTensor t = GradTensor.of(new float[]{1,2,3,4,5,6}, 2, 3);
        GradTensor p = TensorOps.permute(t, 1, 0); // transpose
        assertArrayEquals(new long[]{3, 2}, p.shape());
        // t[0,1]=2 → p[1,0]=2
        assertEquals(2f, p.data()[2], 1e-5f);
    }

    @Test
    void permute3D() {
        GradTensor t = GradTensor.randn(2, 3, 4);
        GradTensor p = TensorOps.permute(t, 0, 2, 1); // swap last two dims
        assertArrayEquals(new long[]{2, 4, 3}, p.shape());
    }

    // ── TensorOps: repeat ─────────────────────────────────────────────────

    @Test
    void repeatDim0() {
        GradTensor t = GradTensor.of(new float[]{1, 2, 3}, 3);
        GradTensor r = TensorOps.repeat(t, 2);
        assertArrayEquals(new long[]{6}, r.shape());
        assertArrayEquals(new float[]{1,2,3,1,2,3}, r.data(), 1e-5f);
    }

    @Test
    void repeat2D() {
        GradTensor t = GradTensor.of(new float[]{1,2,3,4}, 2, 2);
        GradTensor r = TensorOps.repeat(t, 1, 2); // repeat cols twice
        assertArrayEquals(new long[]{2, 4}, r.shape());
    }

    // ── TensorOps: chunk ──────────────────────────────────────────────────

    @Test
    void chunkEvenSplit() {
        GradTensor t = GradTensor.of(new float[]{1,2,3,4,5,6}, 6);
        List<GradTensor> chunks = TensorOps.chunk(t, 2, 0);
        assertEquals(3, chunks.size());
        assertArrayEquals(new long[]{2}, chunks.get(0).shape());
    }

    @Test
    void chunkUnevenSplit() {
        GradTensor t = GradTensor.randn(7, 4);
        List<GradTensor> chunks = TensorOps.chunk(t, 3, 0);
        assertEquals(3, chunks.size()); // [3,4], [3,4], [1,4]
        assertEquals(1, chunks.get(2).shape()[0]);
    }
}
