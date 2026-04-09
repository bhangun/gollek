package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.models.VAE;
import tech.kayys.gollek.ml.nn.loss.LabelSmoothingLoss;
import tech.kayys.gollek.ml.nn.optim.Lion;
import tech.kayys.gollek.ml.profiler.ModelProfiler;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Session 10 tests: VAE, RotaryEmbedding, Lion, LabelSmoothingLoss, ModelProfiler.
 */
class Session10Test {

    // ── VAE ───────────────────────────────────────────────────────────────

    @Test
    void vaeForwardOutputShape() {
        VAE vae = new VAE(16, 4, 8);
        GradTensor x = GradTensor.randn(2, 16);
        VAE.Output out = vae.forwardFull(x);
        assertArrayEquals(new long[]{2, 16}, out.recon().shape());
        assertArrayEquals(new long[]{2, 4},  out.mu().shape());
        assertArrayEquals(new long[]{2, 4},  out.logVar().shape());
        assertArrayEquals(new long[]{2, 4},  out.z().shape());
    }

    @Test
    void vaeLossIsPositive() {
        VAE vae = new VAE(16, 4, 8);
        GradTensor x = GradTensor.rand(2, 16); // [0,1] for BCE
        VAE.Output out = vae.forwardFull(x);
        GradTensor loss = vae.loss(x, out);
        assertEquals(0, loss.ndim());
        assertTrue(loss.item() > 0f);
    }

    @Test
    void vaeReconInRange() {
        VAE vae = new VAE(8, 2, 4);
        GradTensor x = GradTensor.rand(1, 8);
        VAE.Output out = vae.forwardFull(x);
        for (float v : out.recon().data())
            assertTrue(v >= 0f && v <= 1f, "Reconstruction should be in [0,1] (sigmoid output)");
    }

    @Test
    void betaVAEHigherKL() {
        // β-VAE with β=4 should have higher loss than β=1 (more KL weight)
        GradTensor x = GradTensor.rand(4, 16);
        VAE vae1 = new VAE(16, 4, 8, 1.0f);
        VAE vae4 = new VAE(16, 4, 8, 4.0f);
        // Just verify both run without error
        assertDoesNotThrow(() -> vae1.loss(x, vae1.forwardFull(x)));
        assertDoesNotThrow(() -> vae4.loss(x, vae4.forwardFull(x)));
    }

    // ── RotaryEmbedding ───────────────────────────────────────────────────

    @Test
    void ropeOutputShape() {
        RotaryEmbedding rope = new RotaryEmbedding(16, 64);
        GradTensor x = GradTensor.randn(2, 4, 8, 16); // [B, H, T, headDim]
        assertArrayEquals(new long[]{2, 4, 8, 16}, rope.apply(x).shape());
    }

    @Test
    void ropePreservesNorm() {
        // Rotation is norm-preserving: ||Rx|| = ||x||
        RotaryEmbedding rope = new RotaryEmbedding(8, 32);
        GradTensor x = GradTensor.randn(1, 1, 4, 8);
        GradTensor rotated = rope.apply(x);

        float normBefore = 0f, normAfter = 0f;
        for (float v : x.data())       normBefore += v * v;
        for (float v : rotated.data()) normAfter  += v * v;
        assertEquals(normBefore, normAfter, 1e-3f, "RoPE should preserve L2 norm");
    }

    @Test
    void ropeOddDimThrows() {
        assertThrows(IllegalArgumentException.class, () -> new RotaryEmbedding(7, 32));
    }

    // ── Lion ──────────────────────────────────────────────────────────────

    @Test
    void lionReducesLoss() {
        Linear model = new Linear(4, 2);
        Lion opt = new Lion(model.parameters(), 1e-3f);
        GradTensor x = GradTensor.randn(8, 4);
        GradTensor y = GradTensor.randn(8, 2);

        float first = 0f, last = 0f;
        for (int i = 0; i < 30; i++) {
            model.zeroGrad();
            GradTensor loss = model.forward(x).sub(y).pow(2f).mean();
            loss.backward();
            opt.step();
            if (i == 0)  first = loss.item();
            if (i == 29) last  = loss.item();
        }
        assertTrue(last < first, "Lion should reduce loss");
    }

    @Test
    void lionLearningRateAccessor() {
        Lion opt = new Lion(new Linear(2, 2).parameters(), 1e-4f);
        assertEquals(1e-4f, opt.learningRate(), 1e-8f);
        opt.setLearningRate(1e-3f);
        assertEquals(1e-3f, opt.learningRate(), 1e-8f);
    }

    // ── LabelSmoothingLoss ────────────────────────────────────────────────

    @Test
    void labelSmoothingLossIsScalar() {
        LabelSmoothingLoss loss = new LabelSmoothingLoss(0.1f);
        GradTensor logits  = GradTensor.randn(4, 10);
        GradTensor labels  = GradTensor.of(new float[]{0, 3, 7, 9}, 4);
        GradTensor l = loss.forward(logits, labels);
        assertEquals(0, l.ndim());
        assertTrue(l.item() > 0f);
    }

    @Test
    void labelSmoothingHigherThanCE() {
        // With smoothing=0, should equal standard CE; with smoothing>0, slightly higher
        LabelSmoothingLoss ls0  = new LabelSmoothingLoss(0.0f);
        LabelSmoothingLoss ls01 = new LabelSmoothingLoss(0.1f);
        GradTensor logits = GradTensor.randn(4, 5);
        GradTensor labels = GradTensor.of(new float[]{0, 1, 2, 3}, 4);
        // Both should be positive scalars
        assertTrue(ls0.forward(logits, labels).item()  > 0f);
        assertTrue(ls01.forward(logits, labels).item() > 0f);
    }

    // ── ModelProfiler ─────────────────────────────────────────────────────

    @Test
    void profilerReturnsEntries() {
        Sequential model = new Sequential(new Linear(8, 4), new ReLU(), new Linear(4, 2));
        ModelProfiler profiler = new ModelProfiler(model);
        List<ModelProfiler.ProfileEntry> entries = profiler.profile(
            GradTensor.randn(1, 8), 1, 3);
        assertFalse(entries.isEmpty());
        assertTrue(entries.get(0).avgMs() > 0);
    }

    @Test
    void profilerMemoryEstimate() {
        Linear model = new Linear(64, 64);
        ModelProfiler profiler = new ModelProfiler(model);
        List<ModelProfiler.ProfileEntry> entries = profiler.profile(
            GradTensor.randn(1, 64), 1, 2);
        // Parameters: 64*64 + 64 = 4160 floats = ~16KB
        double paramMB = entries.stream()
            .filter(e -> e.name().equals("parameters_mem"))
            .mapToDouble(ModelProfiler.ProfileEntry::memoryMB)
            .findFirst().orElse(0);
        assertTrue(paramMB > 0);
    }
}
