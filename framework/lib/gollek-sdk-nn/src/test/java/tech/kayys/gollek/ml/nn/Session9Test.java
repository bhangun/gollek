package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.metrics.RegressionMetrics;
import tech.kayys.gollek.ml.nn.metrics.SegmentationMetrics;
import tech.kayys.gollek.ml.nn.optim.Adagrad;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Session 9 tests: BatchNorm2d, GroupNorm, Adagrad,
 * RegressionMetrics, SegmentationMetrics.
 */
class Session9Test {

    // ── BatchNorm2d ───────────────────────────────────────────────────────

    @Test
    void batchNorm2dOutputShape() {
        BatchNorm2d bn = new BatchNorm2d(8);
        GradTensor x = GradTensor.randn(2, 8, 4, 4);
        assertArrayEquals(new long[]{2, 8, 4, 4}, bn.forward(x).shape());
    }

    @Test
    void batchNorm2dNormalizesTraining() {
        BatchNorm2d bn = new BatchNorm2d(1);
        // All same value → output should be near 0 (normalized)
        float[] data = new float[1 * 1 * 4 * 4]; java.util.Arrays.fill(data, 3f);
        GradTensor x = GradTensor.of(data, 1, 1, 4, 4);
        GradTensor out = bn.forward(x);
        for (float v : out.data()) assertEquals(0f, v, 1e-4f);
    }

    @Test
    void batchNorm2dEvalUsesRunningStats() {
        BatchNorm2d bn = new BatchNorm2d(2);
        // Train once to populate running stats
        bn.train();
        bn.forward(GradTensor.randn(4, 2, 4, 4));
        // Eval should not throw
        bn.eval();
        assertDoesNotThrow(() -> bn.forward(GradTensor.randn(1, 2, 4, 4)));
    }

    // ── GroupNorm ─────────────────────────────────────────────────────────

    @Test
    void groupNormOutputShape() {
        GroupNorm gn = new GroupNorm(4, 8);
        GradTensor x = GradTensor.randn(2, 8, 4, 4);
        assertArrayEquals(new long[]{2, 8, 4, 4}, gn.forward(x).shape());
    }

    @Test
    void groupNormSingleGroup() {
        // numGroups=1 is equivalent to LayerNorm over spatial dims
        GroupNorm gn = new GroupNorm(1, 4);
        GradTensor x = GradTensor.randn(2, 4, 3, 3);
        assertArrayEquals(new long[]{2, 4, 3, 3}, gn.forward(x).shape());
    }

    @Test
    void groupNormInvalidGroupsThrows() {
        assertThrows(IllegalArgumentException.class, () -> new GroupNorm(3, 8));
    }

    // ── Adagrad ───────────────────────────────────────────────────────────

    @Test
    void adagradReducesLoss() {
        Linear model = new Linear(4, 2);
        Adagrad opt = new Adagrad(model.parameters(), 0.1f);
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
        assertTrue(last < first, "Adagrad should reduce loss");
    }

    @Test
    void adagradLearningRateAccessor() {
        Adagrad opt = new Adagrad(new Linear(2, 2).parameters(), 0.05f);
        assertEquals(0.05f, opt.learningRate(), 1e-6f);
        opt.setLearningRate(0.01f);
        assertEquals(0.01f, opt.learningRate(), 1e-6f);
    }

    // ── RegressionMetrics ─────────────────────────────────────────────────

    @Test
    void regressionMetricsPerfect() {
        float[] pred = {1f, 2f, 3f}, actual = {1f, 2f, 3f};
        RegressionMetrics.Result r = RegressionMetrics.compute(pred, actual);
        assertEquals(0f, r.mae(),  1e-5f);
        assertEquals(0f, r.rmse(), 1e-5f);
        assertEquals(1f, r.r2(),   1e-5f);
    }

    @Test
    void regressionMetricsMAE() {
        float[] pred = {1f, 3f}, actual = {2f, 2f};
        assertEquals(1f, RegressionMetrics.mae(pred, actual), 1e-5f);
    }

    @Test
    void regressionMetricsRMSE() {
        float[] pred = {0f, 2f}, actual = {1f, 1f};
        // errors = [-1, 1], MSE = 1, RMSE = 1
        assertEquals(1f, RegressionMetrics.rmse(pred, actual), 1e-5f);
    }

    @Test
    void regressionMetricsR2() {
        // Constant prediction → R² = 0
        float[] pred = {2f, 2f, 2f}, actual = {1f, 2f, 3f};
        float r2 = RegressionMetrics.r2(pred, actual);
        assertEquals(0f, r2, 0.01f);
    }

    // ── SegmentationMetrics ───────────────────────────────────────────────

    @Test
    void iouPerfect() {
        float[] pred = {1f, 1f, 0f, 0f}, target = {1f, 1f, 0f, 0f};
        assertEquals(1f, SegmentationMetrics.iou(pred, target), 1e-5f);
    }

    @Test
    void iouNoOverlap() {
        float[] pred = {1f, 0f}, target = {0f, 1f};
        assertEquals(0f, SegmentationMetrics.iou(pred, target), 1e-5f);
    }

    @Test
    void diceScore() {
        float[] pred = {1f, 1f, 0f}, target = {1f, 0f, 1f};
        // intersection=1, sum=2+2=4, dice=2/4=0.5
        assertEquals(0.5f, SegmentationMetrics.dice(pred, target), 1e-5f);
    }

    @Test
    void meanIoUMultiClass() {
        float[] pred   = {0f, 1f, 2f, 0f};
        float[] target = {0f, 1f, 2f, 1f};
        float mIoU = SegmentationMetrics.meanIoU(pred, target, 3);
        assertTrue(mIoU > 0f && mIoU <= 1f);
    }
}
