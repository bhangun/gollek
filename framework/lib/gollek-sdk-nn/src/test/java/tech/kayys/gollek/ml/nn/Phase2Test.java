package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.metrics.MetricsTracker;
import tech.kayys.gollek.ml.nn.metrics.ClassificationMetrics;
import tech.kayys.gollek.ml.nn.optim.*;
// TODO: optimize package not yet implemented - import tech.kayys.gollek.ml.optimize.PostTrainingQuantizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 2 integration tests covering:
 * ClassificationMetrics, WarmupCosineScheduler, GradientClipper,
 * MetricsTracker, and PostTrainingQuantizer.
 */
class Phase2Test {

    // ── ClassificationMetrics ─────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void classificationMetricsPerfect() {
        int[] pred   = {0, 1, 2, 0, 1, 2};
        int[] actual = {0, 1, 2, 0, 1, 2};
        var r = ClassificationMetrics.compute(pred, actual, 3);
        assertEquals(1f, r.precision(), 1e-5f);
        assertEquals(1f, r.recall(),    1e-5f);
        assertEquals(1f, r.f1(),        1e-5f);
        assertEquals(1f, r.accuracy(),  1e-5f);
    }
    void classificationMetricsAllWrong() {
        int[] pred   = {1, 2, 0};
        int[] actual = {0, 1, 2};
        var r = ClassificationMetrics.compute(pred, actual, 3);
        assertEquals(0f, r.accuracy(), 1e-5f);
    }

    @Disabled("Requires optimize package") @Test
    void confusionMatrixDiagonal() {
        int[] pred   = {0, 1, 2};
        int[] actual = {0, 1, 2};
        float[][] cm = ClassificationMetrics.confusionMatrix(pred, actual, 3);
        for (int i = 0; i < 3; i++) {
            assertEquals(1f, cm[i][i], 1e-5f);
            for (int j = 0; j < 3; j++) if (i != j) assertEquals(0f, cm[i][j], 1e-5f);
        }
    }

    @Disabled("Requires optimize package") @Test
    void topKAccuracyTop1() {
        float[][] logits = {{10f, 1f, 1f}, {1f, 10f, 1f}};
        int[] labels = {0, 1};
        assertEquals(1f, ClassificationMetrics.topKAccuracy(logits, labels, 1), 1e-5f);
    }

    @Disabled("Requires optimize package") @Test
    void topKAccuracyTop3() {
        float[][] logits = {{1f, 2f, 10f}, {10f, 2f, 1f}};
        int[] labels = {0, 2}; // both in top-3
        assertEquals(1f, ClassificationMetrics.topKAccuracy(logits, labels, 3), 1e-5f);
    }

    // ── WarmupCosineScheduler ─────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void warmupSchedulerLinearPhase() {
        Linear model = new Linear(2, 1);
        Adam opt = new Adam(model.parameters(), 0.001f);
        var sched = new WarmupCosineScheduler(opt, 10, 100, 1e-3f, 1e-6f);

        float prev = 0f;
        for (int i = 1; i <= 10; i++) {
            sched.step();
            assertTrue(sched.getLr() > prev, "LR should increase during warmup at step " + i);
            prev = sched.getLr();
        }
        assertEquals(1e-3f, sched.getLr(), 1e-6f, "LR should reach maxLr at end of warmup");
    }

    @Disabled("Requires optimize package") @Test
    void warmupSchedulerCosinePhase() {
        Linear model = new Linear(2, 1);
        Adam opt = new Adam(model.parameters(), 0.001f);
        var sched = new WarmupCosineScheduler(opt, 5, 100, 1e-3f, 1e-6f);

        // Skip warmup
        for (int i = 0; i < 5; i++) sched.step();
        float afterWarmup = sched.getLr();

        // Cosine phase should decrease
        sched.step();
        assertTrue(sched.getLr() < afterWarmup, "LR should decrease after warmup");
    }

    // ── GradientClipper ───────────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void gradientClipByNormEnforced() {
        Linear model = new Linear(4, 4);
        // Manually set large gradients
        for (Parameter p : model.parameters()) {
            p.data().requiresGrad(true);
            float[] g = p.data().grad().data();
            java.util.Arrays.fill(g, 10f);
        }
        float norm = GradientClipper.clipByNorm(model.parameters(), 1.0f);
        assertTrue(norm > 1.0f, "Pre-clip norm should be > 1");

        // After clipping, recompute norm
        float postNorm = 0f;
        for (Parameter p : model.parameters()) {
            float[] g = p.data().grad().data();
            for (float v : g) postNorm += v * v;
        }
        postNorm = (float) Math.sqrt(postNorm);
        assertTrue(postNorm <= 1.0f + 1e-4f, "Post-clip norm should be <= maxNorm");
    }

    @Disabled("Requires optimize package") @Test
    void gradientClipByValueClamped() {
        Linear model = new Linear(2, 2);
        for (Parameter p : model.parameters()) {
            p.data().requiresGrad(true);
            float[] g = p.data().grad().data();
            java.util.Arrays.fill(g, 5f);
        }
        GradientClipper.clipByValue(model.parameters(), -1f, 1f);
        for (Parameter p : model.parameters()) {
            for (float v : p.data().grad().data()) {
                assertTrue(v >= -1f && v <= 1f, "Gradient should be clamped to [-1, 1]");
            }
        }
    }

    // ── MetricsTracker ────────────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void metricsTrackerLogAndRetrieve() {
        MetricsTracker tracker = new MetricsTracker();
        tracker.log("loss", 1.0f, 0);
        tracker.log("loss", 0.5f, 1);
        tracker.log("loss", 0.2f, 2);

        assertEquals(0.2f, tracker.latest("loss"), 1e-5f);
        assertEquals(0.2f, tracker.min("loss"),    1e-5f);
        assertEquals(1.0f, tracker.max("loss"),    1e-5f);
        assertEquals((1.0f + 0.5f + 0.2f) / 3f, tracker.mean("loss"), 1e-4f);
    }

    @Disabled("Requires optimize package") @Test
    void metricsTrackerLogAll() {
        MetricsTracker tracker = new MetricsTracker();
        tracker.logAll(Map.of("loss", 0.5f, "acc", 0.9f), 1);
        assertEquals(0.5f, tracker.latest("loss"), 1e-5f);
        assertEquals(0.9f, tracker.latest("acc"),  1e-5f);
    }

    @Disabled("Requires optimize package") @Test
    void metricsTrackerSummary() {
        MetricsTracker tracker = new MetricsTracker();
        tracker.log("a", 1f, 0); tracker.log("a", 2f, 1);
        tracker.log("b", 3f, 0);
        var s = tracker.summary();
        assertEquals(2f, s.get("a"), 1e-5f);
        assertEquals(3f, s.get("b"), 1e-5f);
    }

    @Disabled("Requires optimize package") @Test
    void metricsTrackerCsvExport(@TempDir Path tmpDir) throws IOException {
        MetricsTracker tracker = new MetricsTracker();
        tracker.log("loss", 1.0f, 0);
        tracker.log("loss", 0.5f, 1);
        tracker.log("acc",  0.8f, 0);

        Path csv = tmpDir.resolve("metrics.csv");
        tracker.exportCsv(csv);

        assertTrue(Files.exists(csv));
        List<String> lines = Files.readAllLines(csv);
        assertEquals("step,name,value", lines.get(0));
        assertEquals(4, lines.size()); // header + 3 data rows
    }

    @Disabled("Requires optimize package") @Test
    void metricsTrackerReset() {
        MetricsTracker tracker = new MetricsTracker();
        tracker.log("loss", 1f, 0);
        tracker.reset();
        assertTrue(tracker.metricNames().isEmpty());
        assertTrue(Float.isNaN(tracker.latest("loss")));
    }

    // ── PostTrainingQuantizer ─────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void quantizerRoundTrip() {
        // TODO: PostTrainingQuantizer not yet implemented
        /*
        PostTrainingQuantizer q = new PostTrainingQuantizer();
        GradTensor t = GradTensor.randn(64, 64);
        PostTrainingQuantizer.QuantizedTensor qt = q.quantize(t);
        GradTensor deq = q.dequantize(qt);

        float[] orig = t.data(), rec = deq.data();
        float maxErr = 0f;
        for (int i = 0; i < orig.length; i++) maxErr = Math.max(maxErr, Math.abs(orig[i] - rec[i]));
        assertTrue(maxErr < 0.05f, "Max dequantization error should be < 0.05, got " + maxErr);
        */
    }

    @Disabled("Requires optimize package") @Test
    void quantizerCompressionRatio() {
        // TODO: PostTrainingQuantizer not yet implemented
        /*
        PostTrainingQuantizer q = new PostTrainingQuantizer();
        Linear model = new Linear(128, 64);
        var qModel = q.quantizeModel(model.stateDict());
        assertEquals(4.0f, q.compressionRatio(model.stateDict(), qModel), 1e-5f);
        */
    }

    @Disabled("Requires optimize package") @Test
    void quantizerPreservesShape() {
        // TODO: PostTrainingQuantizer not yet implemented
        /*
        PostTrainingQuantizer q = new PostTrainingQuantizer();
        GradTensor t = GradTensor.randn(3, 4, 5);
        var qt = q.quantize(t);
        assertArrayEquals(new long[]{3, 4, 5}, qt.shape());
        assertEquals(3 * 4 * 5, qt.numel());
        */
    }

    @Disabled("Requires optimize package") @Test
    void quantizerModelAllKeys() {
        // TODO: PostTrainingQuantizer not yet implemented
        /*
        PostTrainingQuantizer q = new PostTrainingQuantizer();
        Sequential model = new Sequential(new Linear(4, 8), new Linear(8, 2));
        var sd = model.stateDict();
        var qsd = q.quantizeModel(sd);
        assertEquals(sd.keySet(), qsd.keySet());
        */
    }
}
