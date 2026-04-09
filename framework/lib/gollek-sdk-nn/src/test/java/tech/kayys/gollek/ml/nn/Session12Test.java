package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.benchmark.BenchmarkSuite;
import tech.kayys.gollek.ml.interop.PythonBridge;
import tech.kayys.gollek.ml.nn.StateDict;
import tech.kayys.gollek.ml.nn.loss.CTCLoss;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Session 12 tests: FlashAttention, GCNConv, CTCLoss, PythonBridge, BenchmarkSuite.
 */
class Session12Test {

    // ── FlashAttention ────────────────────────────────────────────────────

    @Test
    void flashAttentionOutputShape() {
        FlashAttention attn = new FlashAttention(32, 4);
        GradTensor x = GradTensor.randn(2, 8, 32); // [B, T, dModel]
        assertArrayEquals(new long[]{2, 8, 32}, attn.forward(x).shape());
    }

    @Test
    void flashAttentionCausalOutputShape() {
        FlashAttention attn = new FlashAttention(32, 4, true);
        GradTensor x = GradTensor.randn(1, 16, 32);
        assertArrayEquals(new long[]{1, 16, 32}, attn.forward(x).shape());
    }

    @Test
    void flashAttentionSmallBlockSize() {
        FlashAttention attn = new FlashAttention(16, 2, false, 4);
        GradTensor x = GradTensor.randn(1, 12, 16);
        assertArrayEquals(new long[]{1, 12, 16}, attn.forward(x).shape());
    }

    @Test
    void flashAttentionInvalidDimThrows() {
        assertThrows(IllegalArgumentException.class, () -> new FlashAttention(10, 3));
    }

    // ── GCNConv ───────────────────────────────────────────────────────────

    @Test
    void gcnConvOutputShape() {
        GCNConv gcn = new GCNConv(4, 8);
        int N = 5;
        GradTensor x   = GradTensor.randn(N, 4);
        GradTensor adj = GradTensor.of(new float[N * N], N, N); // empty graph
        // Add some edges
        float[] a = adj.data(); a[0*N+1] = 1f; a[1*N+0] = 1f; a[2*N+3] = 1f;
        assertArrayEquals(new long[]{N, 8}, gcn.forward(x, adj).shape());
    }

    @Test
    void gcnConvSelfLoopGraph() {
        GCNConv gcn = new GCNConv(3, 6);
        int N = 3;
        // Identity adjacency (self-loops only)
        float[] a = new float[N * N];
        for (int i = 0; i < N; i++) a[i * N + i] = 1f;
        GradTensor x   = GradTensor.randn(N, 3);
        GradTensor adj = GradTensor.of(a, N, N);
        assertArrayEquals(new long[]{N, 6}, gcn.forward(x, adj).shape());
    }

    @Test
    void gcnConvNoDirectForwardThrows() {
        GCNConv gcn = new GCNConv(4, 4);
        assertThrows(UnsupportedOperationException.class,
            () -> gcn.forward(GradTensor.randn(3, 4)));
    }

    // ── CTCLoss ───────────────────────────────────────────────────────────

    @Test
    void ctcLossIsScalar() {
        CTCLoss loss = new CTCLoss();
        // T=10, N=2, C=5 (including blank=0)
        GradTensor logProbs = GradTensor.randn(10, 2, 5).softmax().log();
        GradTensor targets  = GradTensor.of(new float[]{1, 2, 1, 3}, 2, 2);
        int[] inputLens  = {10, 10};
        int[] targetLens = {2, 2};
        GradTensor l = loss.forward(logProbs, targets, inputLens, targetLens);
        assertEquals(0, l.ndim());
        assertTrue(l.item() > 0f, "CTC loss should be positive");
    }

    @Test
    void ctcLossFiniteValue() {
        CTCLoss loss = new CTCLoss();
        GradTensor logProbs = GradTensor.randn(8, 1, 4).softmax().log();
        GradTensor targets  = GradTensor.of(new float[]{1, 2}, 1, 2);
        int[] inputLens  = {8};
        int[] targetLens = {2};
        float l = loss.forward(logProbs, targets, inputLens, targetLens).item();
        assertTrue(Float.isFinite(l), "CTC loss should be finite");
    }

    // ── PythonBridge ──────────────────────────────────────────────────────

    @Test
    void pythonBridgeLoadSafeTensorsRoundTrip(@TempDir Path tmpDir) throws IOException {
        // Save a state dict, then load it via PythonBridge
        Sequential model = new Sequential(new Linear(4, 2));
        Path file = tmpDir.resolve("model.safetensors");
        StateDict.save(model.stateDict(), file);

        Map<String, GradTensor> loaded = PythonBridge.loadSafeTensors(file);
        assertEquals(model.stateDict().keySet(), loaded.keySet());
    }

    @Test
    void pythonBridgeLoadIntoModel(@TempDir Path tmpDir) throws IOException {
        Sequential model1 = new Sequential(new Linear(4, 2));
        Sequential model2 = new Sequential(new Linear(4, 2));
        Path file = tmpDir.resolve("weights.safetensors");
        StateDict.save(model1.stateDict(), file);

        PythonBridge.loadIntoModel(model2, file, true);
        for (String key : model1.stateDict().keySet())
            assertArrayEquals(model1.stateDict().get(key).data(),
                              model2.stateDict().get(key).data(), 1e-5f);
    }

    // ── BenchmarkSuite ────────────────────────────────────────────────────

    @Test
    void benchmarkSuiteRuns() {
        BenchmarkSuite suite = new BenchmarkSuite(1, 3); // fast for tests
        suite.run();
        assertFalse(suite.results().isEmpty());
    }

    @Test
    void benchmarkResultHasPositiveMean() {
        BenchmarkSuite suite = new BenchmarkSuite(1, 3);
        suite.benchMatmul(64, 64, 64);
        BenchmarkSuite.BenchmarkResult r = suite.results().get(0);
        assertTrue(r.meanMs() > 0);
        assertTrue(r.throughput() > 0);
    }

    @Test
    void benchmarkModelForward() {
        BenchmarkSuite suite = new BenchmarkSuite(1, 3);
        NNModule model = new Sequential(new Linear(8, 4), new ReLU(), new Linear(4, 2));
        suite.benchModel("MLP forward", model, GradTensor.randn(16, 8));
        assertEquals(1, suite.results().size());
    }

    @Test
    void benchmarkCustomOp() {
        BenchmarkSuite suite = new BenchmarkSuite(1, 5);
        float[] data = new float[10_000];
        suite.measure("custom sum", () -> tech.kayys.gollek.ml.tensor.VectorOps.sum(data), data.length);
        assertTrue(suite.results().get(0).meanMs() >= 0);
    }
}
