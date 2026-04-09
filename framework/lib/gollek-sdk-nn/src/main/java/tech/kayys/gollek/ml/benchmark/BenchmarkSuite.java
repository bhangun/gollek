package tech.kayys.gollek.ml.benchmark;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.*;

/**
 * Benchmark suite — measures throughput and latency of core operations
 * and compares against reference baselines.
 *
 * <p>Benchmarks are run with JVM warmup to ensure JIT compilation.
 * Results include mean, std, min, max, and throughput (ops/sec or tokens/sec).
 *
 * <h3>Example</h3>
 * <pre>{@code
 * BenchmarkSuite suite = new BenchmarkSuite();
 * suite.run();
 * suite.printReport();
 * }</pre>
 */
public final class BenchmarkSuite {

    private final int warmupRuns;
    private final int measureRuns;
    private final List<BenchmarkResult> results = new ArrayList<>();

    /**
     * Creates a benchmark suite with default 5 warmup and 20 measure runs.
     */
    public BenchmarkSuite() { this(5, 20); }

    /**
     * Creates a benchmark suite with custom run counts.
     *
     * @param warmupRuns  JVM warmup iterations (not measured)
     * @param measureRuns measured iterations
     */
    public BenchmarkSuite(int warmupRuns, int measureRuns) {
        this.warmupRuns  = warmupRuns;
        this.measureRuns = measureRuns;
    }

    /**
     * Immutable benchmark result.
     *
     * @param name       benchmark name
     * @param meanMs     mean latency in milliseconds
     * @param stdMs      standard deviation in milliseconds
     * @param minMs      minimum latency
     * @param maxMs      maximum latency
     * @param throughput operations per second (or tokens/sec for generation)
     */
    public record BenchmarkResult(String name, double meanMs, double stdMs,
                                   double minMs, double maxMs, double throughput) {
        @Override public String toString() {
            return String.format("%-40s  mean=%7.3fms  std=%6.3fms  throughput=%,.0f ops/s",
                name, meanMs, stdMs, throughput);
        }
    }

    // ── Built-in benchmarks ───────────────────────────────────────────────

    /**
     * Runs all built-in benchmarks: matmul, VectorOps, model forward pass.
     */
    public void run() {
        benchMatmul(512, 512, 512);
        benchMatmul(1024, 1024, 1024);
        benchVectorOpsAdd(1_000_000);
        benchVectorOpsSum(1_000_000);
    }

    /**
     * Benchmarks matrix multiplication of shape [M, K] × [K, N].
     *
     * @param M rows of A
     * @param K inner dimension
     * @param N columns of B
     */
    public void benchMatmul(int M, int K, int N) {
        float[] a = new float[M * K], b = new float[K * N];
        new Random().nextLong(); // seed
        String name = String.format("matmul [%d,%d]×[%d,%d]", M, K, K, N);
        measure(name, () -> VectorOps.matmul(a, b, M, K, N), M * N);
    }

    /**
     * Benchmarks element-wise addition of two float arrays.
     *
     * @param size array length
     */
    public void benchVectorOpsAdd(int size) {
        float[] a = new float[size], b = new float[size], out = new float[size];
        measure("VectorOps.add [" + size + "]", () -> VectorOps.add(a, b, out), size);
    }

    /**
     * Benchmarks reduction sum of a float array.
     *
     * @param size array length
     */
    public void benchVectorOpsSum(int size) {
        float[] a = new float[size];
        measure("VectorOps.sum [" + size + "]", () -> VectorOps.sum(a), size);
    }

    /**
     * Benchmarks a model's forward pass.
     *
     * @param name  benchmark label
     * @param model model to benchmark
     * @param input sample input tensor
     */
    public void benchModel(String name, NNModule model, GradTensor input) {
        model.eval();
        measure(name, () -> model.forward(input), (int) input.numel());
    }

    /**
     * Runs a custom benchmark.
     *
     * @param name      benchmark label
     * @param op        operation to benchmark (called warmup+measure times)
     * @param opsPerRun number of operations per call (for throughput calculation)
     */
    public void measure(String name, Runnable op, long opsPerRun) {
        // Warmup
        for (int i = 0; i < warmupRuns; i++) op.run();

        // Measure
        double[] times = new double[measureRuns];
        for (int i = 0; i < measureRuns; i++) {
            long start = System.nanoTime();
            op.run();
            times[i] = (System.nanoTime() - start) / 1e6; // ms
        }

        double mean = Arrays.stream(times).average().orElse(0);
        double variance = Arrays.stream(times).map(t -> (t - mean) * (t - mean)).average().orElse(0);
        double std  = Math.sqrt(variance);
        double min  = Arrays.stream(times).min().orElse(0);
        double max  = Arrays.stream(times).max().orElse(0);
        double throughput = mean > 0 ? opsPerRun / (mean / 1000.0) : 0;

        results.add(new BenchmarkResult(name, mean, std, min, max, throughput));
    }

    /**
     * Returns all benchmark results.
     *
     * @return unmodifiable list of results
     */
    public List<BenchmarkResult> results() { return Collections.unmodifiableList(results); }

    /**
     * Prints a formatted benchmark report to stdout.
     */
    public void printReport() {
        System.out.println("=".repeat(80));
        System.out.println("Gollek Benchmark Suite");
        System.out.println("=".repeat(80));
        results.forEach(System.out::println);
        System.out.println("=".repeat(80));
    }
}
