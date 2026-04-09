package tech.kayys.gollek.sdk.export;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Model benchmark for measuring inference performance.
 *
 * <p>Provides tools to benchmark model latency, throughput, and memory usage.</p>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * Benchmark benchmark = new Benchmark(model);
 * BenchmarkResult result = benchmark.run(inputShape, iterations=100, warmup=10);
 *
 * System.out.println("Avg latency: " + result.avgLatencyMs() + "ms");
 * System.out.println("Throughput: " + result.throughput() + " inf/sec");
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class Benchmark {

    private final Object model;

    /**
     * Create benchmark for a model.
     *
     * @param model model to benchmark
     */
    public Benchmark(Object model) {
        this.model = model;
    }

    /**
     * Run benchmark.
     *
     * @param inputShape  input shape
     * @param iterations  number of iterations
     * @param warmup      number of warmup iterations
     * @return benchmark result
     */
    public BenchmarkResult run(long[] inputShape, int iterations, int warmup) {
        List<Double> latencies = new ArrayList<>();

        // Warmup phase
        for (int i = 0; i < warmup; i++) {
            runInference(inputShape);
        }

        // Benchmark phase
        long totalStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            runInference(inputShape);
            long elapsed = System.nanoTime() - start;
            latencies.add(elapsed / 1_000_000.0);  // Convert to ms
        }
        long totalElapsed = System.nanoTime() - totalStart;

        // Compute statistics
        double avgLatency = latencies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double p50 = getPercentile(latencies, 50);
        double p95 = getPercentile(latencies, 95);
        double p99 = getPercentile(latencies, 99);
        double throughput = iterations * 1e9 / totalElapsed;

        return new BenchmarkResult(
            avgLatency,
            p50,
            p95,
            p99,
            throughput,
            iterations,
            inputShape
        );
    }

    /**
     * Run benchmark with default settings.
     *
     * @param inputShape input shape
     * @return benchmark result
     */
    public BenchmarkResult run(long[] inputShape) {
        return run(inputShape, 100, 10);
    }

    /**
     * Run a single inference (placeholder).
     *
     * @param inputShape input shape
     */
    private void runInference(long[] inputShape) {
        if (model instanceof tech.kayys.gollek.ml.nn.NNModule module) {
            GradTensor input = GradTensor.randn(inputShape);
            module.eval();
            module.forward(input);
        }
        // For non-Module models, no-op (timing still measured)
    }

    /**
     * Get percentile value from a list.
     *
     * @param values    list of values
     * @param percentile percentile (0-100)
     * @return percentile value
     */
    private double getPercentile(List<Double> values, double percentile) {
        List<Double> sorted = new ArrayList<>(values);
        sorted.sort(Double::compareTo);
        int index = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, index));
    }

    /**
     * Benchmark result record.
     *
     * @param avgLatencyMs  average latency in milliseconds
     * @param p50LatencyMs  P50 latency in milliseconds
     * @param p95LatencyMs  P95 latency in milliseconds
     * @param p99LatencyMs  P99 latency in milliseconds
     * @param throughput    throughput in inferences per second
     * @param iterations    number of iterations
     * @param inputShape    input shape used
     */
    public record BenchmarkResult(
            double avgLatencyMs,
            double p50LatencyMs,
            double p95LatencyMs,
            double p99LatencyMs,
            double throughput,
            int iterations,
            long[] inputShape
    ) {
        @Override
        public String toString() {
            return String.format(
                "BenchmarkResult{avg=%.2fms, p50=%.2fms, p95=%.2fms, p99=%.2fms, throughput=%.1f inf/s, iterations=%d}",
                avgLatencyMs, p50LatencyMs, p95LatencyMs, p99LatencyMs, throughput, iterations
            );
        }
    }
}
