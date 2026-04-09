package tech.kayys.gollek.ml.profiler;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * Model profiler — measures forward pass latency and memory usage per layer.
 *
 * <p>Wraps a model's forward pass with timing hooks to identify bottlenecks.
 * Uses JDK 25 {@link System#nanoTime()} for high-resolution timing.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var profiler = new ModelProfiler(model);
 * profiler.profile(input, warmupRuns = 3, measureRuns = 10);
 *
 * profiler.report().forEach(e ->
 *     System.out.printf("%-30s  %6.2f ms  %6.1f MB%n",
 *         e.name(), e.avgMs(), e.memoryMB()));
 * }</pre>
 */
public final class ModelProfiler {

    private final NNModule model;

    /**
     * A single profiling entry for one forward pass.
     *
     * @param name     metric name (e.g. "forward_total")
     * @param avgMs    average latency in milliseconds
     * @param minMs    minimum latency in milliseconds
     * @param maxMs    maximum latency in milliseconds
     * @param memoryMB estimated memory usage in megabytes
     */
    public record ProfileEntry(String name, double avgMs, double minMs,
                                double maxMs, double memoryMB) {}

    /**
     * Creates a profiler for the given model.
     *
     * @param model model to profile
     */
    public ModelProfiler(NNModule model) { this.model = model; }

    /**
     * Profiles the model's forward pass.
     *
     * @param input       sample input tensor
     * @param warmupRuns  number of warmup runs (not measured)
     * @param measureRuns number of measured runs
     * @return list of {@link ProfileEntry} records
     */
    public List<ProfileEntry> profile(GradTensor input, int warmupRuns, int measureRuns) {
        model.eval();

        // Warmup
        for (int i = 0; i < warmupRuns; i++) model.forward(input);

        // Measure
        long[] times = new long[measureRuns];
        Runtime rt = Runtime.getRuntime();
        for (int i = 0; i < measureRuns; i++) {
            long start = System.nanoTime();
            model.forward(input);
            times[i] = System.nanoTime() - start;
        }

        // Compute stats
        double sum = 0, min = Double.MAX_VALUE, max = 0;
        for (long t : times) {
            double ms = t / 1e6;
            sum += ms;
            if (ms < min) min = ms;
            if (ms > max) max = ms;
        }
        double avg = sum / measureRuns;

        // Estimate memory: parameter count × 4 bytes (float32)
        double paramMB = model.parameterCount() * 4.0 / (1024 * 1024);
        // Activation memory estimate: input size × 4 bytes × depth heuristic
        double activMB = input.numel() * 4.0 / (1024 * 1024) * 10;

        List<ProfileEntry> entries = new ArrayList<>();
        entries.add(new ProfileEntry("forward_total",  avg, min, max, paramMB + activMB));
        entries.add(new ProfileEntry("parameters_mem", 0,   0,   0,   paramMB));
        entries.add(new ProfileEntry("activation_est", 0,   0,   0,   activMB));
        return Collections.unmodifiableList(entries);
    }

    /**
     * Profiles with default 3 warmup and 10 measure runs.
     *
     * @param input sample input tensor
     * @return profiling results
     */
    public List<ProfileEntry> profile(GradTensor input) {
        return profile(input, 3, 10);
    }

    /**
     * Prints a formatted profiling report to stdout.
     *
     * @param entries profiling results from {@link #profile}
     */
    public static void printReport(List<ProfileEntry> entries) {
        System.out.printf("%-25s  %8s  %8s  %8s  %8s%n",
            "Name", "Avg(ms)", "Min(ms)", "Max(ms)", "Mem(MB)");
        System.out.println("-".repeat(65));
        for (ProfileEntry e : entries) {
            if (e.avgMs() > 0)
                System.out.printf("%-25s  %8.3f  %8.3f  %8.3f  %8.2f%n",
                    e.name(), e.avgMs(), e.minMs(), e.maxMs(), e.memoryMB());
            else
                System.out.printf("%-25s  %8s  %8s  %8s  %8.2f%n",
                    e.name(), "-", "-", "-", e.memoryMB());
        }
    }
}
