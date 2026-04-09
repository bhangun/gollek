package tech.kayys.gollek.ml.metrics;

import tech.kayys.gollek.ml.tensor.VectorOps;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Lightweight training metrics tracker — logs scalar values per step and
 * provides aggregation, CSV export, and summary utilities.
 *
 * <p>Designed to be used inside a training loop as a drop-in replacement
 * for TensorBoard's scalar logging, with optional CSV export for offline analysis.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * MetricsTracker tracker = new MetricsTracker();
 *
 * for (int epoch = 0; epoch < 100; epoch++) {
 *     float loss = trainStep();
 *     tracker.log("train/loss", loss, epoch);
 *     tracker.log("train/lr",   scheduler.getLr(), epoch);
 * }
 *
 * System.out.println("Best loss: " + tracker.min("train/loss"));
 * tracker.exportCsv(Path.of("metrics.csv"));
 * }</pre>
 */
public final class MetricsTracker {

    /**
     * Internal storage: metric name → list of {@code [step, value]} pairs.
     * Using {@link LinkedHashMap} to preserve insertion order for CSV export.
     */
    private final Map<String, List<float[]>> history = new LinkedHashMap<>();

    // ── Logging ───────────────────────────────────────────────────────────

    /**
     * Logs a single scalar value at the given step.
     *
     * @param name  metric name (e.g. {@code "train/loss"}, {@code "val/accuracy"})
     * @param value scalar value to record
     * @param step  training step or epoch number
     */
    public void log(String name, float value, int step) {
        history.computeIfAbsent(name, k -> new ArrayList<>())
               .add(new float[]{step, value});
    }

    /**
     * Logs multiple metrics at the same step in a single call.
     *
     * @param metrics map of metric name → value
     * @param step    training step or epoch number
     */
    public void logAll(Map<String, Float> metrics, int step) {
        metrics.forEach((name, value) -> log(name, value, step));
    }

    // ── Retrieval ─────────────────────────────────────────────────────────

    /**
     * Returns the full history for a metric as a list of {@code [step, value]} arrays.
     *
     * @param name metric name
     * @return unmodifiable list of {@code float[2]} entries, or empty list if not found
     */
    public List<float[]> get(String name) {
        return Collections.unmodifiableList(history.getOrDefault(name, List.of()));
    }

    /**
     * Returns the most recently logged value for a metric.
     *
     * @param name metric name
     * @return latest value, or {@link Float#NaN} if no data
     */
    public float latest(String name) {
        List<float[]> h = history.get(name);
        return (h == null || h.isEmpty()) ? Float.NaN : h.get(h.size() - 1)[1];
    }

    /**
     * Returns the minimum logged value for a metric.
     * Uses {@link VectorOps} for the scan when the series is large.
     *
     * @param name metric name
     * @return minimum value, or {@link Float#NaN} if no data
     */
    public float min(String name) {
        float[] vals = values(name);
        if (vals.length == 0) return Float.NaN;
        float min = Float.MAX_VALUE;
        for (float v : vals) if (v < min) min = v;
        return min;
    }

    /**
     * Returns the maximum logged value for a metric.
     *
     * @param name metric name
     * @return maximum value, or {@link Float#NaN} if no data
     */
    public float max(String name) {
        float[] vals = values(name);
        return vals.length == 0 ? Float.NaN : VectorOps.max(vals);
    }

    /**
     * Returns the mean of all logged values for a metric.
     * Uses {@link VectorOps#sum} for SIMD-accelerated summation.
     *
     * @param name metric name
     * @return mean value, or {@link Float#NaN} if no data
     */
    public float mean(String name) {
        float[] vals = values(name);
        return vals.length == 0 ? Float.NaN : VectorOps.sum(vals) / vals.length;
    }

    /**
     * Returns a summary map of the latest value per tracked metric.
     *
     * @return map of metric name → latest value
     */
    public Map<String, Float> summary() {
        Map<String, Float> s = new LinkedHashMap<>();
        history.keySet().forEach(name -> s.put(name, latest(name)));
        return Collections.unmodifiableMap(s);
    }

    // ── Export ────────────────────────────────────────────────────────────

    /**
     * Exports all tracked metrics to a CSV file with columns {@code step,name,value}.
     *
     * @param path output file path
     * @throws IOException if the file cannot be written
     */
    public void exportCsv(Path path) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("step,name,value");
        for (var entry : history.entrySet()) {
            for (float[] sv : entry.getValue()) {
                lines.add((int) sv[0] + "," + entry.getKey() + "," + sv[1]);
            }
        }
        Files.write(path, lines);
    }

    /**
     * Clears all tracked metrics.
     */
    public void reset() {
        history.clear();
    }

    /** Returns the set of all tracked metric names. */
    public Set<String> metricNames() {
        return Collections.unmodifiableSet(history.keySet());
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /** Extracts just the value column from the history of a metric. */
    private float[] values(String name) {
        List<float[]> h = history.get(name);
        if (h == null || h.isEmpty()) return new float[0];
        float[] vals = new float[h.size()];
        for (int i = 0; i < h.size(); i++) vals[i] = h.get(i)[1];
        return vals;
    }
}
