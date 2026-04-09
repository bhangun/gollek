package tech.kayys.gollek.sdk.train;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Training metrics tracker.
 *
 * <p>Records and provides access to training metrics including losses,
 * learning rates, and custom metrics.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * TrainingMetrics metrics = trainer.getMetrics();
 *
 * double bestValLoss = metrics.getBestValLoss();
 * int bestEpoch = metrics.getBestValLossEpoch();
 * List<Double> trainLosses = metrics.getTrainLosses();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class TrainingMetrics {

    private final Map<Integer, Double> trainLosses = new ConcurrentHashMap<>();
    private final Map<Integer, Double> valLosses = new ConcurrentHashMap<>();
    private final Map<Integer, Double> learningRates = new ConcurrentHashMap<>();
    private final Map<Integer, Double> batchLosses = new ConcurrentHashMap<>();
    private final Map<String, Map<Integer, Double>> customMetrics = new ConcurrentHashMap<>();

    private final long startTime = System.currentTimeMillis();
    private long endTime = 0;

    /**
     * Update training loss for an epoch.
     *
     * @param epoch epoch number
     * @param loss  average training loss
     */
    public void updateTrainLoss(int epoch, double loss) {
        trainLosses.put(epoch, loss);
    }

    /**
     * Update validation loss for an epoch.
     *
     * @param epoch epoch number
     * @param loss  average validation loss
     */
    public void updateValLoss(int epoch, double loss) {
        valLosses.put(epoch, loss);
    }

    /**
     * Update learning rate for an epoch.
     *
     * @param epoch epoch number
     * @param lr    learning rate
     */
    public void updateLearningRate(int epoch, double lr) {
        learningRates.put(epoch, lr);
    }

    /**
     * Update batch loss.
     *
     * @param step global step
     * @param loss batch loss
     */
    public void updateBatchLoss(int step, double loss) {
        batchLosses.put(step, loss);
    }

    /**
     * Update a custom metric.
     *
     * @param name  metric name
     * @param epoch epoch number
     * @param value metric value
     */
    public void updateCustomMetric(String name, int epoch, double value) {
        customMetrics.computeIfAbsent(name, k -> new ConcurrentHashMap<>())
                .put(epoch, value);
    }

    /**
     * Get training loss for an epoch.
     *
     * @param epoch epoch number
     * @return training loss, or null if not recorded
     */
    public Double getTrainLoss(int epoch) {
        return trainLosses.get(epoch);
    }

    /**
     * Get validation loss for an epoch.
     *
     * @param epoch epoch number
     * @return validation loss, or null if not recorded
     */
    public Double getValLoss(int epoch) {
        return valLosses.get(epoch);
    }

    /**
     * Get all training losses.
     *
     * @return list of training losses in epoch order
     */
    public List<Double> getTrainLosses() {
        return trainLosses.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(Map.Entry::getValue)
                .toList();
    }

    /**
     * Get all validation losses.
     *
     * @return list of validation losses in epoch order
     */
    public List<Double> getValLosses() {
        return valLosses.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(Map.Entry::getValue)
                .toList();
    }

    /**
     * Get best validation loss.
     *
     * @return best (lowest) validation loss, or Double.MAX_VALUE if none recorded
     */
    public double getBestValLoss() {
        return valLosses.values().stream()
                .min(Double::compare)
                .orElse(Double.MAX_VALUE);
    }

    /**
     * Get epoch with best validation loss.
     *
     * @return epoch number, or -1 if none recorded
     */
    public int getBestValLossEpoch() {
        return valLosses.entrySet().stream()
                .min(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(-1);
    }

    /**
     * Get latest training loss.
     *
     * @return latest training loss, or null if none recorded
     */
    public Double getLatestTrainLoss() {
        return trainLosses.entrySet().stream()
                .max(Map.Entry.comparingByKey())
                .map(Map.Entry::getValue)
                .orElse(null);
    }

    /**
     * Get latest validation loss.
     *
     * @return latest validation loss, or null if none recorded
     */
    public Double getLatestValLoss() {
        return valLosses.entrySet().stream()
                .max(Map.Entry.comparingByKey())
                .map(Map.Entry::getValue)
                .orElse(null);
    }

    /**
     * Get number of epochs recorded.
     *
     * @return epoch count
     */
    public int getEpochCount() {
        return trainLosses.size();
    }

    /**
     * Get batch loss at a specific step.
     *
     * @param step global step
     * @return batch loss, or null if not recorded
     */
    public Double getBatchLoss(int step) {
        return batchLosses.get(step);
    }

    /**
     * Get all batch losses.
     *
     * @return map of step → loss
     */
    public Map<Integer, Double> getBatchLosses() {
        return Collections.unmodifiableMap(batchLosses);
    }

    /**
     * Get custom metric values.
     *
     * @param name metric name
     * @return map of epoch → value, or empty map if not recorded
     */
    public Map<Integer, Double> getCustomMetric(String name) {
        return customMetrics.getOrDefault(name, Collections.emptyMap());
    }

    /**
     * Get training duration in milliseconds.
     *
     * @return duration in ms
     */
    public long getDurationMs() {
        return endTime > 0 ? endTime - startTime : System.currentTimeMillis() - startTime;
    }

    /**
     * Mark training as ended.
     */
    public void markEnded() {
        this.endTime = System.currentTimeMillis();
    }

    /**
     * Get all recorded metrics as a summary map.
     *
     * @return summary map
     */
    public Map<String, Object> getSummary() {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("epochs", getEpochCount());
        summary.put("best_val_loss", getBestValLoss());
        summary.put("best_val_loss_epoch", getBestValLossEpoch());
        summary.put("latest_train_loss", getLatestTrainLoss());
        summary.put("latest_val_loss", getLatestValLoss());
        summary.put("duration_ms", getDurationMs());
        summary.put("duration_sec", getDurationMs() / 1000.0);
        return summary;
    }

    @Override
    public String toString() {
        return String.format("TrainingMetrics{epochs=%d, best_val_loss=%.6f, duration=%.1fs}",
                getEpochCount(), getBestValLoss(), getDurationMs() / 1000.0);
    }
}
