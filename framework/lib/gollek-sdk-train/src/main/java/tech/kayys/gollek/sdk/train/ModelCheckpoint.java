package tech.kayys.gollek.sdk.train;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Model checkpoint callback.
 *
 * <p>Saves model checkpoints during training based on monitored metrics.
 * Supports keeping only the best K checkpoints to save disk space.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Callback checkpoint = ModelCheckpoint.builder()
 *     .dirPath(Path.of("checkpoints/"))
 *     .filename("model-{epoch}-{val_loss:.4f}.pt")
 *     .saveTopK(3)  // Keep only best 3 checkpoints
 *     .monitor("val_loss")
 *     .mode(EarlyStopping.Mode.MIN)
 *     .build();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class ModelCheckpoint implements Callback {

    private static final Logger log = LoggerFactory.getLogger(ModelCheckpoint.class);

    private final Path dirPath;
    private final String filenamePattern;
    private final int saveTopK;
    private final String monitor;
    private final EarlyStopping.Mode mode;

    private final List<CheckpointInfo> savedCheckpoints = new ArrayList<>();

    /**
     * Checkpoint metadata.
     */
    public record CheckpointInfo(
            Path path,
            int epoch,
            double metricValue
    ) implements Comparable<CheckpointInfo> {
        @Override
        public int compareTo(CheckpointInfo o) {
            return Double.compare(this.metricValue, o.metricValue);
        }
    }

    private ModelCheckpoint(Builder builder) {
        this.dirPath = builder.dirPath;
        this.filenamePattern = builder.filenamePattern;
        this.saveTopK = builder.saveTopK;
        this.monitor = builder.monitor;
        this.mode = builder.mode;

        // Create checkpoint directory
        try {
            Files.createDirectories(dirPath);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create checkpoint directory: " + dirPath, e);
        }
    }

    /**
     * Create a new builder.
     *
     * @return builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create checkpoint with default settings.
     *
     * @param dirPath checkpoint directory
     * @return configured callback
     */
    public static ModelCheckpoint at(Path dirPath) {
        return builder().dirPath(dirPath).build();
    }

    @Override
    public void onValidationEnd(Trainer trainer, int epoch, double valLoss) {
        double metricValue = valLoss;
        String filename = filenamePattern
                .replace("{epoch}", String.valueOf(epoch))
                .replace("{val_loss}", String.format("%.4f", valLoss));

        Path checkpointPath = dirPath.resolve(filename);

        try {
            saveCheckpoint(trainer, checkpointPath, epoch, metricValue);
            log.info("Checkpoint saved: {} (epoch={}, {}={:.4f})",
                    checkpointPath.getFileName(), epoch, monitor, metricValue);

            // Remove old checkpoints if saveTopK is set
            if (saveTopK > 0 && savedCheckpoints.size() > saveTopK) {
                removeOldCheckpoints();
            }
        } catch (IOException e) {
            log.error("Failed to save checkpoint: {}", checkpointPath, e);
        }
    }

    /**
     * Save a checkpoint.
     *
     * @param trainer trainer instance
     * @param path    checkpoint file path
     * @param epoch   current epoch
     * @param metric  monitored metric value
     * @throws IOException if save fails
     */
    private void saveCheckpoint(Trainer trainer, Path path, int epoch, double metric) throws IOException {
        Files.createDirectories(path.getParent());
        // Serialize model parameters using StateDict (SafeTensors format)
        Trainer.Model m = trainer.getModel();
        if (m instanceof tech.kayys.gollek.ml.nn.NNModule module) {
            tech.kayys.gollek.ml.nn.StateDict.save(module.stateDict(), path);
        } else {
            // Fallback: write checkpoint metadata as JSON
            String meta = String.format("{\"epoch\":%d,\"metric\":%.6f}", epoch, metric);
            Files.writeString(path, meta);
        }
        savedCheckpoints.add(new CheckpointInfo(path, epoch, metric));
    }

    /**
     * Remove old checkpoints keeping only the best K.
     */
    private void removeOldCheckpoints() {
        Comparator<CheckpointInfo> comparator = mode == EarlyStopping.Mode.MIN
                ? Comparator.comparingDouble(CheckpointInfo::metricValue)
                : Comparator.comparingDouble(CheckpointInfo::metricValue).reversed();

        savedCheckpoints.sort(comparator);

        // Remove worst checkpoints
        List<CheckpointInfo> toRemove = savedCheckpoints.subList(saveTopK, savedCheckpoints.size());
        for (CheckpointInfo info : new ArrayList<>(toRemove)) {
            try {
                Files.deleteIfExists(info.path());
                log.debug("Removed old checkpoint: {}", info.path().getFileName());
            } catch (IOException e) {
                log.warn("Failed to remove checkpoint: {}", info.path(), e);
            }
        }
        toRemove.clear();
    }

    /**
     * Get list of saved checkpoints.
     *
     * @return list of checkpoint info
     */
    public List<CheckpointInfo> getSavedCheckpoints() {
        return List.copyOf(savedCheckpoints);
    }

    /**
     * Get the best checkpoint path.
     *
     * @return path to best checkpoint, or null if none saved
     */
    public Path getBestCheckpoint() {
        if (savedCheckpoints.isEmpty()) {
            return null;
        }

        Comparator<CheckpointInfo> comparator = mode == EarlyStopping.Mode.MIN
                ? Comparator.comparingDouble(CheckpointInfo::metricValue)
                : Comparator.comparingDouble(CheckpointInfo::metricValue).reversed();

        return savedCheckpoints.stream()
                .min(comparator)
                .map(CheckpointInfo::path)
                .orElse(null);
    }

    /**
     * Builder for ModelCheckpoint.
     */
    public static class Builder {
        private Path dirPath = Path.of("checkpoints/");
        private String filenamePattern = "model-epoch-{epoch}-val_loss-{val_loss}.pt";
        private int saveTopK = 1;
        private String monitor = "val_loss";
        private EarlyStopping.Mode mode = EarlyStopping.Mode.MIN;

        private Builder() {}

        /**
         * Set checkpoint directory.
         *
         * @param dirPath directory path
         * @return this builder
         */
        public Builder dirPath(Path dirPath) {
            this.dirPath = dirPath;
            return this;
        }

        /**
         * Set filename pattern.
         *
         * <p>Available placeholders: {epoch}, {val_loss}</p>
         *
         * @param pattern filename pattern
         * @return this builder
         */
        public Builder filename(String pattern) {
            this.filenamePattern = pattern;
            return this;
        }

        /**
         * Set number of best checkpoints to keep (0 = keep all).
         *
         * @param saveTopK number of checkpoints
         * @return this builder
         */
        public Builder saveTopK(int saveTopK) {
            this.saveTopK = saveTopK;
            return this;
        }

        /**
         * Set metric to monitor.
         *
         * @param monitor metric name
         * @return this builder
         */
        public Builder monitor(String monitor) {
            this.monitor = monitor;
            return this;
        }

        /**
         * Set monitoring mode.
         *
         * @param mode MIN for loss, MAX for accuracy
         * @return this builder
         */
        public Builder mode(EarlyStopping.Mode mode) {
            this.mode = mode;
            return this;
        }

        /**
         * Build the ModelCheckpoint instance.
         *
         * @return configured callback
         */
        public ModelCheckpoint build() {
            return new ModelCheckpoint(this);
        }
    }
}
