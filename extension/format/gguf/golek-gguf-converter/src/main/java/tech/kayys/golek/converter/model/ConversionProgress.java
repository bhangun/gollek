package tech.kayys.golek.converter.model;

/**
 * Progress update for model conversion.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class ConversionProgress {

    /**
     * Conversion ID.
     */
    private final long conversionId;

    /**
     * Progress percentage (0.0 - 1.0).
     */
    private final float progress;

    /**
     * Current stage description.
     */
    private final String stage;

    /**
     * Timestamp of this progress update.
     */
    private final long timestamp;

    public ConversionProgress(long conversionId, float progress, String stage, long timestamp) {
        this.conversionId = conversionId;
        this.progress = progress;
        this.stage = stage;
        this.timestamp = timestamp;
    }

    public long getConversionId() {
        return conversionId;
    }

    public float getProgress() {
        return progress;
    }

    public String getStage() {
        return stage;
    }

    public long getTimestamp() {
        return timestamp;
    }

    /**
     * Get progress as percentage (0-100).
     *
     * @return percentage
     */
    public int getProgressPercent() {
        return Math.round(progress * 100);
    }

    /**
     * Check if conversion is complete.
     *
     * @return true if progress >= 1.0
     */
    public boolean isComplete() {
        return progress >= 1.0f;
    }
    
    /**
     * Builder class for ConversionProgress.
     */
    public static class Builder {
        private long conversionId;
        private float progress;
        private String stage;
        private long timestamp;

        public Builder conversionId(long conversionId) {
            this.conversionId = conversionId;
            return this;
        }

        public Builder progress(float progress) {
            this.progress = progress;
            return this;
        }

        public Builder stage(String stage) {
            this.stage = stage;
            return this;
        }

        public Builder timestamp(long timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public ConversionProgress build() {
            return new ConversionProgress(conversionId, progress, stage, timestamp);
        }
    }

    public static Builder builder() {
        return new Builder();
    }
}
