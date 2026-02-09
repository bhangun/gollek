package tech.kayys.golek.converter.dto;

import com.fasterxml.jackson.annotation.JsonInclude;

import tech.kayys.golek.converter.model.ConversionProgress;

/**
 * Progress update DTO for streaming.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ProgressUpdate {

    private String type = "progress";
    private long conversionId;
    private float progress;
    private int progressPercent;
    private String stage;
    private long timestamp;
    private boolean complete;

    public ProgressUpdate(long conversionId, float progress, int progressPercent, String stage, long timestamp,
            boolean complete) {
        this.conversionId = conversionId;
        this.progress = progress;
        this.progressPercent = progressPercent;
        this.stage = stage;
        this.timestamp = timestamp;
        this.complete = complete;
    }

    public long getConversionId() {
        return conversionId;
    }

    public float getProgress() {
        return progress;
    }

    public int getProgressPercent() {
        return progressPercent;
    }

    public String getStage() {
        return stage;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public boolean isComplete() {
        return complete;
    }

    static ProgressUpdate fromProgress(ConversionProgress progress) {
        return ProgressUpdate.builder()
                .conversionId(progress.getConversionId())
                .progress(progress.getProgress())
                .progressPercent(progress.getProgressPercent())
                .stage(progress.getStage())
                .timestamp(progress.getTimestamp())
                .complete(progress.isComplete())
                .build();
    }
}
