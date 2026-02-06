package tech.kayys.golek.sdk.core.model;

/**
 * Progress information for model pull operations.
 */
public final class PullProgress {

    private final String status;
    private final String digest;
    private final long total;
    private final long completed;
    private final int percentComplete;

    public PullProgress(String status, String digest, long total, long completed) {
        this.status = status;
        this.digest = digest;
        this.total = total;
        this.completed = completed;
        this.percentComplete = total > 0 ? (int) ((completed * 100) / total) : 0;
    }

    public String getStatus() {
        return status;
    }

    public String getDigest() {
        return digest;
    }

    public long getTotal() {
        return total;
    }

    public long getCompleted() {
        return completed;
    }

    public int getPercentComplete() {
        return percentComplete;
    }

    public boolean isComplete() {
        return total > 0 && completed >= total;
    }

    /**
     * Creates a progress bar string representation.
     */
    public String getProgressBar(int width) {
        int filled = (percentComplete * width) / 100;
        int empty = width - filled;
        return "█".repeat(filled) + "░".repeat(empty);
    }

    public static PullProgress of(String status) {
        return new PullProgress(status, null, 0, 0);
    }

    public static PullProgress of(String status, String digest, long total, long completed) {
        return new PullProgress(status, digest, total, completed);
    }

    @Override
    public String toString() {
        if (total > 0) {
            return String.format("%s [%s] %d%%", status, getProgressBar(20), percentComplete);
        }
        return status;
    }
}
