package tech.kayys.golek.model.download;

/**
 * Progress listener for downloads
 */
public interface DownloadProgressListener {

    /**
     * Called when progress is made
     * 
     * @param downloadedBytes bytes downloaded so far
     * @param totalBytes      total bytes to download (-1 if unknown)
     * @param progress        progress as decimal (0.0 to 1.0)
     */
    void onProgress(long downloadedBytes, long totalBytes, double progress);

    /**
     * Called when download starts
     */
    default void onStart(long totalBytes) {
    }

    /**
     * Called when download completes
     */
    default void onComplete(long totalBytes) {
    }

    /**
     * Called when download fails
     */
    default void onError(Throwable error) {
    }
}