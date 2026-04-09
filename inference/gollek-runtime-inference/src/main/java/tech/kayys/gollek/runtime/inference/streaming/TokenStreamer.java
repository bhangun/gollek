package tech.kayys.gollek.runtime.inference.streaming;

/**
 * Callback interface for streaming token-by-token generation.
 * <p>
 * Each generated token is pushed to the streamer as soon as it's produced,
 * enabling real-time display (like ChatGPT-style streaming).
 */
public interface TokenStreamer {

    /**
     * Called when a new token is generated.
     *
     * @param tokenId the generated token ID
     * @param text    decoded text for this token
     */
    void onToken(int tokenId, String text);

    /** Called when generation is complete (EOS or max tokens reached). */
    void onComplete();

    /** Called if an error occurs during generation. */
    void onError(Throwable t);
}
