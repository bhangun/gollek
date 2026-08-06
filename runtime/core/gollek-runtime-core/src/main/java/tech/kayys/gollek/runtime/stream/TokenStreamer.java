package tech.kayys.gollek.runtime.stream;

/**
 * STREAMING DECODE (LLM CORE)
 * Instead of:
 * run full sequence → return
 * Do:
 * token → emit → token → emit → token → emit
 */
public interface TokenStreamer {
    void onToken(int tokenId);

    void onComplete();

    void onError(Throwable t);
}