package tech.kayys.gollek.sdk.model;

import java.util.function.Consumer;

/**
 * Handle for a streaming generation session.
 *
 * <p>Register listeners before the stream starts to avoid missing early tokens.
 * The stream is automatically cancelled when {@link #close()} is called, making
 * it safe to use in try-with-resources blocks.
 *
 * <pre>{@code
 * try (GenerationStream stream = client.generateStream(request)) {
 *     stream.onToken(System.out::print)
 *           .onComplete(() -> System.out.println("\n[done]"))
 *           .onError(Throwable::printStackTrace);
 * }
 * }</pre>
 */
public interface GenerationStream extends AutoCloseable {

    /**
     * Registers a listener that receives each generated token as it arrives.
     *
     * @param tokenListener callback invoked with each token string
     * @return {@code this} for chaining
     */
    GenerationStream onToken(Consumer<String> tokenListener);

    /**
     * Registers a listener that is invoked when generation completes successfully.
     *
     * @param completeListener callback invoked on normal completion
     * @return {@code this} for chaining
     */
    GenerationStream onComplete(Runnable completeListener);

    /**
     * Registers a listener that is invoked if generation fails with an error.
     *
     * @param errorListener callback invoked with the thrown exception
     * @return {@code this} for chaining
     */
    GenerationStream onError(Consumer<Throwable> errorListener);

    /**
     * Cancels the ongoing generation session.
     * After cancellation, no further token, completion, or error events are delivered.
     */
    void cancel();

    /**
     * Cancels the stream; equivalent to {@link #cancel()}.
     */
    @Override
    default void close() {
        cancel();
    }
}
