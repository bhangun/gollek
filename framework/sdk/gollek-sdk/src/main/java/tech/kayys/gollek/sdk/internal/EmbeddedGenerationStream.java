package tech.kayys.gollek.sdk.internal;

import tech.kayys.gollek.sdk.api.GollekClient;
import tech.kayys.gollek.sdk.api.GollekClient.GenerationResult;

import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

public class EmbeddedGenerationStream implements GollekClient.GenerationStream {
    private Consumer<String> tokenHandler = t -> {};
    private Consumer<GenerationResult> completeHandler = r -> {};
    private Consumer<Throwable> errorHandler = e -> {};
    private final StringBuilder contentBuilder = new StringBuilder();
    private final long startTime = System.currentTimeMillis();
    private int tokenCount = 0;

    @Override
    public GollekClient.GenerationStream onToken(Consumer<String> handler) {
        this.tokenHandler = handler;
        return this;
    }

    @Override
    public GollekClient.GenerationStream onComplete(Consumer<GenerationResult> handler) {
        this.completeHandler = handler;
        return this;
    }

    @Override
    public GollekClient.GenerationStream onError(Consumer<Throwable> handler) {
        this.errorHandler = handler;
        return this;
    }

    @Override
    public CompletableFuture<GenerationResult> toFuture() {
        CompletableFuture<GenerationResult> future = new CompletableFuture<>();
        this.onComplete(future::complete);
        this.onError(future::completeExceptionally);
        return future;
    }

    public void emitToken(String token) {
        contentBuilder.append(token);
        tokenCount++;
        tokenHandler.accept(token);
    }

    public void emitComplete() {
        GenerationResult result = new GenerationResult(
                contentBuilder.toString(),
                tokenCount,
                0, // prompt tokens not tracked in basic stream yet
                System.currentTimeMillis() - startTime
        );
        completeHandler.accept(result);
    }

    public void emitError(Throwable error) {
        errorHandler.accept(error);
    }
}
