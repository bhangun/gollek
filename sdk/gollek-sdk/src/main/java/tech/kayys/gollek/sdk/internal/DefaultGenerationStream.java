package tech.kayys.gollek.sdk.internal;

import tech.kayys.gollek.sdk.model.GenerationStream;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Default in-memory implementation of {@link GenerationStream}.
 *
 * <p>Used by {@link LocalGollekClient} to bridge Mutiny reactive streams to the
 * listener-based {@link GenerationStream} API. Callers register listeners via
 * {@link #onToken}, {@link #onComplete}, and {@link #onError}, then the engine
 * drives the stream by calling {@link #emitToken}, {@link #emitComplete}, and
 * {@link #emitError}.
 *
 * <p>All listener lists use {@link CopyOnWriteArrayList} to allow safe concurrent
 * registration and emission.
 */
public class DefaultGenerationStream implements GenerationStream {

    private final List<Consumer<String>> tokenListeners = new CopyOnWriteArrayList<>();
    private final List<Runnable> completeListeners = new CopyOnWriteArrayList<>();
    private final List<Consumer<Throwable>> errorListeners = new CopyOnWriteArrayList<>();
    private volatile boolean cancelled = false;

    /**
     * Delivers a generated token to all registered token listeners.
     * No-op if the stream has been cancelled.
     *
     * @param token the next generated token
     */
    public void emitToken(String token) {
        if (!cancelled) {
            tokenListeners.forEach(l -> l.accept(token));
        }
    }

    /**
     * Notifies all registered completion listeners that generation has finished.
     * No-op if the stream has been cancelled.
     */
    public void emitComplete() {
        if (!cancelled) {
            completeListeners.forEach(Runnable::run);
        }
    }

    /**
     * Notifies all registered error listeners of a generation failure.
     * No-op if the stream has been cancelled.
     *
     * @param t the error that occurred
     */
    public void emitError(Throwable t) {
        if (!cancelled) {
            errorListeners.forEach(l -> l.accept(t));
        }
    }

    @Override
    public GenerationStream onToken(Consumer<String> tokenListener) {
        tokenListeners.add(tokenListener);
        return this;
    }

    @Override
    public GenerationStream onComplete(Runnable completeListener) {
        completeListeners.add(completeListener);
        return this;
    }

    @Override
    public GenerationStream onError(Consumer<Throwable> errorListener) {
        errorListeners.add(errorListener);
        return this;
    }

    @Override
    public void cancel() {
        this.cancelled = true;
    }
}
