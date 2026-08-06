package tech.kayys.gollek.runtime.control;

import java.util.concurrent.atomic.AtomicReference;

public final class ExecutionController {
    private final AtomicReference<ExecutionState> state = new AtomicReference<>(ExecutionState.RUNNING);

    public void pause() {
        state.compareAndSet(ExecutionState.RUNNING, ExecutionState.PAUSED);
    }

    public void resume() {
        state.compareAndSet(ExecutionState.PAUSED, ExecutionState.RUNNING);
    }

    public void cancel() {
        state.set(ExecutionState.CANCELLED);
    }

    public void shutdown() {
        state.set(ExecutionState.SHUTDOWN);
    }

    public ExecutionState state() {
        return state.get();
    }

    public boolean isCancelled() {
        return state.get() == ExecutionState.CANCELLED;
    }

    public boolean isPaused() {
        return state.get() == ExecutionState.PAUSED;
    }

    public boolean isShutdown() {
        return state.get() == ExecutionState.SHUTDOWN;
    }
}