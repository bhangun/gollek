package tech.kayys.gollek.runtime.control;

public final class ExecutionSession {
    public final ExecutionController controller;
    public final long startTime;

    public ExecutionSession(ExecutionController controller) {
        this.controller = controller;
        this.startTime = System.currentTimeMillis();
    }
}
