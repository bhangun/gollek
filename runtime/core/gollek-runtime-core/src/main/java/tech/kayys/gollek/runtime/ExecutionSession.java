package tech.kayys.gollek.runtime;

import tech.kayys.gollek.runtime.control.ExecutionController;

public final class ExecutionSession {
    public final ExecutionController controller;

    public ExecutionSession(ExecutionController controller) {
        this.controller = controller;
    }
}