package tech.kayys.gollek.runtime.orchestrator;

import tech.kayys.gollek.runtime.backend.*;
import tech.kayys.gollek.runtime.plan.*;
import tech.kayys.gollek.runtime.control.*;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.Map;

public final class ExecutionOrchestrator {
    private final ExecutionBackend backend;

    public ExecutionOrchestrator(ExecutionBackend backend) {
        this.backend = backend;
    }

    public Map<String, Tensor> run(
            ExecutionPlan plan,
            Map<String, Tensor> inputs,
            ExecutionSession session) {
        return backend.execute(plan, inputs, session);
    }
}