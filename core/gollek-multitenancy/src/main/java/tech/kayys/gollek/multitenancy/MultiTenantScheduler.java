package tech.kayys.gollek.multitenancy;

import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.execution.ExecutionRequest;
import tech.kayys.gollek.runtime.orchestrator.ExecutionOrchestrator;

public final class MultiTenantScheduler {
    private final ExecutorService pool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    public Future<Map<String, Tensor>> submit(
            ExecutionOrchestrator orchestrator,
            ExecutionRequest req) {
        return pool.submit(() -> orchestrator.run(req.plan, req.inputs, req.session));
    }
}