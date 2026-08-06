package tech.kayys.gollek.runtime;

import tech.kayys.gollek.runtime.plan.ExecutionPlan;
import tech.kayys.gollek.runtime.batch.*;
import java.util.List;

public final class ServingLoop {
    private final ContinuousBatchScheduler scheduler;
    private final BatchDecodeEngine engine;

    public ServingLoop(ContinuousBatchScheduler scheduler,
            BatchDecodeEngine engine) {
        this.scheduler = scheduler;
        this.engine = engine;
    }

    public void run(ExecutionPlan plan) {
        while (true) {
            List<DecodeRequest> batch = scheduler.nextBatch(32);
            if (batch.isEmpty())
                continue;
            engine.step(batch, plan);
            scheduler.requeue(batch);
        }
    }
}