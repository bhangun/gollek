package tech.kayys.gollek.runtime.plan;

import tech.kayys.gollek.ir.*;
import java.util.*;

public final class ExecutionPlanner {
    public ExecutionPlan plan(GGraph graph) {
        // 1. schedule
        List<GOp> ordered = Scheduler.schedule(graph);

        // 2. build steps
        List<PlanStep> steps = new ArrayList<>();
        for (GOp op : ordered) {
            List<GValueId> inputs = op.inputs().stream()
                    .map(GValueRef::id)
                    .toList();
            steps.add(new PlanStep(op, inputs, op.outputs()));
        }
        // 3. lifetime analysis
        var lifetimes = LifetimeAnalyzer.analyze(ordered);
        // 4. memory planning
        var slots = MemoryPlanner.plan(lifetimes);
        return new ExecutionPlan(steps, slots);
    }
}