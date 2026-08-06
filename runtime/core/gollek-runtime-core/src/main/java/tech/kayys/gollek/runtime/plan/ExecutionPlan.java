package tech.kayys.gollek.runtime.plan;

import tech.kayys.gollek.ir.GValueId;
import java.util.*;

/**
 * 
 * Input:
 * Optimized GGraph
 * Output:
 * ExecutionPlan
 * ├── ordered steps
 * ├── tensor allocation plan
 * └── reuse slots
 * 
 */
public final class ExecutionPlan {
    public final List<PlanStep> steps;
    // memory slot mapping
    public final Map<GValueId, Integer> slotMap;

    public ExecutionPlan(List<PlanStep> steps,
            Map<GValueId, Integer> slotMap) {
        this.steps = steps;
        this.slotMap = slotMap;
    }
}