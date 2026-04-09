package tech.kayys.gollek.runtime.graph;

import java.util.Collections;
import java.util.List;

/**
 * A topologically sorted list of graph nodes ready for execution.
 * <p>
 * Produced by {@link GraphPlanner} and consumed by {@link GraphExecutor}.
 * Optimization passes ({@link FusionOptimizer}, {@link LifetimeAnalyzer})
 * operate on this plan in-place before execution.
 */
public final class ExecutionPlan {

    public final List<GraphNode> orderedNodes;

    public ExecutionPlan(List<GraphNode> orderedNodes) {
        this.orderedNodes = orderedNodes;
    }

    /** Number of operations in the plan. */
    public int size() {
        return orderedNodes.size();
    }

    @Override
    public String toString() {
        return "ExecutionPlan[steps=" + orderedNodes.size() + "]";
    }
}
