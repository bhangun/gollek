package tech.kayys.gollek.runtime.graph;

import java.util.List;

/**
 * Computes tensor lifetimes within an {@link ExecutionPlan}.
 * <p>
 * For each node, determines:
 * <ul>
 *   <li>{@code firstUse} — the index where the tensor is produced</li>
 *   <li>{@code lastUse} — the last index where the tensor is consumed as input</li>
 * </ul>
 * <p>
 * The {@link GraphMemoryPlanner} uses this information to reuse memory blocks
 * whose lifetimes don't overlap, reducing peak memory from O(n_nodes) to
 * O(max_concurrent_tensors).
 * <p>
 * This is the same technique used by XLA, TensorRT, and ONNX Runtime.
 */
public final class LifetimeAnalyzer {

    private LifetimeAnalyzer() {}

    /**
     * Analyze tensor lifetimes in-place on the execution plan.
     *
     * @param plan the execution plan with topologically sorted nodes
     */
    public static void analyze(ExecutionPlan plan) {
        List<GraphNode> nodes = plan.orderedNodes;

        for (int i = 0; i < nodes.size(); i++) {
            GraphNode node = nodes.get(i);

            // The tensor is produced at this index
            node.firstUse = i;
            node.lastUse = i;

            // Extend the lifetime of all input tensors
            for (GraphNode input : node.inputs) {
                input.lastUse = Math.max(input.lastUse, i);
            }
        }
    }
}
