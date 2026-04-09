package tech.kayys.gollek.runtime.graph;

import java.util.List;

/**
 * Optimization pass that fuses adjacent operations in an {@link ExecutionPlan}.
 * <p>
 * Currently supports:
 * <ul>
 *   <li>{@code add + relu → add_relu}</li>
 *   <li>{@code matmul + relu → matmul_relu}</li>
 *   <li>{@code matmul + add → matmul_add} (bias fusion)</li>
 * </ul>
 * <p>
 * Fused operations reduce kernel launch overhead and enable backends to
 * use specialized fused kernels (e.g. CUDA fused_add_relu).
 */
public final class FusionOptimizer {

    private FusionOptimizer() {}

    /**
     * Apply fusion passes in-place to the execution plan.
     *
     * @param plan the plan to optimize
     * @return number of fusions applied
     */
    public static int optimize(ExecutionPlan plan) {
        int fusions = 0;
        List<GraphNode> nodes = plan.orderedNodes;

        for (int i = 0; i < nodes.size() - 1; i++) {
            GraphNode a = nodes.get(i);
            GraphNode b = nodes.get(i + 1);

            String fusedOp = tryFuse(a.op, b.op);
            if (fusedOp != null && canFuse(a, b, nodes)) {
                // Replace b with fused op, consuming a's inputs
                b.op = fusedOp;
                b.inputs.clear();
                b.inputs.addAll(a.inputs);

                nodes.remove(i);
                i--;
                fusions++;
            }
        }

        return fusions;
    }

    /**
     * Check if two ops can be fused into a single operation.
     *
     * @return fused op name, or null if not fusible
     */
    private static String tryFuse(String first, String second) {
        // add + relu → add_relu
        if ("add".equals(first) && "relu".equals(second)) return "add_relu";

        // matmul + relu → matmul_relu
        if ("matmul".equals(first) && "relu".equals(second)) return "matmul_relu";

        // matmul + add → matmul_add (bias fusion)
        if ("matmul".equals(first) && "add".equals(second)) return "matmul_add";

        return null;
    }

    /**
     * Safety check: ensure the first op's output is ONLY consumed by the second op.
     * If other nodes also read from a, fusion would break the graph.
     */
    private static boolean canFuse(GraphNode a, GraphNode b, List<GraphNode> allNodes) {
        // Count how many nodes consume a's output
        long consumers = allNodes.stream()
            .filter(n -> n != b && n.inputs.contains(a))
            .count();
        return consumers == 0;
    }
}
