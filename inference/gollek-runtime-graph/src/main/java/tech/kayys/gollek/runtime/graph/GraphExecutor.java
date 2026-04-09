package tech.kayys.gollek.runtime.graph;

import tech.kayys.gollek.runtime.tensor.*;

import java.util.List;

/**
 * Executes a compiled {@link ComputationGraph} with full optimization pipeline:
 * <ol>
 *   <li>Topological sort ({@link GraphPlanner})</li>
 *   <li>Operator fusion ({@link FusionOptimizer})</li>
 *   <li>Lifetime analysis ({@link LifetimeAnalyzer})</li>
 *   <li>Memory-planned execution ({@link GraphMemoryPlanner})</li>
 * </ol>
 * <p>
 * This is the central engine that transforms a lazy computation graph into
 * materialized tensor results with optimal memory usage and fused operations.
 */
public final class GraphExecutor {

    private GraphExecutor() {}

    /**
     * Execute a computation graph end-to-end.
     *
     * @param graph   the computation graph to execute
     * @param backend the backend to use for all operations
     * @param pool    tensor pool for memory reuse
     * @param ctx     execution context for lifecycle management
     * @return the output tensor of the last node in the graph
     */
    public static Tensor execute(
        ComputationGraph graph,
        Backend backend,
        TensorPool pool,
        ExecutionContext ctx
    ) {
        // 1. Compile: topological sort
        ExecutionPlan plan = GraphPlanner.plan(graph);

        // 2. Optimize: fuse adjacent ops
        FusionOptimizer.optimize(plan);

        // 3. Analyze: compute tensor lifetimes
        LifetimeAnalyzer.analyze(plan);

        // 4. Execute with memory planning
        GraphMemoryPlanner memoryPlanner = new GraphMemoryPlanner();
        List<GraphNode> nodes = plan.orderedNodes;

        for (int i = 0; i < nodes.size(); i++) {
            GraphNode node = nodes.get(i);

            // Skip input nodes (their output is pre-set)
            if ("input".equals(node.op)) continue;

            // Gather materialized input tensors
            List<Tensor> inputs = node.inputs.stream()
                .map(n -> n.output)
                .toList();

            // Execute the node
            node.output = executeNode(node, inputs, backend, ctx);

            // Release memory for tensors that are no longer needed
            memoryPlanner.releaseExpired(i, nodes);
        }

        // Return the last node's output
        return nodes.getLast().output;
    }

    /**
     * Execute a single graph node, dispatching to the appropriate backend op.
     */
    private static Tensor executeNode(
        GraphNode node,
        List<Tensor> inputs,
        Backend backend,
        ExecutionContext ctx
    ) {
        return switch (node.op) {

            case "add" -> backend.add(inputs.get(0), inputs.get(1), ctx);

            case "relu" -> backend.relu(inputs.get(0), ctx);

            case "matmul" -> backend.matmul(inputs.get(0), inputs.get(1), ctx);

            // ── Fused operations ──────────────────────────────────────
            case "add_relu" -> {
                Tensor tmp = backend.add(inputs.get(0), inputs.get(1), ctx);
                yield backend.relu(tmp, ctx);
            }

            case "matmul_relu" -> {
                Tensor tmp = backend.matmul(inputs.get(0), inputs.get(1), ctx);
                yield backend.relu(tmp, ctx);
            }

            case "matmul_add" -> {
                Tensor tmp = backend.matmul(inputs.get(0), inputs.get(1), ctx);
                // Third input is the bias (from the fused add)
                if (inputs.size() > 2) {
                    yield backend.add(tmp, inputs.get(2), ctx);
                }
                yield tmp;
            }

            default -> throw new UnsupportedOperationException(
                "Unknown graph op: " + node.op);
        };
    }
}
