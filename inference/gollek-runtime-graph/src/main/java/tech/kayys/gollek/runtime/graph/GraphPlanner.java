package tech.kayys.gollek.runtime.graph;

import java.util.*;

/**
 * Compiles a {@link ComputationGraph} into an {@link ExecutionPlan} by
 * performing a topological sort of the graph nodes.
 * <p>
 * The resulting plan ensures that every node's inputs are computed before
 * the node itself is executed.
 */
public final class GraphPlanner {

    private GraphPlanner() {}

    /**
     * Plan execution order via depth-first topological sort.
     *
     * @param graph the computation graph to compile
     * @return an execution plan with nodes in valid execution order
     */
    public static ExecutionPlan plan(ComputationGraph graph) {
        List<GraphNode> ordered = new ArrayList<>();
        Set<GraphNode> visited = new HashSet<>();

        for (GraphNode node : graph.nodes()) {
            dfs(node, visited, ordered);
        }

        return new ExecutionPlan(ordered);
    }

    private static void dfs(GraphNode node,
                             Set<GraphNode> visited,
                             List<GraphNode> result) {
        if (visited.contains(node)) return;

        for (GraphNode input : node.inputs) {
            dfs(input, visited, result);
        }

        visited.add(node);
        result.add(node);
    }
}
