package tech.kayys.gollek.runtime.graph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A directed acyclic graph (DAG) of tensor operations.
 * <p>
 * Operations are recorded lazily via {@link LazyTensor} and accumulated
 * here. The graph is then compiled into an {@link ExecutionPlan} by
 * {@link GraphPlanner}, optimized by {@link FusionOptimizer}, and
 * executed by {@link GraphExecutor}.
 */
public final class ComputationGraph {

    private final List<GraphNode> nodes = new ArrayList<>();

    /** Add a node to the graph. Returns the same node for chaining. */
    public GraphNode addNode(GraphNode node) {
        nodes.add(node);
        return node;
    }

    /** All nodes in insertion order. */
    public List<GraphNode> nodes() {
        return Collections.unmodifiableList(nodes);
    }

    /** Number of nodes in the graph. */
    public int size() {
        return nodes.size();
    }

    @Override
    public String toString() {
        return "ComputationGraph[nodes=" + nodes.size() + "]";
    }
}
