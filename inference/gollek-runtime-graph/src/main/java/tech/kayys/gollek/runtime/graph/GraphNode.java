package tech.kayys.gollek.runtime.graph;

import tech.kayys.gollek.runtime.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A node in the computation graph.
 * <p>
 * Each node represents a single tensor operation (add, relu, matmul, etc.)
 * with references to its input nodes. After execution, the {@link #output}
 * field holds the materialized result tensor.
 * <p>
 * Lifetime fields ({@link #firstUse}, {@link #lastUse}) are populated by
 * {@link LifetimeAnalyzer} and used by {@link GraphMemoryPlanner} to
 * determine when a tensor's memory can be reused.
 */
public final class GraphNode {

    /** Operation name (e.g. "add", "relu", "matmul", "add_relu"). */
    public String op;

    /** Input nodes whose outputs feed into this operation. */
    public final List<GraphNode> inputs;

    /** Optional attributes (e.g. axis, epsilon, etc.). */
    public final Map<String, Object> attrs;

    /** Materialized output tensor (set during execution). */
    public Tensor output;

    // ── Lifetime tracking (set by LifetimeAnalyzer) ───────────────────

    /** Index in the execution plan where this node's output is first produced. */
    public int firstUse = -1;

    /** Index in the execution plan where this node's output is last consumed. */
    public int lastUse = -1;

    public GraphNode(String op, List<GraphNode> inputs) {
        this.op = op;
        this.inputs = new ArrayList<>(inputs);
        this.attrs = new HashMap<>();
    }

    /** Create a leaf node (no inputs — represents a constant or input tensor). */
    public static GraphNode input(String name) {
        GraphNode node = new GraphNode("input", List.of());
        node.attrs.put("name", name);
        return node;
    }

    @Override
    public String toString() {
        return "GraphNode[op=" + op
            + ", inputs=" + inputs.size()
            + ", lifetime=[" + firstUse + "→" + lastUse + "]]";
    }
}
