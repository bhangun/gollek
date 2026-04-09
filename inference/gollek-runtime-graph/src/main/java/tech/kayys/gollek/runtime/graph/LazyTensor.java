package tech.kayys.gollek.runtime.graph;

import tech.kayys.gollek.runtime.tensor.*;

import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * A tensor that records operations into a {@link ComputationGraph} instead
 * of executing them immediately.
 * <p>
 * This enables graph-level optimizations (fusion, memory planning) that
 * are impossible with eager execution. The graph is materialized later
 * by {@link GraphExecutor}.
 * <p>
 * Usage:
 * <pre>{@code
 * ComputationGraph graph = new ComputationGraph();
 * LazyTensor a = LazyTensor.input(graph, "a");
 * LazyTensor b = LazyTensor.input(graph, "b");
 * Tensor c = a.add(b, null).relu(null);
 * // Nothing executed yet — graph recorded for later execution
 * }</pre>
 */
public final class LazyTensor implements Tensor {

    private final GraphNode node;
    private final ComputationGraph graph;

    // Metadata for shape inference (may be null until materialized)
    private final long[] shape;
    private final DType dtype;
    private final Device device;
    private final BackendType backendType;

    public LazyTensor(GraphNode node, ComputationGraph graph,
                      long[] shape, DType dtype, Device device, BackendType backendType) {
        this.node = node;
        this.graph = graph;
        this.shape = shape;
        this.dtype = dtype;
        this.device = device;
        this.backendType = backendType;
    }

    /** Create a lazy input placeholder. */
    public static LazyTensor input(ComputationGraph graph, String name,
                                    long[] shape, DType dtype, Device device, BackendType backendType) {
        GraphNode node = GraphNode.input(name);
        graph.addNode(node);
        return new LazyTensor(node, graph, shape, dtype, device, backendType);
    }

    /** The underlying graph node. */
    public GraphNode node() {
        return node;
    }

    // ── Lazy operations (record, don't execute) ───────────────────────

    @Override
    public Tensor add(Tensor other, ExecutionContext ctx) {
        GraphNode n = new GraphNode("add",
            List.of(this.node, toLazy(other).node));
        graph.addNode(n);
        return new LazyTensor(n, graph, shape, dtype, device, backendType);
    }

    @Override
    public Tensor matmul(Tensor other, ExecutionContext ctx) {
        LazyTensor o = toLazy(other);
        long[] resultShape = inferMatmulShape(this.shape, o.shape);
        GraphNode n = new GraphNode("matmul",
            List.of(this.node, o.node));
        graph.addNode(n);
        return new LazyTensor(n, graph, resultShape, dtype, device, backendType);
    }

    @Override
    public Tensor relu(ExecutionContext ctx) {
        GraphNode n = new GraphNode("relu", List.of(this.node));
        graph.addNode(n);
        return new LazyTensor(n, graph, shape, dtype, device, backendType);
    }

    @Override
    public Tensor reshape(long... newShape) {
        GraphNode n = new GraphNode("reshape", List.of(this.node));
        n.attrs.put("shape", newShape);
        graph.addNode(n);
        return new LazyTensor(n, graph, newShape, dtype, device, backendType);
    }

    @Override
    public Tensor slice(int dim, long start, long end) {
        long[] newShape = shape != null ? shape.clone() : new long[0];
        if (newShape.length > dim) {
            newShape[dim] = end - start;
        }
        GraphNode n = new GraphNode("slice", List.of(this.node));
        n.attrs.put("dim", dim);
        n.attrs.put("start", start);
        n.attrs.put("end", end);
        graph.addNode(n);
        return new LazyTensor(n, graph, newShape, dtype, device, backendType);
    }

    @Override
    public Tensor squeeze() {
        GraphNode n = new GraphNode("squeeze", List.of(this.node));
        graph.addNode(n);
        return new LazyTensor(n, graph, shape, dtype, device, backendType); // Shape inference omitted for simplicity
    }

    @Override
    public Tensor unsqueeze(long dim) {
        GraphNode n = new GraphNode("unsqueeze", List.of(this.node));
        n.attrs.put("dim", dim);
        graph.addNode(n);
        return new LazyTensor(n, graph, shape, dtype, device, backendType);
    }

    @Override
    public List<Tensor> split(long splitSize, long dim) {
        throw new UnsupportedOperationException("Split not implemented for LazyTensor yet");
    }

    // ── Metadata ──────────────────────────────────────────────────────

    @Override
    public long[] shape() { return shape != null ? shape.clone() : new long[0]; }

    @Override
    public DType dtype() { return dtype; }

    @Override
    public Device device() { return device; }

    @Override
    public BackendType backend() { return backendType; }

    @Override
    public long numel() {
        if (shape == null) return 0;
        long n = 1;
        for (long d : shape) n *= d;
        return n;
    }

    @Override
    public MemorySegment nativeHandle() {
        if (node.output != null) {
            return node.output.nativeHandle();
        }
        throw new IllegalStateException(
            "LazyTensor has not been materialized. Execute the graph first.");
    }

    @Override
    public void close() {
        if (node.output != null) {
            node.output.close();
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────

    private LazyTensor toLazy(Tensor t) {
        if (t instanceof LazyTensor lt) return lt;
        throw new IllegalArgumentException(
            "Cannot mix LazyTensor with eager Tensor in graph mode");
    }

    private long[] inferMatmulShape(long[] a, long[] b) {
        if (a == null || b == null || a.length < 2 || b.length < 2) {
            return a; // fallback
        }
        long[] result = a.clone();
        result[result.length - 1] = b[b.length - 1];
        return result;
    }
}
