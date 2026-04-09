package tech.kayys.gollek.runtime.inference.router;

import tech.kayys.gollek.runtime.graph.GraphNode;
import tech.kayys.gollek.runtime.tensor.Backend;
import tech.kayys.gollek.runtime.tensor.Tensor;

import java.util.List;

/**
 * Selects which backend should execute a given graph node.
 * <p>
 * Enables hybrid execution where different operations route to different
 * backends based on heuristics or cost models:
 * <pre>
 *   matmul → LibTorch (GPU)
 *   attention → GGML (INT4 CPU)
 *   softmax → LiteRT
 * </pre>
 */
public interface BackendRouter {

    /**
     * Select the optimal backend for the given node.
     *
     * @param node   the graph node to execute
     * @param inputs the materialized input tensors
     * @return the backend to execute on
     */
    Backend selectBackend(GraphNode node, List<Tensor> inputs);
}
