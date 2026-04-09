package tech.kayys.gollek.runtime.inference.router;

import tech.kayys.gollek.runtime.graph.GraphNode;
import tech.kayys.gollek.runtime.tensor.*;

import java.util.List;

/**
 * Heuristic-based backend router that selects backends based on
 * operation type and tensor size.
 * <p>
 * Rules:
 * <ul>
 *   <li>Large matmul (>1M elements) → GPU backend (LibTorch)</li>
 *   <li>Small matmul / attention → CPU quantized backend (GGML)</li>
 *   <li>Simple element-wise ops → lightweight backend (LiteRT)</li>
 * </ul>
 */
public final class HeuristicRouter implements BackendRouter {

    private final BackendType gpuBackend;
    private final BackendType cpuBackend;
    private final BackendType defaultBackend;

    public HeuristicRouter(BackendType gpuBackend, BackendType cpuBackend, BackendType defaultBackend) {
        this.gpuBackend = gpuBackend;
        this.cpuBackend = cpuBackend;
        this.defaultBackend = defaultBackend;
    }

    /** Create a router with standard defaults. */
    public static HeuristicRouter standard() {
        return new HeuristicRouter(
            BackendType.LIBTORCH,
            BackendType.GGML,
            BackendType.LIBTORCH
        );
    }

    @Override
    public Backend selectBackend(GraphNode node, List<Tensor> inputs) {
        String op = node.op;

        if ("matmul".equals(op) || "matmul_relu".equals(op) || "matmul_add".equals(op)) {
            if (!inputs.isEmpty() && inputs.get(0).numel() > 1_000_000) {
                return tryGet(gpuBackend);
            }
            return tryGet(cpuBackend);
        }

        if ("attention".equals(op) || "qmatmul".equals(op)) {
            return tryGet(cpuBackend);
        }

        if ("softmax".equals(op)) {
            // Softmax benefits from FP precision — use GPU backend
            return tryGet(gpuBackend);
        }

        return tryGet(defaultBackend);
    }

    private Backend tryGet(BackendType type) {
        if (BackendRegistry.isAvailable(type)) {
            return BackendRegistry.get(type);
        }
        // Fallback to whatever is registered
        return BackendRegistry.get(defaultBackend);
    }
}
