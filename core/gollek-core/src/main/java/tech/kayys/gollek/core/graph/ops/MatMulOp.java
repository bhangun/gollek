package tech.kayys.gollek.core.graph.ops;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.alkhawarizm.core.tensor.Tensor;

public final class MatMulOp implements OpBinary {
    @Override
    public Tensor apply(ExecutionContext ctx, Tensor a, Tensor b) {
        return a.matmul(b);
    }
}
