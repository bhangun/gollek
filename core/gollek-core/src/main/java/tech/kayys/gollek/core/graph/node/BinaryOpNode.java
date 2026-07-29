package tech.kayys.gollek.core.graph.node;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.alkhawarizm.core.tensor.Tensor;

public final class BinaryOpNode extends AbstractNode implements HasInputs {
    private final Node a;
    private final Node b;
    private final OpBinary op;

    public BinaryOpNode(Node a, Node b, OpBinary op) {
        this.a = a;
        this.b = b;
        this.op = op;
    }

    public OpBinary op() {
        return op;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        Tensor ta = a.eval(ctx);
        Tensor tb = b.eval(ctx);
        return op.apply(ctx, ta, tb);
    }

    @Override
    public Node[] inputs() {
        return new Node[] { a, b };
    }
}
