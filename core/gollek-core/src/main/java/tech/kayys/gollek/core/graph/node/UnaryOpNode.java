package tech.kayys.gollek.core.graph.node;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.alkhawarizm.core.tensor.Tensor;

public final class UnaryOpNode extends AbstractNode implements HasInputs {
    private final Node input;
    private final OpUnary op;

    public UnaryOpNode(Node input, OpUnary op) {
        this.input = input;
        this.op = op;
    }

    public OpUnary op() {
        return op;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        Tensor x = input.eval(ctx);
        return op.apply(ctx, x);
    }

    @Override
    public Node[] inputs() {
        return new Node[] { input };
    }
}