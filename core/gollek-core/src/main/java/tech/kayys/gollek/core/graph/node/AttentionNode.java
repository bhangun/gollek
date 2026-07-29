package tech.kayys.gollek.core.graph.node;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.alkhawarizm.core.tensor.Tensor;

public final class AttentionNode extends AbstractNode implements HasInputs {
    private final Node Q, K, V;

    public AttentionNode(Node Q, Node K, Node V) {
        this.Q = Q;
        this.K = K;
        this.V = V;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        return ctx.backend().attention(
                Q.eval(ctx),
                K.eval(ctx),
                V.eval(ctx));
    }

    @Override
    public Node[] inputs() {
        return new Node[] { Q, K, V };
    }
}