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

public final class FusedAddMulNode extends AbstractNode implements HasInputs {
    private final Node a;
    private final Node b;
    private final float scalar;

    public FusedAddMulNode(Node a, Node b, float scalar) {
        this.a = a;
        this.b = b;
        this.scalar = scalar;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        Tensor ta = a.eval(ctx);
        Tensor tb = b.eval(ctx);
        // delegate to backend (must implement fused op later)
        return ctx.backend().add(ta, tb).mul(scalar);
    }

    @Override
    public Node[] inputs() {
        return new Node[] { a, b };
    }
}