package tech.kayys.gollek.core.graph.ops;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.alkhawarizm.core.tensor.Tensor;

public final class MulScalarOp implements OpUnary {
    private final float scalar;

    public MulScalarOp(float scalar) {
        this.scalar = scalar;
    }

    public float scalar() {
        return scalar;
    }

    @Override
    public Tensor apply(ExecutionContext ctx, Tensor input) {
        return ctx.backend().mul(input, scalar);
    }
}