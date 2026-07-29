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

public final class AddOp implements OpBinary {
    @Override
    public Tensor apply(ExecutionContext ctx, Tensor a, Tensor b) {
        return ctx.backend().add(a, b);
    }
}