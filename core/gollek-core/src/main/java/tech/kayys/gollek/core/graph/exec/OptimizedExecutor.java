package tech.kayys.gollek.core.graph.exec;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.data.GTensor;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.Map;

public final class OptimizedExecutor {
    public Tensor run(Node root, ExecutionContext ctx) {
        ExecutionPlan plan = new ExecutionPlanner().plan(root);
        Map<Node, Integer> refCount = RefCounter.count(root);
        Tensor result = null;
        for (Node node : plan.order()) {
            Tensor t = node.eval(ctx);
            ctx.put(new GTensor(node.id(), t));

            if (node == root)
                result = t;

            if (node instanceof HasInputs hi) {
                for (Node in : hi.inputs()) {
                    int r = refCount.computeIfPresent(in, (k, v) -> v - 1);
                    if (r == 0) {
                        GTensor g = ctx.get(in.id());
                        if (g != null) {
                            g.tensor().release(); // 🔥 memory freed
                        }
                    }
                }
            }
        }
        return result;
    }
}