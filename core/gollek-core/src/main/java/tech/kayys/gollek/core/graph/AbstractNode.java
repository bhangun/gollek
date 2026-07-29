package tech.kayys.gollek.core.graph;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.UUID;

public abstract class AbstractNode implements Node {
    private final String id;

    protected AbstractNode() {
        this.id = UUID.randomUUID().toString();
    }

    protected AbstractNode(String id) {
        this.id = id;
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public final Tensor eval(ExecutionContext ctx) {
        Tensor cached = ctx.getCached(this);
        if (cached != null)
            return cached;
        Tensor result = compute(ctx);
        ctx.cache(this, result);
        return result;
    }

    protected abstract Tensor compute(ExecutionContext ctx);
}