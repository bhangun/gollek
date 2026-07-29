package tech.kayys.gollek.diffusion.graph;

import tech.kayys.gollek.core.graph.ExecutionContext;
import tech.kayys.gollek.core.graph.Node;
import tech.kayys.gollek.core.graph.AbstractNode;
import tech.kayys.gollek.core.graph.HasInputs;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.diffusion.model.UNetModel;

public final class BatchedUNetNode extends AbstractNode implements HasInputs {
    private final Node latents;
    private final Node cond;
    private final Node uncond;
    private final int timestep;
    private final UNetModel model;

    public BatchedUNetNode(Node latents, Node cond, Node uncond, int timestep, UNetModel model) {
        this.latents = latents;
        this.cond = cond;
        this.uncond = uncond;
        this.timestep = timestep;
        this.model = model;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        Tensor l = latents.eval(ctx);
        Tensor c = cond.eval(ctx);
        Tensor u = uncond.eval(ctx);
        // backend handles batching (shape: [2, ...])
        return model.predictBatched(l, c, u, timestep);
    }

    @Override
    public Node[] inputs() {
        return new Node[] { latents, cond, uncond };
    }
}
