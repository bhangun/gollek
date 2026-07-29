package tech.kayys.gollek.diffusion.graph;

import tech.kayys.gollek.core.graph.ExecutionContext;
import tech.kayys.gollek.core.graph.Node;
import tech.kayys.gollek.core.graph.AbstractNode;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.diffusion.scheduler.Scheduler;

public final class SchedulerNode extends AbstractNode {
    private final Node latents;
    private final Node noise;
    private final Scheduler scheduler;

    private final int tIndex;

    public SchedulerNode(Node latents, Node noise, Scheduler scheduler, int tIndex) {
        this.latents = latents;
        this.noise = noise;
        this.scheduler = scheduler;
        this.tIndex = tIndex;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        return scheduler.step(
                latents.eval(ctx),
                noise.eval(ctx),
                tIndex);
    }
}
