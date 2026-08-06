package tech.kayys.gollek.diffusion;

import tech.kayys.gollek.core.graph.Node;
import tech.kayys.gollek.core.graph.node.ConstantNode;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.diffusion.graph.*;
import tech.kayys.gollek.diffusion.model.UNetModel;
import tech.kayys.gollek.diffusion.scheduler.Scheduler;

public final class DiffusionGraphBuilder {
    public Node build(
            Tensor initialLatents,
            Tensor condEmbedding,
            Tensor uncondEmbedding,
            UNetModel model,
            Scheduler scheduler,
            float guidanceScale) {
        Node latents = new ConstantNode(initialLatents);
        Node cond = new ConstantNode(condEmbedding);
        Node uncond = new ConstantNode(uncondEmbedding);
        int[] timesteps = scheduler.timesteps();
        for (int i = 0; i < timesteps.length; i++) {
            Node epsCond = new tech.kayys.gollek.diffusion.graph.UNetNode(latents, cond, timesteps[i], model);
            Node epsUncond = new tech.kayys.gollek.diffusion.graph.UNetNode(latents, uncond, timesteps[i], model);
            Node guided = new CFGNode(epsCond, epsUncond, guidanceScale);
            latents = new tech.kayys.gollek.diffusion.graph.SchedulerNode(latents, guided, scheduler, i);
        }
        return latents;
    }
}