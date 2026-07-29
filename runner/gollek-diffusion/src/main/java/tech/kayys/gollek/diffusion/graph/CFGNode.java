package tech.kayys.gollek.diffusion.graph;

import tech.kayys.gollek.core.graph.ExecutionContext;
import tech.kayys.gollek.core.graph.Node;
import tech.kayys.gollek.core.graph.AbstractNode;
import tech.kayys.alkhawarizm.core.tensor.Tensor;

/**
 * CLASSIFIER-FREE GUIDANCE (CFG)
 * 
 * @author bhangun
 */
public final class CFGNode extends AbstractNode {
    private final Node cond;
    private final Node uncond;
    private final float scale;

    public CFGNode(Node cond, Node uncond, float scale) {
        this.cond = cond;
        this.uncond = uncond;
        this.scale = scale;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        Tensor uc = uncond.eval(ctx);
        Tensor c = cond.eval(ctx);
        return uc.add(
                c.sub(uc).mul(scale));
    }
}
