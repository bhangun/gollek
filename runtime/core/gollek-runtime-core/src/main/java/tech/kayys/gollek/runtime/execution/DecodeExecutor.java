package tech.kayys.gollek.runtime.execution;

import tech.kayys.gollek.runtime.data.GScalar;
import tech.kayys.gollek.runtime.data.GTensor;
// import tech.kayys.gollek.runtime.LLMArtifact;

/**
 * Executor dedicated to the decoding phase of LLMs, 
 * pushing scalar tokens into the unified context.
 */
public final class DecodeExecutor {
    private final ArtifactExecutor executor;

    public DecodeExecutor(ArtifactExecutor executor) {
        this.executor = executor;
    }

    /*
    public int step(LLMArtifact model, ExecutionContext ctx, int token) {
        ctx.put(new GScalar("input_ids", token));
        
        executor.run(model.decode, ctx);
        
        GTensor logits = ctx.get("logits");
        return logits.tensor().argmax();
    }
    */
}
