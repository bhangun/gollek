package tech.kayys.gollek.runtime.execution;

import tech.kayys.gollek.runtime.data.GData;
import tech.kayys.gollek.runtime.data.GTensor;
import tech.kayys.gollek.runtime.data.GScalar;
// import tech.kayys.gollek.backend.Backend;
// import tech.kayys.gollek.ir.Op;
// import tech.kayys.gollek.runtime.GraphArtifact;
// import tech.kayys.gollek.runtime.ExecutionStep;

/**
 * Generic executor that processes a computation graph artifact
 * using the unified ExecutionContext.
 */
public final class ArtifactExecutor {
    
    // Commented out the exact implementation until GraphArtifact and Op are available, 
    // but this represents the core loop from the design doc.
    /*
    public void run(GraphArtifact artifact, ExecutionContext ctx) {
        for (ExecutionStep step : artifact.plan().steps()) {
            Op op = resolveOp(step.opId());
            Backend backend = ctx.selectBackend(op);
            
            GData[] inputs = step.inputs().stream()
                    .map(ctx::get)
                    .toArray(GData[]::new);
                    
            GData output = backend.execute(op, inputs);
            ctx.put(output);
        }
    }
    */
}
