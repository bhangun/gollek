package tech.kayys.golek.core.pipeline;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.core.inference.InferencePhase;

/**
 * Pipeline that executes all phases in order.
 */
public interface InferencePipeline {

    /**
     * Execute all phases for the given context
     */
    Uni<ExecutionContext> execute(ExecutionContext context);

    /**
     * Execute a specific phase
     */
    Uni<ExecutionContext> executePhase(ExecutionContext context, InferencePhase phase);
}