package tech.kayys.gollek.runtime.execution;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.plan.ExecutionPlan;
import tech.kayys.gollek.runtime.control.ExecutionSession;

import java.util.Collections;
import java.util.Map;

public final class ExecutionRequest {
    public final String tenantId;
    public final ExecutionPlan plan;
    public final ExecutionContext context;
    public final ExecutionSession session;
    /** Named tensor inputs for the execution plan. */
    public final Map<String, Tensor> inputs;

    public ExecutionRequest(String tenantId,
            ExecutionPlan plan,
            ExecutionContext context,
            ExecutionSession session,
            Map<String, Tensor> inputs) {
        this.tenantId = tenantId;
        this.plan = plan;
        this.context = context;
        this.session = session;
        this.inputs = inputs != null ? inputs : Collections.emptyMap();
    }

    /** Backward-compatible constructor — uses an empty inputs map. */
    public ExecutionRequest(String tenantId,
            ExecutionPlan plan,
            ExecutionContext context,
            ExecutionSession session) {
        this(tenantId, plan, context, session, Collections.emptyMap());
    }
}