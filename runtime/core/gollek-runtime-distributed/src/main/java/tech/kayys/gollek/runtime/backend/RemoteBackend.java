package tech.kayys.gollek.runtime.backend;

import tech.kayys.gollek.runtime.plan.ExecutionPlan;
import tech.kayys.gollek.runtime.control.ExecutionSession;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.client.RemoteClient;
import java.util.Map;

public final class RemoteBackend implements ExecutionBackend {
    private final RemoteClient client;

    public RemoteBackend(RemoteClient client) {
        this.client = client;
    }

    @Override
    public Map<String, Tensor> execute(
            ExecutionPlan plan,
            Map<String, Tensor> inputs,
            ExecutionSession session) {
        return client.execute(plan, inputs, session);
    }
}
