package tech.kayys.golek.engine.execution;

import tech.kayys.golek.api.tenant.TenantContext;
import tech.kayys.golek.model.ModelManifest;

public interface ModelRunnerFactory {
    ModelRunner getRunner(ModelManifest manifest, String runnerName, TenantContext tenantContext);
}