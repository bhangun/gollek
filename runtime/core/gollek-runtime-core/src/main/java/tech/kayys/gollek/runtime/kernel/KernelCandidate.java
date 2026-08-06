package tech.kayys.gollek.runtime.kernel;

import tech.kayys.alkhawarizm.core.tensor.DeviceType;

public final class KernelCandidate {
    public final String id;
    public final DeviceType device;

    public KernelCandidate(String id, DeviceType device) {
        this.id = id;
        this.device = device;
    }
}
