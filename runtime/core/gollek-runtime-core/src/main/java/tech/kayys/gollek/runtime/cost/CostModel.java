package tech.kayys.gollek.runtime.cost;

import tech.kayys.gollek.ir.*;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.gollek.runtime.kernel.*;

public interface CostModel {
    double estimate(
            GOp op,
            KernelCandidate kernel,
            DeviceType device);
}
