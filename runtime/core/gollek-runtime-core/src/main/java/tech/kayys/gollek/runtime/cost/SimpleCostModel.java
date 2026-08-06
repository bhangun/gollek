package tech.kayys.gollek.runtime.cost;

import tech.kayys.gollek.ir.GOp;
import tech.kayys.gollek.runtime.kernel.KernelCandidate;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;

public final class SimpleCostModel implements CostModel {
    @Override
    public double estimate(GOp op,
            KernelCandidate kernel,
            DeviceType device) {
        // naive baseline
        double base = switch (op.opType()) {
            case "matmul" -> 10;
            case "flash_attention_v3" -> 5;
            default -> 1;
        };
        double deviceFactor = switch (device) {
            case CPU -> 2.0;
            case METAL -> 0.7;
            default -> 1.5; // for REMOTE, etc.
        };
        return base * deviceFactor;
    }
}
