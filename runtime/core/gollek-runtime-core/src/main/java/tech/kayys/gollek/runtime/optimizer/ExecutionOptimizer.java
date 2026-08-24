package tech.kayys.gollek.runtime.optimizer;


import tech.kayys.alkhawarizm.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runtime.plan.*;
import tech.kayys.gollek.runtime.kernel.*;
import tech.kayys.gollek.runtime.cost.*;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.gollek.runtime.*;
import java.util.*;

public final class ExecutionOptimizer {
    private final KernelRegistry kernels;
    private final CostModel costModel;

    public ExecutionOptimizer(KernelRegistry kernels,
            CostModel costModel) {
        this.kernels = kernels;
        this.costModel = costModel;
    }

    public ExecutablePlan optimize(ExecutionPlan plan) {
        List<PlannedStep> steps = new ArrayList<>();
        for (PlanStep step : plan.steps) {
            List<KernelCandidate> candidates = kernels.get(step.op.opType());
            if (candidates.isEmpty()) {
                throw new InferenceException(ErrorCode.INTERNAL_ERROR, "No kernel for " + step.op.opType());
            }
            KernelCandidate best = null;
            double bestCost = Double.MAX_VALUE;
            for (KernelCandidate c : candidates) {
                double cost = costModel.estimate(step.op, c, c.device);
                if (cost < bestCost) {
                    bestCost = cost;
                    best = c;
                }
            }
            steps.add(new PlannedStep(
                    step.op,
                    best.device,
                    best.id));
        }
        return new ExecutablePlan(steps);
    }
}