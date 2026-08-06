package tech.kayys.gollek.runtime.plan;

import java.util.*;

public final class ExecutablePlan {
    public final List<PlannedStep> steps;

    public ExecutablePlan(List<PlannedStep> steps) {
        this.steps = steps;
    }
}