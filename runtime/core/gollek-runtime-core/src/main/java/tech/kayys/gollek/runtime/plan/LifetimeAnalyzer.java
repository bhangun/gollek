package tech.kayys.gollek.runtime.plan;

import tech.kayys.gollek.ir.*;
import java.util.*;

public final class LifetimeAnalyzer {
    public static class Lifetime {
        public int start;
        public int end;
    }

    public static Map<GValueId, Lifetime> analyze(
            List<GOp> orderedOps) {
        Map<GValueId, Lifetime> map = new HashMap<>();
        for (int i = 0; i < orderedOps.size(); i++) {
            GOp op = orderedOps.get(i);
            final int step = i;
            for (GValueRef in : op.inputs()) {
                map.computeIfAbsent(in.id(), k -> {
                    Lifetime l = new Lifetime();
                    l.start = step;
                    l.end = step;
                    return l;
                }).end = step;
            }
            for (GValueId out : op.outputs()) {
                Lifetime l = new Lifetime();
                l.start = i;
                l.end = i;
                map.put(out, l);
            }
        }
        return map;
    }
}