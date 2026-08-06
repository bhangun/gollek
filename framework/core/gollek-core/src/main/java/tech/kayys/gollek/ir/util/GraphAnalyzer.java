package tech.kayys.gollek.ir.util;
import tech.kayys.gollek.ir.*;


import tech.kayys.gollek.ir.*;
import java.util.*;

public final class GraphAnalyzer {
    public static Map<GValueId, Integer> buildProducerMap(GGraph graph) {
        Map<GValueId, Integer> map = new HashMap<>();
        List<GOp> ops = graph.ops();
        for (int i = 0; i < ops.size(); i++) {
            for (GValueId out : ops.get(i).outputs()) {
                map.put(out, i);
            }
        }
        return map;
    }

    public static Map<GValueId, List<Integer>> buildConsumers(GGraph graph) {
        Map<GValueId, List<Integer>> map = new HashMap<>();
        List<GOp> ops = graph.ops();
        for (int i = 0; i < ops.size(); i++) {
            for (GValueRef in : ops.get(i).inputs()) {
                map.computeIfAbsent(in.id(), k -> new ArrayList<>())
                        .add(i);
            }
        }
        return map;
    }
}