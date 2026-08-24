package tech.kayys.gollek.ir;


import tech.kayys.alkhawarizm.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;

import tech.kayys.gollek.ir.util.GraphAnalyzer;
import java.util.*;

public final class Scheduler {
    public static List<GOp> schedule(GGraph graph) {
        List<GOp> ops = graph.ops();
        Map<GValueId, Integer> producer = GraphAnalyzer.buildProducerMap(graph);
        int n = ops.size();
        int[] indegree = new int[n];
        // compute dependencies
        for (int i = 0; i < n; i++) {
            for (GValueRef in : ops.get(i).inputs()) {
                Integer p = producer.get(in.id());
                if (p != null) {
                    indegree[i]++;
                }
            }
        }
        Queue<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (indegree[i] == 0)
                q.add(i);
        }
        List<GOp> ordered = new ArrayList<>();
        while (!q.isEmpty()) {
            int i = q.poll();
            ordered.add(ops.get(i));
            for (int j = 0; j < n; j++) {
                for (GValueRef in : ops.get(j).inputs()) {
                    if (producer.get(in.id()) != null &&
                            producer.get(in.id()) == i) {
                        indegree[j]--;
                        if (indegree[j] == 0) {
                            q.add(j);
                        }
                    }
                }
            }
        }
        if (ordered.size() != n)
            throw new InferenceException(ErrorCode.INTERNAL_ERROR, "Graph has cycle");
        return ordered;
    }
}