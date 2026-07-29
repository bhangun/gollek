package tech.kayys.gollek.ir;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;

import tech.kayys.gollek.ir.*;
import java.util.*;

public final class GradGraphAnalyzer {
    public static Map<GValueId, Integer> buildUseCount(GGraph graph) {
        Map<GValueId, Integer> useCount = new HashMap<>();
        for (GOp op : graph.ops()) {
            for (GValueRef in : op.inputs()) {
                useCount.merge(in.id(), 1, Integer::sum);
            }
        }
        return useCount;
    }

    public static Map<GValueId, GOp> buildProducerMap(GGraph graph) {
        Map<GValueId, GOp> map = new HashMap<>();
        for (GOp op : graph.ops()) {
            for (GValueId out : op.outputs()) {
                map.put(out, op);
            }
        }
        return map;
    }
}