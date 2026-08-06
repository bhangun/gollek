package tech.kayys.gollek.nn;


import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.gollek.ir.*;
import java.util.*;

public final class TransformerModelBuilder {
    private final TransformerBlockBuilder block;

    public TransformerModelBuilder(TransformerBlockBuilder block) {
        this.block = block;
    }

    public GGraph build(int layers,
            GValueRef input,
            Map<String, GValueRef> params,
            GValueId output) {
        List<GOp> ops = new ArrayList<>();
        GValueRef current = input;
        for (int i = 0; i < layers; i++) {
            List<GValueId> outs = new ArrayList<>();
            ops.addAll(block.build(
                    "layer" + i,
                    current,
                    params,
                    outs));
            current = new GValueRef(outs.get(0));
        }
        return new GGraph(
                ops,
                List.of(current.id()),
                List.of(input.id()));
    }
}