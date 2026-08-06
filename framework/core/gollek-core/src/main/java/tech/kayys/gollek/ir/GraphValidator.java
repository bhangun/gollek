package tech.kayys.gollek.ir;


import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;
import java.util.stream.Collectors;

public final class GraphValidator {
    private final OpSchemaRegistry registry;

    public GraphValidator(OpSchemaRegistry registry) {
        this.registry = registry;
    }

    public GContext validate(GGraph graph) {
        GContext ctx = new GContext();
        for (GOp op : graph.ops()) {
            OpSchema schema = registry.get(op.opType());
            if (schema == null)
                throw new InferenceException(ErrorCode.INTERNAL_ERROR, "Unknown op: " + op.opType());

            List<GValue> inputs = op.inputs().stream()
                    .map(ref -> ctx.get(ref.id()))
                    .collect(Collectors.toList());

            schema.validator().validate(inputs, op.attrs());
            Shape[] shapes = schema.shapeInfer().infer(inputs, op.attrs());
            GType[] types = schema.typeInfer().infer(inputs, op.attrs());

            for (int i = 0; i < op.outputs().size(); i++) {
                GValueId outId = op.outputs().get(i);
                GValue out = new GValue.Builder(outId)
                        .type(types[i])
                        .shape(shapes[i])
                        .build();
                ctx.define(out);
            }
        }
        return ctx;
    }
}