package tech.kayys.gollek.ir;


import tech.kayys.alkhawarizm.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.alkhawarizm.core.tensor.*;
import java.util.*;
import java.util.stream.Collectors;

public final class GGraphBuilder {
    private final List<GOp> ops = new ArrayList<>();
    private final Set<GValueId> defined = new HashSet<>();
    private final List<GValueId> inputs = new ArrayList<>();
    private final List<GValueId> outputs = new ArrayList<>();

    public GValueId input(String name, GType type, Shape shape) {
        GValueId id = new GValueId(name);
        defined.add(id);
        inputs.add(id);
        return id;
    }

    public GValueId op(
            String opType,
            List<GValueId> inputs,
            String outputName,
            Map<String, GAttrValue> attrs) {
        for (GValueId in : inputs) {
            if (!defined.contains(in))
                throw new InferenceException(ErrorCode.INTERNAL_ERROR, "Undefined input: " + in);
        }
        GValueId out = new GValueId(outputName);
        OpDescriptor desc = new OpDescriptor(new OpId(opType));

        ops.add(new GOp(
                desc,
                outputName,
                inputs.stream().map(GValueRef::new).collect(Collectors.toList()),
                List.of(out),
                attrs));

        defined.add(out);
        return out;
    }

    public void output(GValueId id) {
        outputs.add(id);
    }

    public GGraph build() {
        return new GGraph(ops, inputs, outputs);
    }
}
