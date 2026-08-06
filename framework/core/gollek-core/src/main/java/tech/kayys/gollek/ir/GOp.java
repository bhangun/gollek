package tech.kayys.gollek.ir;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import java.util.*;

public final class GOp {
    private final OpDescriptor op;
    private final String name;
    private final List<GValueRef> inputs;
    private final List<GValueId> outputs;
    private final Map<String, GAttrValue> attrs;

    public GOp(String opName,
            String name,
            List<GValueRef> inputs,
            List<GValueId> outputs,
            Map<String, GAttrValue> attrs) {
        this(new OpDescriptor(new OpId(opName)), name, inputs, outputs, attrs);
    }

    public GOp(OpDescriptor op,
            String name,
            List<GValueRef> inputs,
            List<GValueId> outputs,
            Map<String, GAttrValue> attrs) {
        this.op = op;
        this.name = name;
        this.inputs = inputs;
        this.outputs = outputs;
        this.attrs = attrs;
    }

    public OpDescriptor op() {
        return op;
    }

    public String name() {
        return name;
    }

    public List<GValueRef> inputs() {
        return inputs;
    }

    public List<GValueId> outputs() {
        return outputs;
    }

    public String opType() {
        return op != null ? op.id().name() : "unknown";
    }

    public Map<String, GAttrValue> attrs() {
        return attrs;
    }
}