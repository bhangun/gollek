package tech.kayys.gollek.ir;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;

import java.util.*;

public final class OpRegistry {

    private final Map<String, OpDescriptor> ops = new HashMap<>();

    public void register(OpDescriptor op) {
        ops.put(op.id().key(), op);
    }

    public OpDescriptor get(OpId id) {
        OpDescriptor op = ops.get(id.key());
        if (op == null) {
            throw new RuntimeException("Unknown op: " + id.key());
        }
        return op;
    }
}
