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

public final class GContext {
    private final Map<GValueId, GValue> values = new HashMap<>();

    public void define(GValue value) {
        values.put(value.id(), value);
    }

    public GValue get(GValueId id) {
        GValue v = values.get(id);
        if (v == null)
            throw new IRException("Undefined value: " + id);
        return v;
    }

    public boolean exists(GValueId id) {
        return values.containsKey(id);
    }
}
