package tech.kayys.gollek.core.data;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

public final class GScalar implements GData {
    private final String id;
    private final float value;

    public GScalar(String id, float value) {
        this.id = id;
        this.value = value;
    }

    public float value() {
        return value;
    }

    @Override
    public String id() {
        return id;
    }
}
