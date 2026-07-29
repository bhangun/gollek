package tech.kayys.gollek.ir;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;

import java.util.Objects;

public final class GValueRef {
    private final GValueId id;

    public GValueRef(GValueId id) {
        this.id = Objects.requireNonNull(id);
    }

    public GValueId id() {
        return id;
    }
}