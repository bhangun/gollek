package tech.kayys.gollek.ir;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

public final class OpSchema {
    private final String opType;
    private final int minInputs;
    private final int maxInputs;
    private final int outputs;
    private final ShapeInfer shapeInfer;
    private final TypeInfer typeInfer;
    private final Validator validator;

    public OpSchema(
            String opType,
            int minInputs,
            int maxInputs,
            int outputs,
            ShapeInfer shapeInfer,
            TypeInfer typeInfer,
            Validator validator) {
        this.opType = opType;
        this.minInputs = minInputs;
        this.maxInputs = maxInputs;
        this.outputs = outputs;
        this.shapeInfer = shapeInfer;
        this.typeInfer = typeInfer;
        this.validator = validator;
    }

    public String opType() {
        return opType;
    }

    public int minInputs() {
        return minInputs;
    }

    public int maxInputs() {
        return maxInputs;
    }

    public int outputs() {
        return outputs;
    }

    public ShapeInfer shapeInfer() {
        return shapeInfer;
    }

    public TypeInfer typeInfer() {
        return typeInfer;
    }

    public Validator validator() {
        return validator;
    }
}