package tech.kayys.gollek.ir;


import tech.kayys.alkhawarizm.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;

import java.util.Objects;

public final class GValue {
    private final GValueId id;
    private final GType type;
    private final Shape shape;
    private final String storageKey;
    private final boolean requiresGrad;
    private final boolean isConst;
    private final boolean isInput;
    private final boolean isOutput;

    private GValue(Builder builder) {
        this.id = builder.id;
        this.type = builder.type;
        this.shape = builder.shape;
        this.storageKey = builder.storageKey;
        this.requiresGrad = builder.requiresGrad;
        this.isConst = builder.isConst;
        this.isInput = builder.isInput;
        this.isOutput = builder.isOutput;
    }

    public GValueId id() {
        return id;
    }

    public GType type() {
        return type;
    }

    public Shape shape() {
        return shape;
    }

    public String storageKey() {
        return storageKey;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public boolean isConst() {
        return isConst;
    }

    public boolean isInput() {
        return isInput;
    }

    public boolean isOutput() {
        return isOutput;
    }

    public static class Builder {
        private final GValueId id;
        private GType type;
        private Shape shape;
        private String storageKey;
        private boolean requiresGrad = false;
        private boolean isConst = false;
        private boolean isInput = false;
        private boolean isOutput = false;

        public Builder(GValueId id) {
            this.id = Objects.requireNonNull(id);
        }

        public Builder type(GType type) {
            this.type = type;
            return this;
        }

        public Builder shape(Shape shape) {
            this.shape = shape;
            return this;
        }

        public Builder storageKey(String key) {
            this.storageKey = key;
            return this;
        }

        public Builder requiresGrad(boolean requiresGrad) {
            this.requiresGrad = requiresGrad;
            return this;
        }

        public Builder constVal(boolean isConst) {
            this.isConst = isConst;
            return this;
        }

        public Builder input(boolean isInput) {
            this.isInput = isInput;
            return this;
        }

        public Builder output(boolean isOutput) {
            this.isOutput = isOutput;
            return this;
        }

        public GValue build() {
            if (type == null)
                throw new InferenceException(ErrorCode.RUNTIME_INVALID_STATE, "Missing type for " + id);
            if (shape == null)
                throw new InferenceException(ErrorCode.RUNTIME_INVALID_STATE, "Missing shape for " + id);
            return new GValue(this);
        }
    }
}