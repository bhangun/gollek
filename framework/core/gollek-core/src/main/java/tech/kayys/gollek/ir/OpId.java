package tech.kayys.gollek.ir;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;

public final class OpId {
    private final String namespace; // e.g. "core", "nn", "custom"
    private final String name; // e.g. "softmax"
    private final int version;

    public OpId(String name) {
        this("core", name, 1);
    }

    public OpId(String namespace, String name, int version) {
        this.namespace = namespace;
        this.name = name;
        this.version = version;
    }

    public String key() {
        return namespace + ":" + name + ":" + version;
    }

    public String name() {
        return name;
    }

    public String namespace() {
        return namespace;
    }

    public int version() {
        return version;
    }
}