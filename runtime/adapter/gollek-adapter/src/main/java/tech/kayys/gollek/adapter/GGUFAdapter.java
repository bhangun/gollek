package tech.kayys.gollek.adapter;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

public final class GGUFAdapter implements WeightAdapter {
    @Override
    public Tensor getWeight(String name) {
        // load quantized block
        // convert OR use special kernel
        throw new UnsupportedOperationException("quantized path");
    }

    @Override
    public int numLayers() {
        return 32;
    }

    @Override
    public int hiddenSize() {
        return 4096;
    }

    @Override
    public int numHeads() {
        return 32;
    }
}