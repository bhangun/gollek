package tech.kayys.gollek.core.model;

import tech.kayys.gollek.ir.GGraph;
import tech.kayys.gollek.core.weight.WeightStore;
import tech.kayys.alkhawarizm.spi.spec.*;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.model.ModelFormat;

public final class LoadedModel {
    public final GGraph graph;
    public final WeightStore weights;

    public LoadedModel(GGraph graph, WeightStore weights) {
        this.graph = graph;
        this.weights = weights;
    }
}