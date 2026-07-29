package tech.kayys.gollek.core.graph;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.tensor.Tensor;

public interface Node {
    String id();

    Tensor eval(ExecutionContext ctx);

    default DeviceType preferredDevice() {
        return DeviceType.CPU;
    }
}
