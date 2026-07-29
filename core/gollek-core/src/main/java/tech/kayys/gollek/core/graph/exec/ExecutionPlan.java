package tech.kayys.gollek.core.graph.exec;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.gollek.core.graph.Node;
import java.util.List;

public final class ExecutionPlan {
    private final List<Node> order;

    public ExecutionPlan(List<Node> order) {
        this.order = order;
    }

    public List<Node> order() {
        return order;
    }
}