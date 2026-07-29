package tech.kayys.gollek.core.graph.exec;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.gollek.core.graph.Node;

import tech.kayys.gollek.core.graph.util.GraphUtils;

public final class ExecutionPlanner {
    public ExecutionPlan plan(Node root) {
        return new ExecutionPlan(
                GraphUtils.topoSort(root));
    }
}