package tech.kayys.gollek.core.graph.util;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.gollek.core.graph.Node;

import java.util.*;

public final class GraphUtils {
    public static List<Node> topoSort(Node root) {
        List<Node> result = new ArrayList<>();
        Set<Node> visited = new HashSet<>();
        dfs(root, visited, result);
        Collections.reverse(result);
        return result;
    }

    private static void dfs(Node node, Set<Node> visited, List<Node> result) {
        if (!visited.add(node))
            return;
        if (node instanceof HasInputs hi) {
            for (Node in : hi.inputs()) {
                dfs(in, visited, result);
            }
        }
        result.add(node);
    }
}