package tech.kayys.gollek.core.graph.exec;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.gollek.core.graph.*;
import java.util.*;

public final class RefCounter {
    public static Map<Node, Integer> count(Node root) {
        Map<Node, Integer> ref = new HashMap<>();
        traverse(root, ref);
        return ref;
    }

    private static void traverse(Node node, Map<Node, Integer> ref) {
        if (!(node instanceof HasInputs hi))
            return;
        for (Node in : hi.inputs()) {
            ref.put(in, ref.getOrDefault(in, 0) + 1);

            traverse(in, ref);
        }
    }
}
