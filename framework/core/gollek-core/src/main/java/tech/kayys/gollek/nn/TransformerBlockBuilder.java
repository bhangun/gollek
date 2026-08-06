package tech.kayys.gollek.nn;


import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import java.util.*;

/* OpDescriptor MATMUL = new OpDescriptor(new OpId("core", "matmul", 1));
OpDescriptor ADD = new OpDescriptor(new OpId("core", "add", 1));
OpDescriptor GELU = new OpDescriptor(new OpId("nn", "gelu", 1));
OpDescriptor LAYERNORM = new OpDescriptor(new OpId("nn", "layernorm", 1));
OpDescriptor ATTENTION = new OpDescriptor(new OpId("nn", "attention", 1));
OpDescriptor LINEAR = new OpDescriptor(new OpId("nn", "linear", 1)); 
 */
public final class TransformerBlockBuilder {
        private final OpDescriptor ADD;
        private final OpDescriptor LINEAR;
        private final OpDescriptor GELU;
        private final OpDescriptor LAYERNORM;
        private final OpDescriptor ATTENTION;

        public TransformerBlockBuilder(OpDescriptor ADD,
                        OpDescriptor LINEAR,
                        OpDescriptor GELU,
                        OpDescriptor LAYERNORM,
                        OpDescriptor ATTENTION) {
                this.ADD = ADD;
                this.LINEAR = LINEAR;
                this.GELU = GELU;
                this.LAYERNORM = LAYERNORM;
                this.ATTENTION = ATTENTION;
        }

        public List<GOp> build(String prefix,
                        GValueRef input,
                        Map<String, GValueRef> params,
                        List<GValueId> outputs) {
                List<GOp> ops = new ArrayList<>();
                // ---- LayerNorm 1 ----
                GValueId ln1 = id(prefix, "ln1");
                ops.add(new GOp(
                                LAYERNORM,
                                prefix + "_ln1",
                                List.of(input, params.get("ln1_gamma"), params.get("ln1_beta")),
                                List.of(ln1),
                                Map.of()));
                // ---- QKV projection ----
                GValueId q = id(prefix, "q");
                GValueId k = id(prefix, "k");
                GValueId v = id(prefix, "v");
                ops.add(linear(prefix, "q_proj", ln1, params, q));
                ops.add(linear(prefix, "k_proj", ln1, params, k));
                ops.add(linear(prefix, "v_proj", ln1, params, v));
                // ---- Attention ----
                GValueId attn = id(prefix, "attn");

                ops.add(new GOp(
                                ATTENTION,
                                prefix + "_attn",
                                List.of(ref(q), ref(k), ref(v)),
                                List.of(attn),
                                Map.of()));
                // ---- Residual 1 ----
                GValueId res1 = id(prefix, "res1");
                ops.add(new GOp(
                                ADD,
                                prefix + "_res1",
                                List.of(input, ref(attn)),
                                List.of(res1),
                                Map.of()));
                // ---- LayerNorm 2 ----
                GValueId ln2 = id(prefix, "ln2");
                ops.add(new GOp(
                                LAYERNORM,
                                prefix + "_ln2",
                                List.of(ref(res1), params.get("ln2_gamma"), params.get("ln2_beta")),
                                List.of(ln2),
                                Map.of()));
                // ---- MLP ----
                GValueId fc1 = id(prefix, "fc1");
                GValueId gelu = id(prefix, "gelu");
                GValueId fc2 = id(prefix, "fc2");
                ops.add(linear(prefix, "fc1", ln2, params, fc1));
                ops.add(new GOp(
                                GELU,
                                prefix + "_gelu",
                                List.of(ref(fc1)),
                                List.of(gelu),
                                Map.of()));
                ops.add(linear(prefix, "fc2", gelu, params, fc2));
                // ---- Residual 2 ----
                GValueId res2 = id(prefix, "res2");
                ops.add(new GOp(ADD,
                                prefix + "_res2",
                                List.of(ref(res1), ref(fc2)),
                                List.of(res2),
                                Map.of()));
                outputs.add(res2);
                return ops;
        }

        private GOp linear(String prefix,
                        String name,
                        GValueId input,
                        Map<String, GValueRef> params,
                        GValueId out) {
                return new GOp(LINEAR,
                                prefix + "_" + name,
                                List.of(
                                                ref(input),
                                                params.get(name + "_weight"),
                                                params.get(name + "_bias")),
                                List.of(out),
                                Map.of());
        }

        private GValueId id(String p, String n) {
                return new GValueId(p + "_" + n);
        }

        private GValueRef ref(GValueId id) {
                return new GValueRef(id);
        }
}