package tech.kayys.gollek.nn;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.gollek.ir.*;
import java.util.*;

/* OpDescriptor EMBEDDING = new OpDescriptor(new OpId("nn", "embedding", 1));
OpDescriptor ROPE = new OpDescriptor(new OpId("nn", "rope", 1));
OpDescriptor LINEAR = new OpDescriptor(new OpId("nn", "linear", 1));
OpDescriptor LAYERNORM = new OpDescriptor(new OpId("nn", "layernorm", 1));
OpDescriptor ATTENTION = new OpDescriptor(new OpId("nn", "attention", 1));
OpDescriptor GELU = new OpDescriptor(new OpId("nn", "gelu", 1));
OpDescriptor ADD = new OpDescriptor(new OpId("core", "add", 1));
OpDescriptor CROSS_ENTROPY = new OpDescriptor(new OpId("nn", "cross_entropy", 1));
 */
public final class LLMModelBuilder {
        private final TransformerBlockBuilder block;
        private final OpDescriptor EMBEDDING;
        private final OpDescriptor ROPE;
        private final OpDescriptor LINEAR;
        private final OpDescriptor LAYERNORM;
        private final OpDescriptor CROSS_ENTROPY;

        public LLMModelBuilder(TransformerBlockBuilder block,
                        OpDescriptor EMBEDDING,
                        OpDescriptor ROPE,
                        OpDescriptor LINEAR,
                        OpDescriptor LAYERNORM,
                        OpDescriptor CROSS_ENTROPY) {
                this.block = block;
                this.EMBEDDING = EMBEDDING;
                this.ROPE = ROPE;
                this.LINEAR = LINEAR;
                this.LAYERNORM = LAYERNORM;
                this.CROSS_ENTROPY = CROSS_ENTROPY;
        }

        public GGraph build(int layers,
                        GValueRef tokens,
                        GValueRef targets,
                        Map<String, GValueRef> params,
                        GValueId lossOut) {
                List<GOp> ops = new ArrayList<>();
                // ---- Embedding ----
                GValueId embed = id("embed");
                ops.add(new GOp(
                                EMBEDDING,
                                "embedding",
                                List.of(tokens, params.get("token_embedding")),
                                List.of(embed),
                                Map.of()));
                // ---- RoPE ----
                GValueId rope = id("rope");
                ops.add(new GOp(
                                ROPE,
                                "rope",
                                List.of(ref(embed)),
                                List.of(rope),
                                Map.of()));
                // ---- Transformer Stack ----
                GValueRef current = ref(rope);

                for (int i = 0; i < layers; i++) {
                        List<GValueId> outs = new ArrayList<>();
                        ops.addAll(block.build(
                                        "layer" + i,
                                        current,
                                        params,
                                        outs));
                        current = ref(outs.get(0));
                }
                // ---- Final LayerNorm ----
                GValueId ln = id("final_ln");
                ops.add(new GOp(
                                LAYERNORM,
                                "final_ln",
                                List.of(
                                                current,
                                                params.get("final_ln_gamma"),
                                                params.get("final_ln_beta")),
                                List.of(ln),
                                Map.of()));
                // ---- Logits ----
                GValueId logits = id("logits");
                ops.add(new GOp(
                                LINEAR,
                                "lm_head",
                                List.of(
                                                ref(ln),
                                                params.get("lm_head_weight"),
                                                params.get("lm_head_bias")),
                                List.of(logits),
                                Map.of()));
                // ---- Loss ----
                ops.add(new GOp(
                                CROSS_ENTROPY,
                                "loss",
                                List.of(
                                                ref(logits),
                                                targets),
                                List.of(lossOut),
                                Map.of()));
                return new GGraph(
                                ops,
                                List.of(lossOut),
                                List.of(tokens.id(), targets.id()));
        }

        private GValueId id(String n) {
                return new GValueId(n);
        }

        private GValueRef ref(GValueId id) {
                return new GValueRef(id);
        }
}