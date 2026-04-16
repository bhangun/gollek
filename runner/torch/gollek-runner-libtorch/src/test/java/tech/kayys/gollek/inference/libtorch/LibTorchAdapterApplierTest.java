package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class LibTorchAdapterApplierTest {

        @Test
        void discoversLoraPairsWithWeightSuffix() {
                Map<String, LibTorchAdapterApplier.LoraPairKeys> pairs = LibTorchAdapterApplier
                                .discoverPairNames(Set.of(
                                                "model.layers.0.self_attn.q_proj.lora_A.weight",
                                                "model.layers.0.self_attn.q_proj.lora_B.weight",
                                                "model.layers.0.self_attn.k_proj.lora_A.weight"));

                assertThat(pairs).hasSize(1);
                assertThat(pairs).containsKey("model.layers.0.self_attn.q_proj");
                assertThat(pairs.get("model.layers.0.self_attn.q_proj").aKey())
                                .isEqualTo("model.layers.0.self_attn.q_proj.lora_A.weight");
                assertThat(pairs.get("model.layers.0.self_attn.q_proj").bKey())
                                .isEqualTo("model.layers.0.self_attn.q_proj.lora_B.weight");
        }

        @Test
        void discoversLoraPairsWithoutWeightSuffix() {
                Map<String, LibTorchAdapterApplier.LoraPairKeys> pairs = LibTorchAdapterApplier
                                .discoverPairNames(Set.of(
                                                "layers.3.mlp.down_proj.lora_A",
                                                "layers.3.mlp.down_proj.lora_B"));

                assertThat(pairs).hasSize(1);
                assertThat(pairs).containsKey("layers.3.mlp.down_proj");
                assertThat(pairs.get("layers.3.mlp.down_proj").aKey()).isEqualTo("layers.3.mlp.down_proj.lora_A");
                assertThat(pairs.get("layers.3.mlp.down_proj").bKey()).isEqualTo("layers.3.mlp.down_proj.lora_B");
        }
}
