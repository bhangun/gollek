package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class OnnxTextSessionContractTest {

    @Test
    void buildsKvContractWithPositionIdsAndDefaultNames() {
        OnnxTextSessionContract contract = OnnxTextSessionContract.create(
                new String[] {
                        "input_ids",
                        "attention_mask",
                        "position_ids",
                        "past_key_values.0.key",
                        "past_key_values.0.value",
                        "past_key_values.1.key",
                        "past_key_values.1.value"
                },
                new String[] {
                        "logits",
                        "present.0.key",
                        "present.0.value",
                        "present.1.key",
                        "present.1.value"
                },
                defaultNames(),
                true,
                32);

        assertTrue(contract.hasKvInputs());
        assertTrue(contract.hasPositionIds());
        assertEquals(2, contract.kvLayerCount());
        assertEquals(4, contract.pastKvInputCount());
        assertArrayEquals(new String[] {
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "past_key_values.1.key",
                "past_key_values.1.value"
        }, contract.runInputNames());
        assertArrayEquals(new String[] {
                "logits",
                "present.0.key",
                "present.0.value",
                "present.1.key",
                "present.1.value"
        }, contract.runOutputNames());
    }

    @Test
    void disablesKvWhenPresentOutputsAreIncomplete() {
        OnnxTextSessionContract contract = OnnxTextSessionContract.create(
                new String[] {
                        "input_ids",
                        "attention_mask",
                        "past_key_values.0.key",
                        "past_key_values.0.value"
                },
                new String[] { "logits", "present.0.key" },
                defaultNames(),
                true,
                1);

        assertFalse(contract.hasKvInputs());
        assertFalse(contract.hasPositionIds());
        assertEquals(0, contract.kvLayerCount());
        assertArrayEquals(new String[] { "input_ids", "attention_mask" }, contract.runInputNames());
        assertArrayEquals(new String[] { "logits" }, contract.runOutputNames());
    }

    @Test
    void honorsGenAiTensorNameTemplates() {
        OnnxTextSessionContract.TensorNames names = new OnnxTextSessionContract.TensorNames(
                "tokens",
                "mask",
                "positions",
                "lm_logits",
                "past.{layer}.k",
                "past.{layer}.v",
                "cache_out.{}.k",
                "cache_out.{}.v");

        OnnxTextSessionContract contract = OnnxTextSessionContract.create(
                new String[] {
                        "tokens",
                        "mask",
                        "positions",
                        "past.0.k",
                        "past.0.v"
                },
                new String[] {
                        "lm_logits",
                        "cache_out.0.k",
                        "cache_out.0.v"
                },
                names,
                true,
                1);

        assertTrue(contract.hasKvInputs());
        assertTrue(contract.hasPositionIds());
        assertEquals("lm_logits", contract.logitsName());
        assertArrayEquals(new String[] { "tokens", "mask", "positions", "past.0.k", "past.0.v" },
                contract.runInputNames());
        assertArrayEquals(new String[] { "lm_logits", "cache_out.0.k", "cache_out.0.v" },
                contract.runOutputNames());
    }

    @Test
    void usePastKvCacheFlagKeepsStatelessRunEvenWhenKvNamesExist() {
        OnnxTextSessionContract contract = OnnxTextSessionContract.create(
                new String[] {
                        "input_ids",
                        "attention_mask",
                        "past_key_values.0.key",
                        "past_key_values.0.value"
                },
                new String[] {
                        "logits",
                        "present.0.key",
                        "present.0.value"
                },
                defaultNames(),
                false,
                1);

        assertFalse(contract.hasKvInputs());
        assertEquals(0, contract.kvLayerCount());
        assertArrayEquals(new String[] { "input_ids", "attention_mask" }, contract.runInputNames());
        assertArrayEquals(new String[] { "logits" }, contract.runOutputNames());
    }

    @Test
    void validatesInputs() {
        assertThrows(NullPointerException.class,
                () -> OnnxTextSessionContract.create(new String[0], new String[0], null, true, 1));
    }

    private static OnnxTextSessionContract.TensorNames defaultNames() {
        return new OnnxTextSessionContract.TensorNames(
                "input_ids",
                "attention_mask",
                "position_ids",
                "logits",
                "past_key_values.%d.key",
                "past_key_values.%d.value",
                "present.%d.key",
                "present.%d.value");
    }
}
