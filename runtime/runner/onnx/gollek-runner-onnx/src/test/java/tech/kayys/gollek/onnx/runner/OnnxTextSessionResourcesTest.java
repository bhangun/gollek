package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;

class OnnxTextSessionResourcesTest {

    @Test
    void exposesContractPreparedNamesAndKvShapeSettings() {
        OnnxTextSessionContract contract = contract();
        OnnxTextPreparedRunNames preparedNames = preparedNames(contract);
        OnnxTextSessionResources resources = OnnxTextSessionResources.createForTest(
                contract,
                preparedNames,
                1024,
                4,
                128,
                OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16);

        assertSame(contract, resources.contract());
        assertSame(preparedNames, resources.preparedRunNames());
        assertEquals(1024, resources.vocabSize());
        assertEquals(4, resources.kvHeads());
        assertEquals(128, resources.kvHeadSize());
        assertEquals(OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16, resources.kvElementType());

        resources.close();
        resources.close();

        assertThrows(IllegalStateException.class, resources::preparedRunNames);
        assertThrows(IllegalStateException.class, preparedNames::inputNamePointers);
    }

    @Test
    void emptyResourcesDoNotExposePreparedNames() {
        OnnxTextSessionResources resources = OnnxTextSessionResources.empty();

        assertThrows(IllegalStateException.class, resources::preparedRunNames);

        resources.close();
        resources.close();
    }

    @Test
    void validatesInputs() {
        OnnxTextSessionContract contract = contract();
        OnnxTextPreparedRunNames preparedNames = preparedNames(contract);
        try {
            assertThrows(NullPointerException.class,
                    () -> OnnxTextSessionResources.createForTest(null, preparedNames, 1, 1, 1,
                            OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16));
            assertThrows(NullPointerException.class,
                    () -> OnnxTextSessionResources.createForTest(contract, null, 1, 1, 1,
                            OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16));
            assertThrows(IllegalArgumentException.class,
                    () -> OnnxTextSessionResources.createForTest(contract, preparedNames, 0, 1, 1,
                            OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16));
            assertThrows(IllegalArgumentException.class,
                    () -> OnnxTextSessionResources.createForTest(contract, preparedNames, 1, 0, 1,
                            OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16));
            assertThrows(IllegalArgumentException.class,
                    () -> OnnxTextSessionResources.createForTest(contract, preparedNames, 1, 1, 0,
                            OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16));
        } finally {
            preparedNames.close();
        }
    }

    private static OnnxTextPreparedRunNames preparedNames(OnnxTextSessionContract contract) {
        return OnnxTextPreparedRunNames.createForTest(
                (arena, names) -> arena.allocate(Long.BYTES),
                Arena.ofConfined(),
                contract);
    }

    private static OnnxTextSessionContract contract() {
        return OnnxTextSessionContract.create(
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
                new OnnxTextSessionContract.TensorNames(
                        "input_ids",
                        "attention_mask",
                        "position_ids",
                        "logits",
                        "past_key_values.%d.key",
                        "past_key_values.%d.value",
                        "present.%d.key",
                        "present.%d.value"),
                true,
                1);
    }
}
