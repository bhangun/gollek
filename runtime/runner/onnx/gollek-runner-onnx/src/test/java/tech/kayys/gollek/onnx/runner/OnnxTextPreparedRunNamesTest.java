package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class OnnxTextPreparedRunNamesTest {

    @Test
    void packsRunNamesOnceForSessionLifetime() {
        Arena arena = Arena.ofConfined();
        FakeOps ops = new FakeOps();
        OnnxTextPreparedRunNames names = OnnxTextPreparedRunNames.createForTest(
                ops,
                arena,
                contract());

        assertSame(ops.packedPointers.get(0), names.inputNamePointers());
        assertSame(ops.packedPointers.get(1), names.outputNamePointers());
        assertEquals(4, names.inputCount());
        assertEquals(3, names.outputCount());
        assertArrayEquals(new String[] {
                "input_ids",
                "attention_mask",
                "past_key_values.0.key",
                "past_key_values.0.value"
        }, ops.packedNames.get(0));
        assertArrayEquals(new String[] {
                "logits",
                "present.0.key",
                "present.0.value"
        }, ops.packedNames.get(1));

        names.close();
        names.close();

        assertThrows(IllegalStateException.class, names::inputNamePointers);
        assertThrows(IllegalStateException.class, names::outputNamePointers);
    }

    @Test
    void validatesInputs() {
        Arena arena = Arena.ofConfined();
        try {
            assertThrows(NullPointerException.class,
                    () -> OnnxTextPreparedRunNames.createForTest(null, arena, contract()));
            assertThrows(NullPointerException.class,
                    () -> OnnxTextPreparedRunNames.createForTest(new FakeOps(), null, contract()));
            assertThrows(NullPointerException.class,
                    () -> OnnxTextPreparedRunNames.createForTest(new FakeOps(), arena, null));
        } finally {
            arena.close();
        }
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

    private static final class FakeOps implements OnnxTextPreparedRunNames.Ops {
        private final List<String[]> packedNames = new ArrayList<>();
        private final List<MemorySegment> packedPointers = new ArrayList<>();

        @Override
        public MemorySegment packStringPointers(Arena arena, String[] names) {
            packedNames.add(names.clone());
            MemorySegment pointers = arena.allocate(Long.BYTES);
            packedPointers.add(pointers);
            return pointers;
        }
    }
}
