package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;

class OnnxTensorDataViewTest {

    @Test
    void tracksLogicalByteLengthSeparatelyFromBackingSegment() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment data = arena.allocate(4L * Long.BYTES, Long.BYTES);

            OnnxTensorDataView view = OnnxTensorDataView.int64Elements(data, 2);

            assertSame(data, view.data());
            assertEquals(2L * Long.BYTES, view.byteLength());
            assertEquals(2L * Long.BYTES, view.asSizedSegment().byteSize());
        }
    }

    @Test
    void rejectsInvalidLogicalLengths() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment data = arena.allocate(8);

            assertThrows(IllegalArgumentException.class, () -> new OnnxTensorDataView(data, -1));
            assertThrows(IllegalArgumentException.class, () -> new OnnxTensorDataView(data, 9));
            assertThrows(IllegalArgumentException.class, () -> OnnxTensorDataView.int64Elements(data, -1));
        }
    }
}
