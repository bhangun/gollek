package tech.kayys.gollek.onnx.binding;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

class OnnxRuntimeBindingPointerBufferTest {

    @Test
    void writePointerArrayOverwritesReusablePointerBuffer() {
        OnnxRuntimeBinding binding = new OnnxRuntimeBinding();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment pointers = binding.allocatePointerArray(arena, 2);
            MemorySegment first = arena.allocate(8);
            MemorySegment second = arena.allocate(8);
            MemorySegment third = arena.allocate(8);

            binding.writePointerArray(pointers, new MemorySegment[] { first, second }, 2);

            assertEquals(first, pointers.getAtIndex(ValueLayout.ADDRESS, 0));
            assertEquals(second, pointers.getAtIndex(ValueLayout.ADDRESS, 1));

            binding.writePointerArray(pointers, new MemorySegment[] { third, first }, 2);

            assertEquals(third, pointers.getAtIndex(ValueLayout.ADDRESS, 0));
            assertEquals(first, pointers.getAtIndex(ValueLayout.ADDRESS, 1));
        }
    }

    @Test
    void uncheckedPointerArrayWriterUsesTrustedValuesWithoutValidation() {
        OnnxRuntimeBinding binding = new OnnxRuntimeBinding();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment pointers = binding.allocatePointerArray(arena, 2);
            MemorySegment first = arena.allocate(8);
            MemorySegment second = arena.allocate(8);
            MemorySegment third = arena.allocate(8);

            binding.writePointerArrayUnchecked(pointers, new MemorySegment[] { first, second }, 2);

            assertEquals(first, pointers.getAtIndex(ValueLayout.ADDRESS, 0));
            assertEquals(second, pointers.getAtIndex(ValueLayout.ADDRESS, 1));

            binding.writePointerArrayUnchecked(pointers, new MemorySegment[] { third }, 1);

            assertEquals(third, pointers.getAtIndex(ValueLayout.ADDRESS, 0));
            assertEquals(second, pointers.getAtIndex(ValueLayout.ADDRESS, 1));
        }
    }

    @Test
    void pointerArrayRejectsInvalidCounts() {
        OnnxRuntimeBinding binding = new OnnxRuntimeBinding();

        assertThrows(IllegalArgumentException.class, () -> binding.allocatePointerArray(Arena.ofAuto(), -1));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment pointers = binding.allocatePointerArray(arena, 1);
            MemorySegment value = arena.allocate(8);

            assertThrows(IllegalArgumentException.class,
                    () -> binding.writePointerArray(pointers, new MemorySegment[] { value }, 2));
            assertThrows(IllegalArgumentException.class,
                    () -> binding.writePointerArray(MemorySegment.NULL, new MemorySegment[] { value }, 1));
            NullPointerException nullValue = assertThrows(NullPointerException.class,
                    () -> binding.writePointerArray(pointers, new MemorySegment[] { null }, 1));
            assertEquals("values[0]", nullValue.getMessage());
        }
    }

    @Test
    void valuePointerSlotCanBeReusedForOutParameters() {
        OnnxRuntimeBinding binding = new OnnxRuntimeBinding();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment slot = binding.allocateValuePointer(arena);
            MemorySegment first = arena.allocate(8);
            MemorySegment second = arena.allocate(8);

            assertEquals(ValueLayout.ADDRESS.byteSize(), slot.byteSize());

            slot.set(ValueLayout.ADDRESS, 0, first);
            assertEquals(first, slot.get(ValueLayout.ADDRESS, 0));

            slot.set(ValueLayout.ADDRESS, 0, second);
            assertEquals(second, slot.get(ValueLayout.ADDRESS, 0));
        }
    }

    @Test
    void writeShapeHelpersReuseShapeBuffer() {
        OnnxRuntimeBinding binding = new OnnxRuntimeBinding();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shape = binding.allocateShapeBuffer(arena, 4);

            binding.writeShape2d(shape, 1, 128);

            assertEquals(1L, shape.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(128L, shape.getAtIndex(ValueLayout.JAVA_LONG, 1));

            binding.writeShape4d(shape, 1, 8, 0, 64);

            assertEquals(1L, shape.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(8L, shape.getAtIndex(ValueLayout.JAVA_LONG, 1));
            assertEquals(0L, shape.getAtIndex(ValueLayout.JAVA_LONG, 2));
            assertEquals(64L, shape.getAtIndex(ValueLayout.JAVA_LONG, 3));
        }
    }

    @Test
    void shapeHelpersRejectInvalidBuffers() {
        OnnxRuntimeBinding binding = new OnnxRuntimeBinding();

        assertThrows(IllegalArgumentException.class, () -> binding.allocateShapeBuffer(Arena.ofAuto(), 0));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shape = binding.allocateShapeBuffer(arena, 1);

            assertThrows(IllegalArgumentException.class, () -> binding.writeShape(shape, new long[0]));
            assertThrows(IllegalArgumentException.class, () -> binding.writeShape2d(shape, 1, 2));
            assertThrows(IllegalArgumentException.class, () -> binding.writeShape4d(MemorySegment.NULL, 1, 2, 3, 4));
        }
    }

    @Test
    void argmaxFloatDataScansFloat32SlicesWithoutMaterializingArray() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment data = arena.allocate(5L * Float.BYTES, Float.BYTES);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 0.25f);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 1, -1.0f);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 3.5f);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 3, 2.0f);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 4, 9.0f);

            int argmax = OnnxRuntimeBinding.argmaxFloatData(
                    data,
                    OnnxRuntimeBinding.ONNX_TENSOR_FLOAT,
                    1L,
                    3);

            assertEquals(1, argmax);
        }
    }

    @Test
    void argmaxFloatDataScansFloat16AndBfloat16Slices() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment float16 = arena.allocate(3L * Short.BYTES, Short.BYTES);
            float16.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x3c00); // 1.0
            float16.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x4000); // 2.0
            float16.setAtIndex(ValueLayout.JAVA_SHORT, 2, (short) 0x3e00); // 1.5

            assertEquals(1, OnnxRuntimeBinding.argmaxFloatData(
                    float16,
                    OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16,
                    0L,
                    3));

            MemorySegment bfloat16 = arena.allocate(3L * Short.BYTES, Short.BYTES);
            bfloat16.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x3f80); // 1.0
            bfloat16.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x4080); // 4.0
            bfloat16.setAtIndex(ValueLayout.JAVA_SHORT, 2, (short) 0x4000); // 2.0

            assertEquals(1, OnnxRuntimeBinding.argmaxFloatData(
                    bfloat16,
                    OnnxRuntimeBinding.ONNX_TENSOR_BFLOAT16,
                    0L,
                    3));
        }
    }

    @Test
    void argmaxFloatDataRejectsEmptySlices() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment data = arena.allocate(Float.BYTES, Float.BYTES);

            assertThrows(IllegalArgumentException.class,
                    () -> OnnxRuntimeBinding.argmaxFloatData(
                            data,
                            OnnxRuntimeBinding.ONNX_TENSOR_FLOAT,
                            0L,
                            0));
        }
    }

    @Test
    void logitsTailPlanUsesLastFullSequenceRowWhenShapeMatchesSequenceLength() {
        OnnxRuntimeBinding.LogitsTailPlan plan = OnnxRuntimeBinding.logitsTailPlanFromDimensions(
                OnnxRuntimeBinding.ONNX_TENSOR_FLOAT,
                new long[] { 1, 4, 7 },
                99,
                4);

        assertEquals(OnnxRuntimeBinding.ONNX_TENSOR_FLOAT, plan.elementType());
        assertEquals(7, plan.width());
        assertEquals(21L, plan.elementOffset(4));
    }

    @Test
    void logitsTailPlanUsesFirstRowWhenOutputIsAlreadyLastOnly() {
        OnnxRuntimeBinding.LogitsTailPlan plan = OnnxRuntimeBinding.logitsTailPlanFromDimensions(
                OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16,
                new long[] { 1, 7 },
                99,
                4);

        assertEquals(OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16, plan.elementType());
        assertEquals(7, plan.width());
        assertEquals(0L, plan.elementOffset(4));
    }

    @Test
    void argmaxFloatDataCanUseCachedLogitsPlanOffset() {
        OnnxRuntimeBinding.LogitsTailPlan plan = OnnxRuntimeBinding.logitsTailPlanFromDimensions(
                OnnxRuntimeBinding.ONNX_TENSOR_FLOAT,
                new long[] { 1, 2, 3 },
                3,
                2);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment data = arena.allocate(6L * Float.BYTES, Float.BYTES);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 9.0f);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 8.0f);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 7.0f);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 3, 0.5f);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 4, 4.0f);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, 5, 2.0f);

            assertEquals(1, OnnxRuntimeBinding.argmaxFloatData(
                    data,
                    plan.elementType(),
                    plan.elementOffset(2),
                    plan.width()));
        }
    }
}
