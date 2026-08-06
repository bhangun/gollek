package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class OnnxTextWorkspacePoolTest {

    @Test
    void reusesCompatibleIdleWorkspace() {
        FakeFactory factory = new FakeFactory();
        OnnxTextWorkspacePool pool = OnnxTextWorkspacePool.createForTest(factory, 1);

        OnnxTextWorkspacePool.Lease first = pool.acquire(2, 2);
        OnnxTextRunWorkspace firstWorkspace = first.workspace();
        assertFalse(first.reused());
        assertEquals(0, first.evicted());
        first.close();
        OnnxTextWorkspacePool.Lease second = pool.acquire(1, 1);

        assertTrue(second.reused());
        assertEquals(0, second.evicted());
        assertEquals(OnnxTextRunWorkspace.capacityFor(1, 1), second.requestedCapacity());
        assertEquals(OnnxTextRunWorkspace.capacityFor(2, 2), second.workspaceCapacity());
        assertSame(firstWorkspace, second.workspace());
        assertEquals(1, factory.created.size());

        second.close();
        pool.close();
        factory.closeArenas();
    }

    @Test
    void replacesSmallerWorkspaceWhenLargerWorkspaceReturnsToFullPool() {
        FakeFactory factory = new FakeFactory();
        OnnxTextWorkspacePool pool = OnnxTextWorkspacePool.createForTest(factory, 1);

        OnnxTextWorkspacePool.Lease first = pool.acquire(1, 1);
        OnnxTextRunWorkspace firstWorkspace = first.workspace();
        first.close();
        OnnxTextWorkspacePool.Lease second = pool.acquire(4, 4);

        assertFalse(second.reused());
        assertEquals(0, second.evicted());
        assertEquals(2, factory.created.size());
        assertSame(factory.created.get(1), second.workspace());

        second.close();
        assertEquals(1, second.evicted());
        assertThrows(IllegalStateException.class, firstWorkspace::beginRequest);

        OnnxTextWorkspacePool.Lease third = pool.acquire(1, 1);
        assertTrue(third.reused());
        assertSame(factory.created.get(1), third.workspace());
        third.close();

        pool.close();
        factory.closeArenas();
    }

    @Test
    void reportsEvictionWhenReturnedWorkspaceIsDiscardedFromFullPool() {
        FakeFactory factory = new FakeFactory();
        OnnxTextWorkspacePool pool = OnnxTextWorkspacePool.createForTest(factory, 1);

        OnnxTextWorkspacePool.Lease small = pool.acquire(1, 1);
        OnnxTextRunWorkspace smallWorkspace = small.workspace();
        OnnxTextWorkspacePool.Lease large = pool.acquire(5, 5);
        large.close();

        small.close();

        assertEquals(0, large.evicted());
        assertEquals(1, small.evicted());
        assertThrows(IllegalStateException.class, smallWorkspace::beginRequest);

        pool.close();
        factory.closeArenas();
    }

    @Test
    void findsCompatibleWorkspaceWithoutEvictingSmallerIdleWorkspace() {
        FakeFactory factory = new FakeFactory();
        OnnxTextWorkspacePool pool = OnnxTextWorkspacePool.createForTest(factory, 2);

        OnnxTextWorkspacePool.Lease small = pool.acquire(1, 1);
        OnnxTextRunWorkspace smallWorkspace = small.workspace();
        small.close();
        OnnxTextWorkspacePool.Lease large = pool.acquire(5, 5);
        OnnxTextRunWorkspace largeWorkspace = large.workspace();
        large.close();

        OnnxTextWorkspacePool.Lease smallFirst = pool.acquire(1, 1);
        assertSame(smallWorkspace, smallFirst.workspace());
        smallFirst.close();

        OnnxTextWorkspacePool.Lease largeAgain = pool.acquire(4, 4);
        assertTrue(largeAgain.reused());
        assertEquals(0, largeAgain.evicted());
        assertSame(largeWorkspace, largeAgain.workspace());
        assertEquals(1, pool.idleCount());
        largeAgain.close();

        OnnxTextWorkspacePool.Lease smallAgain = pool.acquire(1, 1);
        assertTrue(smallAgain.reused());
        assertSame(smallWorkspace, smallAgain.workspace());
        smallAgain.close();

        pool.close();
        factory.closeArenas();
    }

    @Test
    void poolSizeZeroDestroysWorkspaceOnRelease() {
        FakeFactory factory = new FakeFactory();
        OnnxTextWorkspacePool pool = OnnxTextWorkspacePool.createForTest(factory, 0);

        OnnxTextWorkspacePool.Lease lease = pool.acquire(1, 1);
        OnnxTextRunWorkspace workspace = lease.workspace();
        lease.close();

        assertEquals(0, pool.idleCount());
        assertThrows(IllegalStateException.class, workspace::beginRequest);

        pool.close();
        factory.closeArenas();
    }

    @Test
    void rejectsAcquireAfterClose() {
        FakeFactory factory = new FakeFactory();
        OnnxTextWorkspacePool pool = OnnxTextWorkspacePool.createForTest(factory, 1);

        pool.close();

        assertThrows(IllegalStateException.class, () -> pool.acquire(1, 1));
        factory.closeArenas();
    }

    private static final class FakeFactory implements OnnxTextWorkspacePool.WorkspaceFactory {
        private final List<Arena> arenas = new ArrayList<>();
        private final List<OnnxTextRunWorkspace> created = new ArrayList<>();

        @Override
        public OnnxTextRunWorkspace create(int promptLength, int maxTokens) {
            Arena arena = Arena.ofConfined();
            arenas.add(arena);
            int capacity = OnnxTextRunWorkspace.capacityFor(promptLength, maxTokens);
            OnnxRepeatedInputValue emptyPastKvInput = OnnxRepeatedInputValue.create(0, () -> MemorySegment.NULL,
                    ignored -> {
                    });
            OnnxRunOutputValues outputValues = OnnxRunOutputValues.allocate(1);
            OnnxTextRunWorkspace workspace = OnnxTextRunWorkspace.createForTest(
                    capacity,
                    false,
                    OnnxPastKvState.allocate(0),
                    inputAssembler(arena, emptyPastKvInput),
                    preparedRun(arena, outputValues),
                    outputValues,
                    OnnxRunOutputLifecycle.create(outputValues, OnnxPastKvState.allocate(0), false, ignored -> {
                    }),
                    OnnxLogitsSelector.createForTest(new FakeLogitsOps(), 1, false),
                    emptyPastKvInput,
                    ignored -> {
                    });
            created.add(workspace);
            return workspace;
        }

        void closeArenas() {
            for (Arena arena : arenas) {
                arena.close();
            }
        }

        private static OnnxTextInputAssembler inputAssembler(Arena arena, OnnxRepeatedInputValue emptyPastKvInput) {
            return OnnxTextInputAssembler.createForTest(
                    new FakeInputOps(arena),
                    arena.allocate(Long.BYTES),
                    arena.allocate(2L * Long.BYTES),
                    arena.allocate(Long.BYTES),
                    OnnxRunInputValues.allocate(2, 2),
                    OnnxInputIdsScratch.allocate(arena, 8),
                    OnnxAttentionMaskScratch.allocate(arena, 8),
                    null,
                    false,
                    0,
                    emptyPastKvInput);
        }

        private static OnnxPreparedRun preparedRun(Arena arena, OnnxRunOutputValues outputValues) {
            return OnnxPreparedRun.createForTest(
                    new FakeRunOps(arena.allocate(Long.BYTES)),
                    arena.allocate(Long.BYTES),
                    MemorySegment.NULL,
                    arena.allocate(Long.BYTES),
                    arena.allocate(Long.BYTES),
                    arena.allocate(Long.BYTES),
                    arena.allocate(Long.BYTES),
                    outputValues);
        }
    }

    private static final class FakeInputOps implements OnnxTextInputAssembler.Ops {
        private final Arena arena;

        private FakeInputOps(Arena arena) {
            this.arena = arena;
        }

        @Override
        public void writeShape2d(MemorySegment shapeBuffer, long firstDim, long secondDim) {
            // Shape writing is covered by OnnxTextInputAssemblerTest.
        }

        @Override
        public MemorySegment createInt64Tensor(
                MemorySegment memInfo,
                OnnxTensorDataView view,
                MemorySegment shapeBuffer,
                MemorySegment valueOutPointer) {
            return arena.allocate(Long.BYTES);
        }
    }

    private static final class FakeRunOps implements OnnxPreparedRun.Ops {
        private final MemorySegment logits;

        private FakeRunOps(MemorySegment logits) {
            this.logits = logits;
        }

        @Override
        public void writePointerArray(MemorySegment pointerArray, MemorySegment[] values, int count) {
            // Pointer writing is covered by OnnxPreparedRunTest.
        }

        @Override
        public void runWithPreparedPointers(
                MemorySegment session,
                MemorySegment runOptions,
                MemorySegment inputNamePointers,
                MemorySegment inputValuePointers,
                int inputCount,
                MemorySegment outputNamePointers,
                MemorySegment outputValuePointers,
                MemorySegment[] outputValues) {
            outputValues[0] = logits;
        }
    }

    private static final class FakeLogitsOps implements OnnxLogitsSelector.Ops {
        @Override
        public OnnxRuntimeBinding.LogitsTailPlan createTailPlan(
                MemorySegment logits,
                int fallbackWidth,
                long sequenceLength) {
            return new OnnxRuntimeBinding.LogitsTailPlan(
                    OnnxRuntimeBinding.ONNX_TENSOR_FLOAT,
                    fallbackWidth,
                    true);
        }

        @Override
        public int argmaxLast(MemorySegment logits, int fallbackWidth) {
            return 0;
        }

        @Override
        public int argmaxLast(
                MemorySegment logits,
                OnnxRuntimeBinding.LogitsTailPlan tailPlan,
                long sequenceLength) {
            return 0;
        }
    }
}
