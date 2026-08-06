package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class OnnxTextPreparedRunNames implements AutoCloseable {

    private final Arena arena;
    private final MemorySegment inputNamePointers;
    private final MemorySegment outputNamePointers;
    private final int inputCount;
    private final int outputCount;
    private boolean closed;

    private OnnxTextPreparedRunNames(
            Arena arena,
            MemorySegment inputNamePointers,
            MemorySegment outputNamePointers,
            int inputCount,
            int outputCount) {
        this.arena = Objects.requireNonNull(arena, "arena");
        this.inputNamePointers = Objects.requireNonNull(inputNamePointers, "inputNamePointers");
        this.outputNamePointers = Objects.requireNonNull(outputNamePointers, "outputNamePointers");
        if (inputCount < 0) {
            throw new IllegalArgumentException("inputCount must be >= 0");
        }
        if (outputCount <= 0) {
            throw new IllegalArgumentException("outputCount must be > 0");
        }
        this.inputCount = inputCount;
        this.outputCount = outputCount;
    }

    static OnnxTextPreparedRunNames create(OnnxRuntimeBinding binding, OnnxTextSessionContract contract) {
        Objects.requireNonNull(binding, "binding");
        Arena arena = Arena.ofShared();
        try {
            return createForArena(new BindingOps(binding), arena, contract);
        } catch (RuntimeException | Error e) {
            arena.close();
            throw e;
        }
    }

    static OnnxTextPreparedRunNames createForTest(Ops ops, Arena arena, OnnxTextSessionContract contract) {
        return createForArena(ops, arena, contract);
    }

    private static OnnxTextPreparedRunNames createForArena(
            Ops ops,
            Arena arena,
            OnnxTextSessionContract contract) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(arena, "arena");
        Objects.requireNonNull(contract, "contract");
        String[] runInputNames = contract.runInputNames();
        String[] runOutputNames = contract.runOutputNames();
        return new OnnxTextPreparedRunNames(
                arena,
                ops.packStringPointers(arena, runInputNames),
                ops.packStringPointers(arena, runOutputNames),
                runInputNames.length,
                runOutputNames.length);
    }

    MemorySegment inputNamePointers() {
        ensureOpen();
        return inputNamePointers;
    }

    MemorySegment outputNamePointers() {
        ensureOpen();
        return outputNamePointers;
    }

    int inputCount() {
        return inputCount;
    }

    int outputCount() {
        return outputCount;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        arena.close();
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("ONNX text prepared run names are already closed");
        }
    }

    interface Ops {
        MemorySegment packStringPointers(Arena arena, String[] names);
    }

    private record BindingOps(OnnxRuntimeBinding binding) implements Ops {
        private BindingOps {
            Objects.requireNonNull(binding, "binding");
        }

        @Override
        public MemorySegment packStringPointers(Arena arena, String[] names) {
            return binding.packStringPointers(arena, names);
        }
    }
}
