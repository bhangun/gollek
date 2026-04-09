package tech.kayys.gollek.ml.tensor;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;

/**
 * Off-heap tensor storage using JDK 25 Foreign Function & Memory (FFM) API.
 *
 * <p>Allocates tensor data in native memory via {@link MemorySegment}, enabling:
 * <ul>
 *   <li>Zero-copy sharing with native libraries (llama.cpp, ONNX Runtime, etc.)</li>
 *   <li>Direct DMA transfers to GPU without JVM heap involvement</li>
 *   <li>Larger-than-heap tensors (model weights, activations)</li>
 * </ul>
 *
 * <p>Memory is managed via a {@link Arena} — use {@link Arena#ofConfined()} for
 * single-thread or {@link Arena#ofShared()} for multi-thread access.
 */
public final class NativeTensorStorage implements AutoCloseable {

    private final MemorySegment segment;
    private final Arena arena;
    private final long numel;

    private NativeTensorStorage(long numel, Arena arena) {
        this.numel = numel;
        this.arena = arena;
        this.segment = arena.allocate(numel * Float.BYTES, Float.BYTES);
    }

    /** Allocate off-heap storage for {@code numel} floats (confined arena). */
    public static NativeTensorStorage allocate(long numel) {
        return new NativeTensorStorage(numel, Arena.ofConfined());
    }

    /** Allocate off-heap storage with a shared (thread-safe) arena. */
    public static NativeTensorStorage allocateShared(long numel) {
        return new NativeTensorStorage(numel, Arena.ofShared());
    }

    /** Wrap existing on-heap float[] into a temporary off-heap view (auto arena). */
    public static NativeTensorStorage fromArray(float[] data) {
        NativeTensorStorage storage = allocate(data.length);
        storage.copyFrom(data);
        return storage;
    }

    // ── Read / Write ─────────────────────────────────────────────────────

    public float get(long index) {
        return segment.get(ValueLayout.JAVA_FLOAT, index * Float.BYTES);
    }

    public void set(long index, float value) {
        segment.set(ValueLayout.JAVA_FLOAT, index * Float.BYTES, value);
    }

    /** Bulk copy from on-heap float[]. */
    public void copyFrom(float[] src) {
        MemorySegment.copy(MemorySegment.ofArray(src), 0L,
                           segment, 0L, (long) src.length * Float.BYTES);
    }

    /** Bulk copy to on-heap float[]. */
    public void copyTo(float[] dst) {
        MemorySegment.copy(segment, 0L,
                           MemorySegment.ofArray(dst), 0L, numel * Float.BYTES);
    }

    /** Return a float[] snapshot (copies data to heap). */
    public float[] toArray() {
        float[] out = new float[(int) numel];
        copyTo(out);
        return out;
    }

    // ── Slice / View ─────────────────────────────────────────────────────

    /**
     * Return a zero-copy slice of this storage.
     * The slice shares the same native memory — no allocation.
     */
    public MemorySegment slice(long fromIndex, long length) {
        return segment.asSlice(fromIndex * Float.BYTES, length * Float.BYTES);
    }

    /** Raw segment — for passing to native libraries via FFM. */
    public MemorySegment segment() {
        return segment;
    }

    public long numel() {
        return numel;
    }

    @Override
    public void close() {
        arena.close();
    }
}
