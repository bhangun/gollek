package tech.kayys.gollek.converter.java.gguf;

import java.io.*;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.Map;

/**
 * Writes a {@link GgufModel} to a GGUF v3 file using the
 * JDK 25 Foreign Function &amp; Memory API.
 *
 * <p>
 * Strategy (following the C reference implementation):
 * <ol>
 * <li>Serialize header + KV + tensor-info into an off-heap
 * {@link MemorySegment}.</li>
 * <li>Pad to alignment boundary.</li>
 * <li>Write tensor data blobs one by one (padded per tensor).</li>
 * <li>Use a writable memory-mapped region for the meta section so the
 * final tensor count can be patched in place.</li>
 * </ol>
 */
public final class GgufWriter {

    private static final int INITIAL_META_CAPACITY = 1 << 20; // 1 MiB
    private static final String UTF8 = "UTF-8";

    /** Write {@code model} to {@code dest}. */
    public static void write(GgufModel model, Path dest) throws IOException {

        // ── 1. Build metadata + tensor-info bytes in off-heap memory ──
        try (Arena arena = Arena.ofConfined()) {

            // Allocate a growable wrapper backed by a resizable segment
            GrowableSegment buf = new GrowableSegment(arena, INITIAL_META_CAPACITY);

            // Header
            buf.writeBytes(GgufModel.MAGIC_BYTES); // "GGUF"
            buf.writeI32LE(GgufModel.VERSION); // version = 3
            buf.writeI64LE((long) model.tensors().size()); // tensor count
            buf.writeI64LE((long) model.metadata().size()); // kv count

            // Metadata key-value pairs
            for (Map.Entry<String, GgufMetaValue> e : model.metadata().entrySet()) {
                writeString(buf, e.getKey());
                writeValue(buf, e.getValue());
            }

            // Tensor descriptors
            for (TensorInfo ti : model.tensors()) {
                writeString(buf, ti.name());
                buf.writeI32LE(ti.nDims());
                for (long d : ti.ne())
                    buf.writeI64LE(d);
                buf.writeI32LE(ti.type().id);
                buf.writeI64LE(ti.offset());
            }

            // Alignment padding before tensor data
            int alignment = model.alignment();
            long metaLen = buf.position();
            long padLen = alignUp(metaLen, alignment) - metaLen;
            for (long i = 0; i < padLen; i++)
                buf.writeByte((byte) 0);

            // ── 2. Write everything to disk ───────────────────────────
            try (FileOutputStream fos = new FileOutputStream(dest.toFile());
                    FileChannel fc = fos.getChannel()) {

                // Write meta segment
                MemorySegment metaSeg = buf.segment().asSlice(0, buf.position());
                fc.write(metaSeg.asByteBuffer());

                // Write tensor data blob
                byte[] blob = model.tensorData();
                if (blob != null && blob.length > 0) {
                    // Write each tensor's data with per-tensor alignment padding
                    long blobOffset = 0;
                    for (TensorInfo ti : model.tensors()) {
                        long size = ti.dataSize();
                        fos.write(blob, (int) blobOffset, (int) size);
                        blobOffset += size;
                        // Pad to alignment
                        long tpad = alignUp(blobOffset, alignment) - blobOffset;
                        for (long p = 0; p < tpad; p++)
                            fos.write(0);
                        blobOffset = alignUp(blobOffset, alignment);
                    }
                }
            }
        }
    }

    // ── Serialization helpers ─────────────────────────────────────────────

    private static void writeString(GrowableSegment buf, String s) throws IOException {
        byte[] bytes = s.getBytes(UTF8);
        buf.writeI64LE(bytes.length); // uint64 length prefix
        buf.writeBytes(bytes);
    }

    private static void writeValue(GrowableSegment buf, GgufMetaValue val) throws IOException {
        buf.writeI32LE(val.type().id);
        switch (val) {
            case GgufMetaValue.UInt8Val v -> buf.writeByte((byte) (v.value() & 0xFF));
            case GgufMetaValue.Int8Val v -> buf.writeByte(v.value());
            case GgufMetaValue.UInt16Val v -> buf.writeI16LE((short) (v.value() & 0xFFFF));
            case GgufMetaValue.Int16Val v -> buf.writeI16LE(v.value());
            case GgufMetaValue.UInt32Val v -> buf.writeI32LE((int) v.value());
            case GgufMetaValue.Int32Val v -> buf.writeI32LE(v.value());
            case GgufMetaValue.Float32Val v -> buf.writeF32LE(v.value());
            case GgufMetaValue.BoolVal v -> buf.writeByte(v.value() ? (byte) 1 : (byte) 0);
            case GgufMetaValue.StringVal v -> {
                buf.writeI64LE(v.value().getBytes(UTF8).length);
                buf.writeBytes(v.value().getBytes(UTF8));
            }
            case GgufMetaValue.UInt64Val v -> buf.writeI64LE(v.value());
            case GgufMetaValue.Int64Val v -> buf.writeI64LE(v.value());
            case GgufMetaValue.Float64Val v -> buf.writeF64LE(v.value());
            case GgufMetaValue.ArrayVal v -> {
                buf.writeI32LE(v.elementType().id);
                buf.writeI64LE(v.elements().size());
                // Recurse – but suppress the outer type prefix for elements
                for (GgufMetaValue elem : v.elements()) {
                    writeArrayElement(buf, elem);
                }
            }
        }
    }

    /**
     * Like {@link #writeValue} but without the leading type tag (for array
     * elements).
     */
    private static void writeArrayElement(GrowableSegment buf, GgufMetaValue val)
            throws IOException {
        switch (val) {
            case GgufMetaValue.UInt8Val v -> buf.writeByte((byte) (v.value() & 0xFF));
            case GgufMetaValue.Int8Val v -> buf.writeByte(v.value());
            case GgufMetaValue.UInt16Val v -> buf.writeI16LE((short) (v.value() & 0xFFFF));
            case GgufMetaValue.Int16Val v -> buf.writeI16LE(v.value());
            case GgufMetaValue.UInt32Val v -> buf.writeI32LE((int) v.value());
            case GgufMetaValue.Int32Val v -> buf.writeI32LE(v.value());
            case GgufMetaValue.Float32Val v -> buf.writeF32LE(v.value());
            case GgufMetaValue.BoolVal v -> buf.writeByte(v.value() ? (byte) 1 : (byte) 0);
            case GgufMetaValue.StringVal v -> {
                byte[] b = v.value().getBytes(UTF8);
                buf.writeI64LE(b.length);
                buf.writeBytes(b);
            }
            case GgufMetaValue.UInt64Val v -> buf.writeI64LE(v.value());
            case GgufMetaValue.Int64Val v -> buf.writeI64LE(v.value());
            case GgufMetaValue.Float64Val v -> buf.writeF64LE(v.value());
            case GgufMetaValue.ArrayVal v -> {
                buf.writeI32LE(v.elementType().id);
                buf.writeI64LE(v.elements().size());
                for (GgufMetaValue e : v.elements())
                    writeArrayElement(buf, e);
            }
        }
    }

    private static long alignUp(long offset, long alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    // ── GrowableSegment – off-heap byte buffer backed by FFM ──────────────

    /**
     * An off-heap byte buffer that doubles in capacity when full.
     * Uses {@link MemorySegment} from the FFM API for zero-copy writes.
     */
    private static final class GrowableSegment {
        private final Arena arena;
        private MemorySegment seg;
        private long pos;

        GrowableSegment(Arena arena, long initialCapacity) {
            this.arena = arena;
            this.seg = arena.allocate(initialCapacity);
        }

        long position() {
            return pos;
        }

        MemorySegment segment() {
            return seg;
        }

        void ensureCapacity(long extra) {
            if (pos + extra <= seg.byteSize())
                return;
            long newCap = Math.max(seg.byteSize() * 2, pos + extra);
            MemorySegment grown = arena.allocate(newCap);
            MemorySegment.copy(seg, 0, grown, 0, pos);
            seg = grown;
        }

        void writeByte(byte b) {
            ensureCapacity(1);
            seg.set(ValueLayout.JAVA_BYTE, pos, b);
            pos++;
        }

        void writeBytes(byte[] b) {
            ensureCapacity(b.length);
            MemorySegment.copy(b, 0, seg, ValueLayout.JAVA_BYTE, pos, b.length);
            pos += b.length;
        }

        void writeI16LE(short v) {
            ensureCapacity(2);
            seg.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos, v);
            pos += 2;
        }

        void writeI32LE(int v) {
            ensureCapacity(4);
            seg.set(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos, v);
            pos += 4;
        }

        void writeI64LE(long v) {
            ensureCapacity(8);
            seg.set(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos, v);
            pos += 8;
        }

        void writeF32LE(float v) {
            ensureCapacity(4);
            seg.set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos, v);
            pos += 4;
        }

        void writeF64LE(double v) {
            ensureCapacity(8);
            seg.set(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos, v);
            pos += 8;
        }
    }
}
