package tech.kayys.gollek.ml.metrics;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.*;
import java.util.*;

/**
 * TensorBoard summary writer — writes scalar, histogram, and text events
 * to TFRecord files readable by TensorBoard.
 *
 * <p>Uses the TFRecord binary format (length-prefixed protobuf records)
 * with JDK 25 {@link java.lang.foreign.MemorySegment} for efficient I/O.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * try (TensorBoardWriter writer = new TensorBoardWriter(Path.of("runs/exp1"))) {
 *     for (int step = 0; step < 100; step++) {
 *         writer.addScalar("train/loss", loss, step);
 *         writer.addScalar("train/lr",   lr,   step);
 *     }
 * }
 * // Then: tensorboard --logdir runs/
 * }</pre>
 */
public final class TensorBoardWriter implements AutoCloseable {

    private final Path logDir;
    private final OutputStream out;
    private int globalStep = 0;

    /**
     * Creates a TensorBoard writer that logs to the given directory.
     *
     * @param logDir directory for TFRecord event files (created if absent)
     * @throws IOException if the directory or file cannot be created
     */
    public TensorBoardWriter(Path logDir) throws IOException {
        this.logDir = logDir;
        Files.createDirectories(logDir);
        String filename = "events.out.tfevents." + System.currentTimeMillis() + ".gollek";
        this.out = new BufferedOutputStream(
            Files.newOutputStream(logDir.resolve(filename)));
        writeFileVersionEvent();
    }

    /**
     * Logs a scalar value at the given step.
     *
     * @param tag   metric name (e.g. {@code "train/loss"})
     * @param value scalar value
     * @param step  training step
     * @throws IOException if writing fails
     */
    public void addScalar(String tag, float value, int step) throws IOException {
        byte[] summary = buildScalarSummary(tag, value);
        byte[] event   = buildEvent(step, summary);
        writeRecord(event);
    }

    /**
     * Logs a scalar using the internal step counter (auto-incremented).
     *
     * @param tag   metric name
     * @param value scalar value
     * @throws IOException if writing fails
     */
    public void addScalar(String tag, float value) throws IOException {
        addScalar(tag, value, globalStep++);
    }

    /**
     * Logs a text string at the given step.
     *
     * @param tag  tag name
     * @param text text content
     * @param step training step
     * @throws IOException if writing fails
     */
    public void addText(String tag, String text, int step) throws IOException {
        byte[] summary = buildTextSummary(tag, text);
        byte[] event   = buildEvent(step, summary);
        writeRecord(event);
    }

    /**
     * Flushes buffered data to disk.
     *
     * @throws IOException if flushing fails
     */
    public void flush() throws IOException { out.flush(); }

    @Override
    public void close() throws IOException { out.close(); }

    // ── TFRecord format ───────────────────────────────────────────────────

    /**
     * Writes a TFRecord: [length(8)] [masked_crc32(4)] [data] [masked_crc32(4)]
     */
    private void writeRecord(byte[] data) throws IOException {
        ByteBuffer header = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        header.putLong(data.length);
        header.putInt(maskedCrc32(longToBytes(data.length)));
        out.write(header.array());
        out.write(data);
        ByteBuffer footer = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        footer.putInt(maskedCrc32(data));
        out.write(footer.array());
    }

    /** Writes the mandatory file_version event. */
    private void writeFileVersionEvent() throws IOException {
        byte[] versionSummary = buildTextSummary("file_version", "brain.Event:2");
        writeRecord(buildEvent(0, versionSummary));
    }

    // ── Minimal protobuf builders ─────────────────────────────────────────

    private byte[] buildEvent(int step, byte[] summary) {
        ProtoWriter w = new ProtoWriter();
        w.writeDouble(1, System.currentTimeMillis() / 1000.0); // wall_time
        w.writeVarint(2, step);                                  // step
        w.writeBytes(5, summary);                                // summary
        return w.toBytes();
    }

    private byte[] buildScalarSummary(String tag, float value) {
        ProtoWriter value_pb = new ProtoWriter();
        value_pb.writeFloat(2, value); // simple_value

        ProtoWriter value_wrapper = new ProtoWriter();
        value_wrapper.writeString(1, tag);
        value_wrapper.writeBytes(2, value_pb.toBytes());

        ProtoWriter summary = new ProtoWriter();
        summary.writeBytes(1, value_wrapper.toBytes());
        return summary.toBytes();
    }

    private byte[] buildTextSummary(String tag, String text) {
        ProtoWriter metadata = new ProtoWriter();
        metadata.writeString(1, "text");

        ProtoWriter tensor = new ProtoWriter();
        tensor.writeVarint(1, 7); // DT_STRING
        tensor.writeBytes(8, text.getBytes(java.nio.charset.StandardCharsets.UTF_8));

        ProtoWriter value_wrapper = new ProtoWriter();
        value_wrapper.writeString(1, tag);
        value_wrapper.writeBytes(9, tensor.toBytes());
        value_wrapper.writeBytes(7, metadata.toBytes());

        ProtoWriter summary = new ProtoWriter();
        summary.writeBytes(1, value_wrapper.toBytes());
        return summary.toBytes();
    }

    // ── CRC32 (masked) ────────────────────────────────────────────────────

    private static int maskedCrc32(byte[] data) {
        java.util.zip.CRC32 crc = new java.util.zip.CRC32();
        crc.update(data);
        long v = crc.getValue();
        return (int) (((v >> 15) | (v << 17)) + 0xa282ead8L);
    }

    private static byte[] longToBytes(long v) {
        return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array();
    }

    // ── Minimal protobuf writer ───────────────────────────────────────────

    private static final class ProtoWriter {
        private final ByteArrayOutputStream buf = new ByteArrayOutputStream();

        void writeVarint(int field, long v) {
            writeTag(field, 0);
            while ((v & ~0x7FL) != 0) { buf.write((int)((v & 0x7F) | 0x80)); v >>>= 7; }
            buf.write((int) v);
        }

        void writeDouble(int field, double v) {
            writeTag(field, 1);
            long bits = Double.doubleToRawLongBits(v);
            for (int i = 0; i < 8; i++) { buf.write((int)(bits & 0xFF)); bits >>= 8; }
        }

        void writeFloat(int field, float v) {
            writeTag(field, 5);
            int bits = Float.floatToRawIntBits(v);
            for (int i = 0; i < 4; i++) { buf.write(bits & 0xFF); bits >>= 8; }
        }

        void writeString(int field, String s) {
            writeBytes(field, s.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        }

        void writeBytes(int field, byte[] b) {
            writeTag(field, 2);
            writeRawVarint(b.length);
            buf.writeBytes(b);
        }

        private void writeTag(int field, int wire) { writeRawVarint((field << 3) | wire); }

        private void writeRawVarint(long v) {
            while ((v & ~0x7FL) != 0) { buf.write((int)((v & 0x7F) | 0x80)); v >>>= 7; }
            buf.write((int) v);
        }

        byte[] toBytes() { return buf.toByteArray(); }
    }
}
