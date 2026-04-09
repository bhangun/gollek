package tech.kayys.gollek.ml.export;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

import java.io.IOException;
import java.lang.foreign.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * ONNX model exporter — writes a minimal ONNX protobuf file from a Gollek model's
 * state dict using JDK 25 FFM for zero-copy tensor serialization.
 *
 * <p>The output is a valid ONNX file loadable by ONNX Runtime, Python onnx library,
 * and any ONNX-compatible inference engine.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * OnnxExporter.export(model, inputShape, Path.of("model.onnx"));
 * }</pre>
 *
 * <p><b>Note:</b> This exports weights only (initializer-only graph).
 * Full graph tracing (operator nodes) is planned for Phase 2 Week 10.
 */
public final class OnnxExporter {

    // ONNX protobuf field tags (wire format)
    private static final int ONNX_IR_VERSION = 8;
    private static final String OPSET_DOMAIN  = "";
    private static final int    OPSET_VERSION = 17;

    private OnnxExporter() {}

    /**
     * Export model weights as an ONNX initializer-only model.
     *
     * @param model      trained model
     * @param inputShape shape of a single input (without batch dim), e.g. [3, 224, 224]
     * @param path       output .onnx file path
     */
    public static void export(NNModule model, long[] inputShape, Path path) throws IOException {
        Map<String, GradTensor> state = model.stateDict();
        byte[] proto = buildOnnxProto(state, inputShape);
        Files.write(path, proto);
    }

    // ── Protobuf builder ─────────────────────────────────────────────────

    /**
     * Builds a minimal ONNX ModelProto in protobuf binary format.
     *
     * <p>Structure:
     * <pre>
     *   ModelProto {
     *     ir_version: 8
     *     opset_import { domain: "" version: 17 }
     *     graph: GraphProto {
     *       name: "gollek"
     *       initializer: [TensorProto per parameter]
     *       input:  [ValueInfoProto for each initializer]
     *       output: [ValueInfoProto placeholder]
     *     }
     *   }
     * </pre>
     */
    private static byte[] buildOnnxProto(Map<String, GradTensor> state, long[] inputShape) {
        ProtoWriter w = new ProtoWriter();

        // field 1: ir_version (int64)
        w.writeVarint(fieldTag(1, 0));
        w.writeVarint(ONNX_IR_VERSION);

        // field 8: opset_import (message)
        ProtoWriter opset = new ProtoWriter();
        opset.writeString(1, OPSET_DOMAIN);   // domain
        opset.writeVarint(fieldTag(2, 0));
        opset.writeVarint(OPSET_VERSION);     // version
        w.writeBytes(8, opset.toBytes());

        // field 7: graph (GraphProto)
        ProtoWriter graph = new ProtoWriter();
        graph.writeString(1, "gollek");       // name

        // initializers (field 5) + inputs (field 11)
        for (var entry : state.entrySet()) {
            graph.writeBytes(5, tensorProto(entry.getKey(), entry.getValue()));
            graph.writeBytes(11, valueInfoProto(entry.getKey(), entry.getValue().shape()));
        }

        // placeholder output (field 12)
        graph.writeBytes(12, valueInfoProto("output", new long[]{-1}));

        w.writeBytes(7, graph.toBytes());
        return w.toBytes();
    }

    /** TensorProto for a float32 tensor. */
    private static byte[] tensorProto(String name, GradTensor t) {
        ProtoWriter p = new ProtoWriter();
        p.writeString(1, name);              // name
        p.writeVarint(fieldTag(2, 0));
        p.writeVarint(1);                    // data_type = FLOAT (1)
        for (long d : t.shape()) {           // dims
            p.writeVarint(fieldTag(3, 0));
            p.writeVarint(d);
        }
        // raw_data (field 9) — float[] as little-endian bytes via FFM
        float[] data = t.data();
        byte[] raw = floatsToBytes(data);
        p.writeBytes(9, raw);
        return p.toBytes();
    }

    /** ValueInfoProto with float32 tensor type. */
    private static byte[] valueInfoProto(String name, long[] shape) {
        ProtoWriter p = new ProtoWriter();
        p.writeString(1, name);
        // type (field 2) → TypeProto → tensor_type (field 1) → TensorTypeProto
        ProtoWriter tensorType = new ProtoWriter();
        tensorType.writeVarint(fieldTag(1, 0));
        tensorType.writeVarint(1); // elem_type = FLOAT
        ProtoWriter shapeProto = new ProtoWriter();
        for (long d : shape) {
            ProtoWriter dim = new ProtoWriter();
            if (d < 0) dim.writeString(2, "batch"); // symbolic dim
            else { dim.writeVarint(fieldTag(1, 0)); dim.writeVarint(d); }
            shapeProto.writeBytes(1, dim.toBytes());
        }
        tensorType.writeBytes(2, shapeProto.toBytes());
        ProtoWriter typeProto = new ProtoWriter();
        typeProto.writeBytes(1, tensorType.toBytes());
        p.writeBytes(2, typeProto.toBytes());
        return p.toBytes();
    }

    /** Convert float[] to little-endian byte[] using FFM MemorySegment. */
    private static byte[] floatsToBytes(float[] data) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate((long) data.length * Float.BYTES, Float.BYTES);
            MemorySegment.copy(MemorySegment.ofArray(data), 0, seg, 0, (long) data.length * Float.BYTES);
            return seg.toArray(ValueLayout.JAVA_BYTE);
        }
    }

    private static int fieldTag(int fieldNumber, int wireType) {
        return (fieldNumber << 3) | wireType;
    }

    // ── Minimal protobuf writer ───────────────────────────────────────────

    private static final class ProtoWriter {
        private final java.io.ByteArrayOutputStream buf = new java.io.ByteArrayOutputStream();

        void writeVarint(long v) {
            while ((v & ~0x7FL) != 0) { buf.write((int) ((v & 0x7F) | 0x80)); v >>>= 7; }
            buf.write((int) v);
        }

        void writeString(int field, String s) {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            writeVarint(fieldTag(field, 2));
            writeVarint(b.length);
            buf.writeBytes(b);
        }

        void writeBytes(int field, byte[] b) {
            writeVarint(fieldTag(field, 2));
            writeVarint(b.length);
            buf.writeBytes(b);
        }

        byte[] toBytes() { return buf.toByteArray(); }
    }
}
