package tech.kayys.gollek.ml.export.litert;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.nn.Module;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Exports a Gollek {@link Module} to a TensorFlow Lite FlatBuffer
 * ({@code .tflite})
 * loadable by the LiteRT runtime.
 *
 * <p>
 * Layer → TFLite builtin op mapping:
 * <ul>
 * <li>{@code Linear} → {@code FULLY_CONNECTED} (9)</li>
 * <li>{@code ReLU} → {@code RELU} (19)</li>
 * <li>{@code GELU} → {@code GELU} (137)</li>
 * <li>{@code SiLU} → {@code HARD_SWISH} (117)</li>
 * <li>{@code LayerNorm} → {@code LAYER_NORM} (148)</li>
 * <li>{@code Softmax} → {@code SOFTMAX} (25)</li>
 * <li>{@code Dropout / Sequential} → omitted at inference</li>
 * <li>unknown → {@code CUSTOM} (32)</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * 
 * <pre>{@code
 * LiteRTExporter.fromModel(model, new long[] { 1, 784 })
 *         .export(Path.of("model.tflite"));
 * }</pre>
 */
public final class LiteRTExporter {

    // TFLite builtin op codes
    private static final int OP_FULLY_CONNECTED = 9;
    private static final int OP_RELU = 19;
    private static final int OP_SOFTMAX = 25;
    private static final int OP_GELU = 137;
    private static final int OP_HARD_SWISH = 117;
    private static final int OP_LAYER_NORM = 148;
    private static final int OP_CUSTOM = 32;
    private static final int TENSOR_FLOAT32 = 0;

    private final Module model;
    private final long[] inputShape;

    private LiteRTExporter(Module model, long[] inputShape) {
        this.model = model;
        this.inputShape = inputShape;
    }

    public static LiteRTExporter fromModel(Module model, long[] inputShape) {
        return new LiteRTExporter(model, inputShape);
    }

    public void export(Path outputPath) throws IOException {
        Files.write(outputPath, buildFlatBuffer());
    }

    // ── Graph construction ────────────────────────────────────────────────────

    private byte[] buildFlatBuffer() {
        List<float[]> buffers = new ArrayList<>();
        List<TFLiteTensor> tensors = new ArrayList<>();
        List<TFLiteOp> ops = new ArrayList<>();

        buffers.add(new float[0]); // buffer[0] = empty (TFLite convention)

        // Input tensor — no buffer, filled at runtime
        tensors.add(new TFLiteTensor("input", inputShape, 0));
        int prevOut = 0;
        int nextTensor = 1;

        Map<String, Tensor> params = new LinkedHashMap<>();
        model.namedParameters().forEach((k, v) -> params.put(k, v));

        for (var entry : model.children().entrySet()) {
            String name = entry.getKey();
            if (name.isEmpty() || name.contains("."))
                continue; // direct children only

            String type = entry.getValue().getClass().getSimpleName();
            switch (type) {
                case "Linear" -> {
                    Tensor w = params.get(name + ".weight");
                    Tensor b = params.get(name + ".bias");

                    buffers.add(w != null ? w.toFloatArray() : new float[0]);
                    int wIdx = nextTensor++;
                    tensors.add(new TFLiteTensor(name + ".weight",
                            w != null ? w.shape().dims() : new long[] { 1 }, buffers.size() - 1));

                    List<Integer> inputs = new ArrayList<>(List.of(prevOut, wIdx));
                    if (b != null) {
                        buffers.add(b.toFloatArray());
                        int bIdx = nextTensor++;
                        tensors.add(new TFLiteTensor(name + ".bias",
                                new long[] { b.numel() }, buffers.size() - 1));
                        inputs.add(bIdx);
                    }

                    long outCols = w != null ? w.shape().dim(0) : 1;
                    int outIdx = nextTensor++;
                    tensors.add(new TFLiteTensor(name + "_out",
                            new long[] { inputShape[0], outCols }, 0));
                    ops.add(new TFLiteOp(OP_FULLY_CONNECTED, inputs, List.of(outIdx)));
                    prevOut = outIdx;
                }
                case "ReLU" -> prevOut = addUnary(tensors, ops, OP_RELU, name, prevOut, nextTensor++);
                case "GELU" -> prevOut = addUnary(tensors, ops, OP_GELU, name, prevOut, nextTensor++);
                case "SiLU" -> prevOut = addUnary(tensors, ops, OP_HARD_SWISH, name, prevOut, nextTensor++);
                case "LayerNorm" -> prevOut = addUnary(tensors, ops, OP_LAYER_NORM, name, prevOut, nextTensor++);
                case "Softmax" -> prevOut = addUnary(tensors, ops, OP_SOFTMAX, name, prevOut, nextTensor++);
                case "Dropout", "Sequential" -> {
                    /* skip */ }
                default -> prevOut = addUnary(tensors, ops, OP_CUSTOM, name, prevOut, nextTensor++);
            }
        }

        return encode(buffers, tensors, ops, prevOut);
    }

    private int addUnary(List<TFLiteTensor> tensors, List<TFLiteOp> ops,
            int opCode, String name, int inIdx, int outIdx) {
        tensors.add(new TFLiteTensor(name + "_out", new long[] { -1 }, 0));
        ops.add(new TFLiteOp(opCode, List.of(inIdx), List.of(outIdx)));
        return outIdx;
    }

    // ── FlatBuffer encoding ───────────────────────────────────────────────────

    private byte[] encode(List<float[]> buffers, List<TFLiteTensor> tensors,
            List<TFLiteOp> ops, int outputTensorIdx) {
        FlatBuilder fb = new FlatBuilder();

        int[] bufOffsets = new int[buffers.size()];
        for (int i = buffers.size() - 1; i >= 0; i--)
            bufOffsets[i] = fb.writeBuffer(buffers.get(i));

        int[] tensorOffsets = new int[tensors.size()];
        for (int i = tensors.size() - 1; i >= 0; i--)
            tensorOffsets[i] = fb.writeTensor(tensors.get(i));

        int[] opOffsets = new int[ops.size()];
        for (int i = ops.size() - 1; i >= 0; i--)
            opOffsets[i] = fb.writeOp(ops.get(i));

        int subgraph = fb.writeSubgraph(tensorOffsets, opOffsets,
                new int[] { 0 }, new int[] { outputTensorIdx });
        int model = fb.writeModel(subgraph, bufOffsets);
        return fb.finish(model);
    }

    // ── Descriptors ───────────────────────────────────────────────────────────

    private record TFLiteTensor(String name, long[] shape, int bufferIdx) {
    }

    private record TFLiteOp(int builtinCode, List<Integer> inputs, List<Integer> outputs) {
    }

    // ── Minimal FlatBuffer builder ────────────────────────────────────────────

    private static final class FlatBuilder {

        private final ByteArrayOutputStream buf = new ByteArrayOutputStream();

        int writeBuffer(float[] data) {
            byte[] bytes = new byte[data.length * 4];
            ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            for (float f : data)
                bb.putFloat(f);
            int pos = buf.size();
            writeInt(bytes.length);
            write(bytes);
            align(4);
            return pos;
        }

        int writeTensor(TFLiteTensor t) {
            int pos = buf.size();
            writeString(t.name());
            writeIntVec(t.shape());
            writeInt(TENSOR_FLOAT32);
            writeInt(t.bufferIdx());
            align(4);
            return pos;
        }

        int writeOp(TFLiteOp op) {
            int pos = buf.size();
            writeInt(op.builtinCode());
            writeIntVec(op.inputs().stream().mapToLong(Integer::longValue).toArray());
            writeIntVec(op.outputs().stream().mapToLong(Integer::longValue).toArray());
            align(4);
            return pos;
        }

        int writeSubgraph(int[] tensorOffsets, int[] opOffsets, int[] inputs, int[] outputs) {
            int pos = buf.size();
            writeOffsetVec(tensorOffsets);
            writeOffsetVec(opOffsets);
            writeIntVec(toLong(inputs));
            writeIntVec(toLong(outputs));
            return pos;
        }

        int writeModel(int subgraphOffset, int[] bufferOffsets) {
            int pos = buf.size();
            writeInt(3); // schema version
            writeOffsetVec(new int[] { subgraphOffset });
            writeOffsetVec(bufferOffsets);
            return pos;
        }

        byte[] finish(int rootOffset) {
            byte[] body = buf.toByteArray();
            ByteBuffer out = ByteBuffer.allocate(8 + body.length).order(ByteOrder.LITTLE_ENDIAN);
            out.putInt(body.length + 4);
            out.put(new byte[] { 'T', 'F', 'L', '3' });
            out.put(body);
            return out.array();
        }

        private void writeString(String s) {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            writeInt(b.length);
            write(b);
            align(4);
        }

        private void writeIntVec(long[] values) {
            writeInt(values.length);
            for (long v : values)
                writeInt((int) v);
        }

        private void writeOffsetVec(int[] offsets) {
            writeInt(offsets.length);
            for (int o : offsets)
                writeInt(o);
        }

        private void writeInt(long v) {
            ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            b.putInt((int) v);
            write(b.array());
        }

        private void write(byte[] data) {
            try {
                buf.write(data);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        private void align(int n) {
            int rem = buf.size() % n;
            if (rem != 0)
                write(new byte[n - rem]);
        }

        private static long[] toLong(int[] arr) {
            long[] out = new long[arr.length];
            for (int i = 0; i < arr.length; i++)
                out[i] = arr[i];
            return out;
        }
    }
}
