package tech.kayys.gollek.ml.export.onnx;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.safetensors.SafetensorReader;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Exports a Gollek {@link NNModule} or {@code .safetensors} file to a valid
 * ONNX protobuf (IR version 8, opset 17).
 *
 * <h3>Graph export</h3>
 * When given an {@link NNModule}, the exporter walks the module tree via
 * {@link NNModule#modules()} and maps each layer to its canonical ONNX op:
 * <ul>
 *   <li>{@code Linear}     → {@code Gemm} (weight + optional bias)</li>
 *   <li>{@code LayerNorm}  → {@code LayerNormalization}</li>
 *   <li>{@code ReLU}       → {@code Relu}</li>
 *   <li>{@code GELU}       → {@code Gelu}</li>
 *   <li>{@code SiLU}       → {@code Silu}</li>
 *   <li>{@code Dropout}    → {@code Dropout} (inference: identity)</li>
 *   <li>{@code Embedding}  → {@code Gather}</li>
 *   <li>{@code Sequential} → inlined (no node, just wires children)</li>
 * </ul>
 * Unknown layers fall back to an {@code Identity} node so the graph stays valid.
 *
 * <h3>Weights-only export (SafeTensor)</h3>
 * When called via {@link #fromSafetensor}, all tensors are written as
 * {@code GraphProto.initializer} entries with no operator nodes — correct for
 * a parameter store that will be wired by a downstream graph builder.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * // Full graph from NNModule
 * OnnxExporter.fromModel(model, new long[]{1, 784}).export(Path.of("model.onnx"));
 *
 * // Weights-only from .safetensors
 * OnnxExporter.fromSafetensor(Path.of("model.safetensors")).export(Path.of("model.onnx"));
 * }</pre>
 */
public final class OnnxExporter {

    // ── ONNX protobuf field numbers ───────────────────────────────────────────
    private static final int MODEL_IR_VERSION   = 1;
    private static final int MODEL_PRODUCER     = 2;
    private static final int MODEL_GRAPH        = 7;
    private static final int MODEL_OPSET_IMPORT = 8;
    private static final int OPSET_VERSION      = 2;
    private static final int GRAPH_NAME         = 1;
    private static final int GRAPH_NODE         = 1;
    private static final int GRAPH_INPUT        = 11;
    private static final int GRAPH_OUTPUT       = 12;
    private static final int GRAPH_INITIALIZER  = 6;
    // NodeProto
    private static final int NODE_INPUT         = 1;
    private static final int NODE_OUTPUT        = 2;
    private static final int NODE_NAME          = 3;
    private static final int NODE_OP_TYPE       = 4;
    private static final int NODE_ATTRIBUTE     = 5;
    // AttributeProto
    private static final int ATTR_NAME          = 1;
    private static final int ATTR_F             = 4;  // float
    private static final int ATTR_I             = 3;  // int64
    private static final int ATTR_TYPE          = 20;
    private static final int ATTR_TYPE_FLOAT    = 1;
    private static final int ATTR_TYPE_INT      = 2;
    // ValueInfoProto
    private static final int VALUEINFO_NAME     = 1;
    private static final int VALUEINFO_TYPE     = 2;
    // TypeProto
    private static final int TYPE_TENSOR        = 1;
    // TypeProto.Tensor
    private static final int TENSOR_TYPE_ELEM   = 1;
    private static final int TENSOR_TYPE_SHAPE  = 2;
    // TensorShapeProto
    private static final int SHAPE_DIM          = 1;
    private static final int DIM_VALUE          = 1;
    // TensorProto
    private static final int TENSOR_DIMS        = 1;
    private static final int TENSOR_DATA_TYPE   = 2;
    private static final int TENSOR_NAME        = 8;
    private static final int TENSOR_RAW_DATA    = 9;
    private static final int ONNX_FLOAT         = 1;

    // ── State ─────────────────────────────────────────────────────────────────

    private final Map<String, GradTensor> weights;   // name → tensor (initializers)
    private final List<OnnxNode>          nodes;     // ordered graph nodes
    private final long[]                  inputShape; // null = weights-only export

    private OnnxExporter(Map<String, GradTensor> weights, List<OnnxNode> nodes, long[] inputShape) {
        this.weights    = weights;
        this.nodes      = nodes;
        this.inputShape = inputShape;
    }

    // ── Factories ─────────────────────────────────────────────────────────────

    /**
     * Build a full execution graph from an {@link NNModule}.
     *
     * @param model      trained model
     * @param inputShape shape of one input sample, e.g. {@code {1, 784}}
     */
    public static OnnxExporter fromModel(NNModule model, long[] inputShape) {
        Map<String, GradTensor> weights = new LinkedHashMap<>();
        List<OnnxNode> nodes = new ArrayList<>();

        // Collect all named parameters as initializers
        model.namedParameters().forEach((name, param) ->
            weights.put(name, param.data()));

        // Walk module tree to build nodes
        GraphBuilder builder = new GraphBuilder("input", weights);
        builder.buildFromModule("", model, model.modules());
        nodes.addAll(builder.nodes);

        return new OnnxExporter(weights, nodes, inputShape);
    }

    /** Weights-only export from a {@code .safetensors} file. */
    public static OnnxExporter fromSafetensor(Path path, Map<String, long[]> shapeHints) throws IOException {
        Map<String, float[]> raw = SafetensorReader.read(path);
        Map<String, GradTensor> dict = new LinkedHashMap<>();
        for (var e : raw.entrySet()) {
            float[] data = e.getValue();
            long[] shape = shapeHints.getOrDefault(e.getKey(), new long[]{data.length});
            dict.put(e.getKey(), GradTensor.of(data, shape));
        }
        return new OnnxExporter(dict, List.of(), null);
    }

    public static OnnxExporter fromSafetensor(Path path) throws IOException {
        return fromSafetensor(path, Map.of());
    }

    // ── Export ────────────────────────────────────────────────────────────────

    public void export(Path outputPath) throws IOException {
        Files.write(outputPath, buildModelProto());
    }

    // ── Protobuf serialization ────────────────────────────────────────────────

    private byte[] buildModelProto() throws IOException {
        var out = new ByteArrayOutputStream();
        writeVarint(out, tag(MODEL_IR_VERSION, 0));
        writeVarint(out, 8L);
        writeString(out, MODEL_PRODUCER, "gollek");

        var opset = new ByteArrayOutputStream();
        writeVarint(opset, tag(OPSET_VERSION, 0));
        writeVarint(opset, 17L);
        writeBytes(out, MODEL_OPSET_IMPORT, opset.toByteArray());

        writeBytes(out, MODEL_GRAPH, buildGraphProto());
        return out.toByteArray();
    }

    private byte[] buildGraphProto() throws IOException {
        var out = new ByteArrayOutputStream();
        writeString(out, GRAPH_NAME, "gollek_model");

        // Initializers (weights)
        for (var e : weights.entrySet()) {
            writeBytes(out, GRAPH_INITIALIZER, buildTensorProto(e.getKey(), e.getValue()));
        }

        if (inputShape != null && !nodes.isEmpty()) {
            // Graph input
            writeBytes(out, GRAPH_INPUT, buildValueInfo("input", inputShape));

            // Nodes
            for (OnnxNode node : nodes) {
                writeBytes(out, GRAPH_NODE, buildNodeProto(node));
            }

            // Graph output — last node's output
            String lastOutput = nodes.get(nodes.size() - 1).outputs.get(0);
            writeBytes(out, GRAPH_OUTPUT, buildValueInfo(lastOutput, new long[]{-1}));
        }

        return out.toByteArray();
    }

    private byte[] buildNodeProto(OnnxNode node) throws IOException {
        var out = new ByteArrayOutputStream();
        for (String input  : node.inputs)  writeString(out, NODE_INPUT,   input);
        for (String output : node.outputs) writeString(out, NODE_OUTPUT,  output);
        writeString(out, NODE_NAME,    node.name);
        writeString(out, NODE_OP_TYPE, node.opType);
        for (var attr : node.attributes.entrySet()) {
            writeBytes(out, NODE_ATTRIBUTE, buildAttribute(attr.getKey(), attr.getValue()));
        }
        return out.toByteArray();
    }

    private byte[] buildAttribute(String name, Object value) throws IOException {
        var out = new ByteArrayOutputStream();
        writeString(out, ATTR_NAME, name);
        if (value instanceof Float f) {
            writeVarint(out, tag(ATTR_TYPE, 0));
            writeVarint(out, ATTR_TYPE_FLOAT);
            // float field: wire type 5 (32-bit)
            out.write(tag(ATTR_F, 5));
            ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            buf.putFloat(f);
            out.write(buf.array());
        } else if (value instanceof Long l) {
            writeVarint(out, tag(ATTR_TYPE, 0));
            writeVarint(out, ATTR_TYPE_INT);
            writeVarint(out, tag(ATTR_I, 0));
            writeVarint(out, l);
        }
        return out.toByteArray();
    }

    private byte[] buildValueInfo(String name, long[] shape) throws IOException {
        var out = new ByteArrayOutputStream();
        writeString(out, VALUEINFO_NAME, name);

        // TypeProto.Tensor
        var tensor = new ByteArrayOutputStream();
        writeVarint(tensor, tag(TENSOR_TYPE_ELEM, 0));
        writeVarint(tensor, ONNX_FLOAT);

        // TensorShapeProto
        var shapeProto = new ByteArrayOutputStream();
        for (long dim : shape) {
            var dimProto = new ByteArrayOutputStream();
            writeVarint(dimProto, tag(DIM_VALUE, 0));
            writeVarint(dimProto, dim < 0 ? 0 : dim); // 0 = dynamic
            writeBytes(shapeProto, SHAPE_DIM, dimProto.toByteArray());
        }
        writeBytes(tensor, TENSOR_TYPE_SHAPE, shapeProto.toByteArray());

        // TypeProto
        var typeProto = new ByteArrayOutputStream();
        writeBytes(typeProto, TYPE_TENSOR, tensor.toByteArray());
        writeBytes(out, VALUEINFO_TYPE, typeProto.toByteArray());

        return out.toByteArray();
    }

    private byte[] buildTensorProto(String name, GradTensor tensor) throws IOException {
        var out = new ByteArrayOutputStream();
        for (long dim : tensor.shape()) {
            writeVarint(out, tag(TENSOR_DIMS, 0));
            writeVarint(out, dim);
        }
        writeVarint(out, tag(TENSOR_DATA_TYPE, 0));
        writeVarint(out, ONNX_FLOAT);
        writeString(out, TENSOR_NAME, name);

        float[] data = tensor.data();
        ByteBuffer buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : data) buf.putFloat(v);
        writeBytes(out, TENSOR_RAW_DATA, buf.array());
        return out.toByteArray();
    }

    // ── Protobuf wire helpers ─────────────────────────────────────────────────

    private static int tag(int field, int wireType) { return (field << 3) | wireType; }

    private static void writeVarint(OutputStream out, long v) throws IOException {
        while ((v & ~0x7FL) != 0) { out.write((int) ((v & 0x7F) | 0x80)); v >>>= 7; }
        out.write((int) v);
    }

    private static void writeString(OutputStream out, int field, String value) throws IOException {
        writeBytes(out, field, value.getBytes(StandardCharsets.UTF_8));
    }

    private static void writeBytes(OutputStream out, int field, byte[] value) throws IOException {
        writeVarint(out, tag(field, 2));
        writeVarint(out, value.length);
        out.write(value);
    }

    // ── Graph builder ─────────────────────────────────────────────────────────

    /** Walks the NNModule tree and emits ONNX nodes. */
    private static final class GraphBuilder {

        final List<OnnxNode> nodes = new ArrayList<>();
        private final Map<String, GradTensor> weights;
        private String currentOutput; // wire name flowing through the graph
        private int nodeIdx = 0;

        GraphBuilder(String inputName, Map<String, GradTensor> weights) {
            this.currentOutput = inputName;
            this.weights = weights;
        }

        void buildFromModule(String path, NNModule root, Map<String, NNModule> allModules) {
            // Process only direct children of root in registration order
            for (var entry : allModules.entrySet()) {
                String name = entry.getKey();
                NNModule module = entry.getValue();
                if (name.isEmpty() || name.equals(path)) continue;
                // Only direct children (no dots beyond the current path prefix)
                String relative = path.isEmpty() ? name : (name.startsWith(path + ".") ? name.substring(path.length() + 1) : null);
                if (relative == null || relative.contains(".")) continue;

                emitNode(name, module);
            }
        }

        private void emitNode(String name, NNModule module) {
            String simpleName = module.getClass().getSimpleName();
            String outputName = name + "_out_" + nodeIdx++;

            OnnxNode node = switch (simpleName) {
                case "Linear"     -> linearNode(name, outputName);
                case "LayerNorm"  -> layerNormNode(name, outputName);
                case "ReLU"       -> unaryNode(name, outputName, "Relu");
                case "GELU"       -> unaryNode(name, outputName, "Gelu");
                case "SiLU"       -> unaryNode(name, outputName, "Silu");
                case "Tanh"       -> unaryNode(name, outputName, "Tanh");
                case "Sigmoid"    -> unaryNode(name, outputName, "Sigmoid");
                case "Dropout"    -> unaryNode(name, outputName, "Identity"); // inference = identity
                case "Flatten"    -> unaryNode(name, outputName, "Flatten");
                case "Embedding"  -> embeddingNode(name, outputName);
                case "Sequential" -> null; // transparent — children wired directly
                default           -> unaryNode(name, outputName, "Identity");
            };

            if (node != null) {
                nodes.add(node);
                currentOutput = outputName;
            }
        }

        /** Gemm: Y = alpha * A * B^T + beta * C */
        private OnnxNode linearNode(String name, String output) {
            List<String> inputs = new ArrayList<>();
            inputs.add(currentOutput);
            inputs.add(name + ".weight");
            if (weights.containsKey(name + ".bias")) inputs.add(name + ".bias");

            Map<String, Object> attrs = new LinkedHashMap<>();
            attrs.put("alpha", 1.0f);
            attrs.put("beta",  1.0f);
            attrs.put("transB", 1L); // weight is [out, in] → transpose B

            return new OnnxNode("Gemm", name + "/Gemm", inputs, List.of(output), attrs);
        }

        private OnnxNode layerNormNode(String name, String output) {
            List<String> inputs = List.of(currentOutput, name + ".weight", name + ".bias");
            Map<String, Object> attrs = new LinkedHashMap<>();
            attrs.put("epsilon", 1e-5f);
            return new OnnxNode("LayerNormalization", name + "/LN", inputs, List.of(output), attrs);
        }

        private OnnxNode embeddingNode(String name, String output) {
            // Gather(weight, indices) — indices come from input
            List<String> inputs = List.of(name + ".weight", currentOutput);
            return new OnnxNode("Gather", name + "/Gather", inputs, List.of(output), Map.of("axis", 0L));
        }

        private OnnxNode unaryNode(String name, String output, String opType) {
            return new OnnxNode(opType, name + "/" + opType,
                List.of(currentOutput), List.of(output), Map.of());
        }
    }

    /** Lightweight ONNX node descriptor. */
    private record OnnxNode(
        String opType,
        String name,
        List<String> inputs,
        List<String> outputs,
        Map<String, Object> attributes
    ) {}
}
