package tech.kayys.gollek.onnx.runner;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Stream;

public final class OnnxModelDiagnostics {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final long MAX_GRAPH_IO_INSPECT_BYTES = 64L * 1024L * 1024L;

    private OnnxModelDiagnostics() {
    }

    public static Report inspect(Path modelPath) {
        Path normalizedPath = modelPath == null ? null : modelPath.toAbsolutePath().normalize();
        Path modelDir = resolveModelDir(normalizedPath);
        String modelType = null;
        List<String> architectures = List.of();
        List<String> warnings = new ArrayList<>();

        JsonNode config = readJson(modelDir == null ? null : modelDir.resolve("config.json"));
        JsonNode genaiConfig = readJson(modelDir == null ? null : modelDir.resolve("genai_config.json"));
        if (config != null) {
            modelType = text(config.get("model_type"));
            architectures = stringArray(config.get("architectures"));
        }
        if (modelType == null) {
            JsonNode genaiModel = genaiConfig == null ? null : genaiConfig.get("model");
            modelType = text(genaiModel == null ? null : genaiModel.get("type"));
        }
        GenAiInfo genAiInfo = genAiInfo(genaiConfig);

        List<GraphInfo> graphs = findOnnxGraphs(modelDir);
        List<String> sidecars = findSidecars(modelDir);
        String pipelineType = pipelineType(modelType, architectures, graphs, sidecars);
        List<String> capabilities = capabilities(pipelineType, graphs, sidecars);

        if (isPaddleOcrVl(modelType, architectures, graphs)) {
            warnings.add("PaddleOCR-VL is a multi-session OCR/VL pipeline; it needs image preprocessing, "
                    + "vision encoder, embedding, decoder orchestration, and tokenizer postprocessing.");
        } else if (isCustomMultiSession(graphs)) {
            warnings.add("Multiple ONNX graph roles were detected; this repository likely needs custom pipeline orchestration.");
        }
        if (graphs.stream().anyMatch(GraphInfo::externalData)) {
            warnings.add("One or more graphs use external .data tensor files; keep sidecar files beside the .onnx files.");
        }
        if (graphs.isEmpty()) {
            warnings.add("No .onnx graph files were found under the resolved model directory.");
        }

        return new Report(
                normalizedPath,
                modelDir,
                pipelineType,
                modelType,
                architectures,
                genAiInfo,
                graphs,
                sidecars,
                capabilities,
                List.copyOf(warnings),
                recommendations(pipelineType, graphs, sidecars));
    }

    public static boolean requiresCustomPipelineOrchestration(Path modelPath) {
        return inspect(modelPath).requiresCustomPipelineOrchestration();
    }

    private static Path resolveModelDir(Path modelPath) {
        if (modelPath == null) {
            return null;
        }
        if (Files.isRegularFile(modelPath)) {
            Path parent = modelPath.getParent();
            if (parent != null && "onnx".equalsIgnoreCase(fileName(parent))) {
                Path repoRoot = parent.getParent();
                if (repoRoot != null && Files.isRegularFile(repoRoot.resolve("config.json"))) {
                    return repoRoot.toAbsolutePath().normalize();
                }
            }
            return parent == null ? modelPath : parent.toAbsolutePath().normalize();
        }
        return modelPath.toAbsolutePath().normalize();
    }

    private static JsonNode readJson(Path path) {
        if (path == null || !Files.isRegularFile(path)) {
            return null;
        }
        try {
            return MAPPER.readTree(path.toFile());
        } catch (Exception ignored) {
            return null;
        }
    }

    private static List<GraphInfo> findOnnxGraphs(Path modelDir) {
        if (modelDir == null || !Files.exists(modelDir)) {
            return List.of();
        }
        try (Stream<Path> walk = Files.walk(modelDir, 8)) {
            return walk.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".onnx"))
                    .sorted(Comparator.comparing(path -> modelDir.relativize(path).toString()))
                    .map(path -> graphInfo(modelDir, path))
                    .toList();
        } catch (Exception ignored) {
            return List.of();
        }
    }

    private static GraphInfo graphInfo(Path modelDir, Path graph) {
        String relative = modelDir.relativize(graph).toString().replace('\\', '/');
        long size = 0L;
        try {
            size = Files.size(graph);
        } catch (Exception ignored) {
            // Size is diagnostic only.
        }
        long graphSize = size;
        long externalDataSize = externalDataSize(graph);
        GraphIoInfo graphIo = inspectGraphIo(graph, graphSize);
        return new GraphInfo(
                relative,
                graphRole(relative),
                graphSize + externalDataSize,
                externalDataSize > 0,
                quantization(relative),
                graphIo.inputs(),
                graphIo.outputs(),
                graphIo.status());
    }

    private static long externalDataSize(Path graph) {
        String name = graph.getFileName().toString();
        Path parent = graph.getParent();
        if (parent == null) {
            return 0L;
        }
        long bytes = 0L;
        Set<Path> candidates = new LinkedHashSet<>(List.of(
                parent.resolve(name + ".data"),
                parent.resolve(name.replaceFirst("\\.onnx$", ".onnx.data")),
                parent.resolve(name.replaceFirst("\\.onnx$", ".data"))));
        for (Path candidate : candidates) {
            if (!Files.isRegularFile(candidate)) {
                continue;
            }
            try {
                bytes += Files.size(candidate);
            } catch (Exception ignored) {
                // Keep the diagnostic best-effort.
            }
        }
        return bytes;
    }

    private static String graphRole(String relativePath) {
        String value = relativePath.toLowerCase(Locale.ROOT);
        if (value.contains("vision_encoder")) {
            return "vision_encoder";
        }
        if (value.contains("text_encoder")) {
            return "text_encoder";
        }
        if (value.contains("embedding") || value.contains("embed")) {
            return "embedding";
        }
        if (value.contains("decoder")) {
            return "decoder";
        }
        if (value.contains("prefill")) {
            return "prefill";
        }
        if (value.contains("decode")) {
            return "decode";
        }
        if (value.contains("local") && value.contains("frame")) {
            return "local_frame_sampling";
        }
        if (value.contains("unet")) {
            return "unet";
        }
        if (value.contains("vae")) {
            return "vae";
        }
        if (value.contains("tokenizer") || value.contains("codec")) {
            return "codec";
        }
        return "graph";
    }

    private static String quantization(String relativePath) {
        String value = relativePath.toLowerCase(Locale.ROOT);
        if (value.contains("quint8")) {
            return "uint8";
        }
        if (value.contains("q8") || value.contains("int8")) {
            return "q8";
        }
        if (value.contains("q4") || value.contains("int4")) {
            return "q4";
        }
        if (value.contains("fp16") || value.contains("float16")) {
            return "fp16";
        }
        return "full";
    }

    private static GraphIoInfo inspectGraphIo(Path graph, long graphSize) {
        if (graph == null || !Files.isRegularFile(graph)) {
            return new GraphIoInfo(List.of(), List.of(), "unavailable: graph file not found");
        }
        if (graphSize <= 0L) {
            return new GraphIoInfo(List.of(), List.of(), "unavailable: graph file is empty");
        }
        if (graphSize > MAX_GRAPH_IO_INSPECT_BYTES) {
            return new GraphIoInfo(
                    List.of(),
                    List.of(),
                    "skipped: graph protobuf is " + graphSize
                            + " bytes; use a runtime session probe for large embedded-weight graphs");
        }
        try {
            GraphIoInfo parsed = parseModelGraphIo(Files.readAllBytes(graph));
            if (parsed.inputs().isEmpty() && parsed.outputs().isEmpty()) {
                return new GraphIoInfo(
                        parsed.inputs(),
                        parsed.outputs(),
                        "unavailable: graph inputs/outputs were not found in protobuf metadata");
            }
            return new GraphIoInfo(parsed.inputs(), parsed.outputs(), "parsed");
        } catch (Exception e) {
            return new GraphIoInfo(
                    List.of(),
                    List.of(),
                    "unavailable: " + conciseMessage(e));
        }
    }

    private static GraphIoInfo parseModelGraphIo(byte[] bytes) throws IOException {
        ProtoReader model = new ProtoReader(bytes, 0, bytes.length);
        while (model.hasRemaining()) {
            int tag = model.readTag();
            if (tag == 0) {
                break;
            }
            int field = tag >>> 3;
            int wire = tag & 7;
            if (field == 7 && wire == ProtoReader.WIRE_LENGTH_DELIMITED) {
                return parseGraph(model.readMessage());
            }
            model.skip(wire);
        }
        return new GraphIoInfo(List.of(), List.of(), "unavailable: graph protobuf not found");
    }

    private static GraphIoInfo parseGraph(ProtoReader graph) throws IOException {
        List<TensorIoInfo> inputs = new ArrayList<>();
        List<TensorIoInfo> outputs = new ArrayList<>();
        while (graph.hasRemaining()) {
            int tag = graph.readTag();
            if (tag == 0) {
                break;
            }
            int field = tag >>> 3;
            int wire = tag & 7;
            if (wire == ProtoReader.WIRE_LENGTH_DELIMITED && field == 11) {
                inputs.add(parseValueInfo(graph.readMessage()));
            } else if (wire == ProtoReader.WIRE_LENGTH_DELIMITED && field == 12) {
                outputs.add(parseValueInfo(graph.readMessage()));
            } else {
                graph.skip(wire);
            }
        }
        return new GraphIoInfo(List.copyOf(inputs), List.copyOf(outputs), "parsed");
    }

    private static TensorIoInfo parseValueInfo(ProtoReader valueInfo) throws IOException {
        String name = null;
        TensorTypeInfo type = TensorTypeInfo.UNKNOWN;
        while (valueInfo.hasRemaining()) {
            int tag = valueInfo.readTag();
            if (tag == 0) {
                break;
            }
            int field = tag >>> 3;
            int wire = tag & 7;
            if (field == 1 && wire == ProtoReader.WIRE_LENGTH_DELIMITED) {
                name = valueInfo.readString();
            } else if (field == 2 && wire == ProtoReader.WIRE_LENGTH_DELIMITED) {
                type = parseType(valueInfo.readMessage());
            } else {
                valueInfo.skip(wire);
            }
        }
        return new TensorIoInfo(
                name == null || name.isBlank() ? "(unnamed)" : name,
                type.elementType(),
                type.shape());
    }

    private static TensorTypeInfo parseType(ProtoReader type) throws IOException {
        TensorTypeInfo tensorType = TensorTypeInfo.UNKNOWN;
        while (type.hasRemaining()) {
            int tag = type.readTag();
            if (tag == 0) {
                break;
            }
            int field = tag >>> 3;
            int wire = tag & 7;
            if (field == 1 && wire == ProtoReader.WIRE_LENGTH_DELIMITED) {
                tensorType = parseTensorType(type.readMessage());
            } else {
                type.skip(wire);
            }
        }
        return tensorType;
    }

    private static TensorTypeInfo parseTensorType(ProtoReader tensorType) throws IOException {
        String elementType = "unknown";
        List<String> shape = List.of();
        while (tensorType.hasRemaining()) {
            int tag = tensorType.readTag();
            if (tag == 0) {
                break;
            }
            int field = tag >>> 3;
            int wire = tag & 7;
            if (field == 1 && wire == ProtoReader.WIRE_VARINT) {
                elementType = onnxElementType(tensorType.readVarint());
            } else if (field == 2 && wire == ProtoReader.WIRE_LENGTH_DELIMITED) {
                shape = parseTensorShape(tensorType.readMessage());
            } else {
                tensorType.skip(wire);
            }
        }
        return new TensorTypeInfo(elementType, shape);
    }

    private static List<String> parseTensorShape(ProtoReader shape) throws IOException {
        List<String> dims = new ArrayList<>();
        while (shape.hasRemaining()) {
            int tag = shape.readTag();
            if (tag == 0) {
                break;
            }
            int field = tag >>> 3;
            int wire = tag & 7;
            if (field == 1 && wire == ProtoReader.WIRE_LENGTH_DELIMITED) {
                dims.add(parseDimension(shape.readMessage()));
            } else {
                shape.skip(wire);
            }
        }
        return List.copyOf(dims);
    }

    private static String parseDimension(ProtoReader dimension) throws IOException {
        String value = null;
        while (dimension.hasRemaining()) {
            int tag = dimension.readTag();
            if (tag == 0) {
                break;
            }
            int field = tag >>> 3;
            int wire = tag & 7;
            if (field == 1 && wire == ProtoReader.WIRE_VARINT) {
                value = Long.toString(dimension.readVarint());
            } else if (field == 2 && wire == ProtoReader.WIRE_LENGTH_DELIMITED) {
                value = dimension.readString();
            } else {
                dimension.skip(wire);
            }
        }
        return value == null || value.isBlank() ? "?" : value;
    }

    private static String onnxElementType(long value) {
        return switch ((int) value) {
            case 0 -> "undefined";
            case 1 -> "float32";
            case 2 -> "uint8";
            case 3 -> "int8";
            case 4 -> "uint16";
            case 5 -> "int16";
            case 6 -> "int32";
            case 7 -> "int64";
            case 8 -> "string";
            case 9 -> "bool";
            case 10 -> "float16";
            case 11 -> "float64";
            case 12 -> "uint32";
            case 13 -> "uint64";
            case 14 -> "complex64";
            case 15 -> "complex128";
            case 16 -> "bfloat16";
            case 17 -> "float8e4m3fn";
            case 18 -> "float8e4m3fnuz";
            case 19 -> "float8e5m2";
            case 20 -> "float8e5m2fnuz";
            case 21 -> "uint4";
            case 22 -> "int4";
            case 23 -> "float4e2m1";
            default -> "type" + value;
        };
    }

    private static String conciseMessage(Exception e) {
        String message = e.getMessage();
        if (message == null || message.isBlank()) {
            return e.getClass().getSimpleName();
        }
        return message.replace('\n', ' ').replace('\r', ' ');
    }

    private static List<String> findSidecars(Path modelDir) {
        if (modelDir == null || !Files.exists(modelDir)) {
            return List.of();
        }
        try (Stream<Path> walk = Files.walk(modelDir, 4)) {
            return walk.filter(Files::isRegularFile)
                    .map(path -> modelDir.relativize(path).toString().replace('\\', '/'))
                    .filter(OnnxModelDiagnostics::isSidecar)
                    .sorted()
                    .toList();
        } catch (Exception ignored) {
            return List.of();
        }
    }

    private static boolean isSidecar(String relativePath) {
        String name = relativePath.toLowerCase(Locale.ROOT);
        return name.equals("config.json")
                || name.equals("genai_config.json")
                || name.equals("tokenizer.json")
                || name.equals("tokenizer_config.json")
                || name.equals("processor_config.json")
                || name.equals("preprocessor_config.json")
                || name.equals("chat_template.jinja")
                || name.equals("readme.md")
                || name.endsWith("_onnx_meta.json")
                || name.startsWith("processing_")
                || name.startsWith("image_processing_")
                || name.startsWith("configuration_");
    }

    private static String pipelineType(
            String modelType,
            List<String> architectures,
            List<GraphInfo> graphs,
            List<String> sidecars) {
        if (isPaddleOcrVl(modelType, architectures, graphs)) {
            return "PaddleOCR-VL OCR/VL pipeline";
        }
        if (sidecars.contains("tts_browser_onnx_meta.json")) {
            return "MOSS TTS pipeline";
        }
        if (sidecars.contains("codec_browser_onnx_meta.json")) {
            return "MOSS audio tokenizer/codec";
        }
        Set<String> roles = graphRoles(graphs);
        if (roles.contains("unet") && roles.contains("vae")) {
            return "Stable Diffusion pipeline";
        }
        if (sidecars.contains("genai_config.json") && graphs.size() == 1) {
            return "ORT GenAI single graph";
        }
        if (graphs.size() == 1) {
            return "Single ONNX graph";
        }
        if (graphs.size() > 1) {
            return "Multi-session ONNX pipeline";
        }
        return "ONNX repository";
    }

    private static List<String> capabilities(String pipelineType, List<GraphInfo> graphs, List<String> sidecars) {
        List<String> values = new ArrayList<>();
        if (pipelineType.startsWith("PaddleOCR-VL")) {
            values.add("ocr");
            values.add("vision-language");
        } else if (pipelineType.startsWith("MOSS TTS")) {
            values.add("text-to-speech");
            values.add("audio");
        } else if (pipelineType.startsWith("MOSS audio")) {
            values.add("audio-codec");
        } else if (pipelineType.startsWith("Stable Diffusion")) {
            values.add("text-to-image");
        } else if (pipelineType.startsWith("ORT GenAI")) {
            values.add("single-graph");
            values.add("ort-genai");
        } else if (graphs.size() == 1) {
            values.add("single-graph");
        }
        if (sidecars.contains("tokenizer.json")) {
            values.add("tokenizer");
        }
        if (sidecars.contains("processor_config.json")) {
            values.add("processor");
        }
        return List.copyOf(values);
    }

    private static List<String> recommendations(String pipelineType, List<GraphInfo> graphs, List<String> sidecars) {
        List<String> values = new ArrayList<>();
        if (pipelineType.startsWith("PaddleOCR-VL")) {
            values.add("Use --onnx-diagnostics to inspect graph roles before running OCR.");
            values.add("Runtime work needed: image preprocessing -> vision_encoder -> embedding -> decoder -> tokenizer postprocess.");
            values.add("When OCR execution is wired, pass input pages with --image or --images.");
        } else if ("Multi-session ONNX pipeline".equals(pipelineType)) {
            values.add("Add a model-specific pipeline adapter before routing this repository to the generic text runner.");
        } else if (pipelineType.startsWith("ORT GenAI")) {
            values.add("ORT GenAI metadata detected; model-specific prompt wiring and KV-cache handling may be needed.");
        } else if (graphs.size() == 1) {
            values.add("Single graph detected; generic ONNX execution may work if input/output names match the text runner.");
        }
        if (!sidecars.contains("config.json") && !sidecars.contains("genai_config.json")) {
            values.add("config.json was not found; model type detection may be incomplete.");
        }
        return List.copyOf(values);
    }

    private static boolean isPaddleOcrVl(String modelType, List<String> architectures, List<GraphInfo> graphs) {
        if ("paddleocr_vl".equalsIgnoreCase(modelType)) {
            return true;
        }
        for (String architecture : architectures) {
            if (architecture != null && architecture.toLowerCase(Locale.ROOT).contains("paddleocrvl")) {
                return true;
            }
        }
        Set<String> roles = graphRoles(graphs);
        return roles.contains("vision_encoder") && roles.contains("embedding") && roles.contains("decoder");
    }

    private static boolean isCustomMultiSession(List<GraphInfo> graphs) {
        Set<String> roles = graphRoles(graphs);
        return roles.contains("vision_encoder") && roles.contains("decoder")
                || roles.contains("embedding") && roles.contains("decoder")
                || roles.contains("prefill") && roles.contains("decode");
    }

    private static Set<String> graphRoles(List<GraphInfo> graphs) {
        Set<String> roles = new LinkedHashSet<>();
        for (GraphInfo graph : graphs) {
            roles.add(graph.role());
        }
        return roles;
    }

    private static GenAiInfo genAiInfo(JsonNode root) {
        if (root == null || root.isMissingNode() || root.isNull()) {
            return null;
        }
        JsonNode model = root.path("model");
        JsonNode decoder = model.path("decoder");
        JsonNode inputs = decoder.path("inputs");
        JsonNode outputs = decoder.path("outputs");
        return new GenAiInfo(
                text(model.path("type")),
                positiveInt(decoder.path("num_hidden_layers")),
                positiveInt(decoder.path("num_attention_heads")),
                positiveInt(decoder.path("num_key_value_heads")),
                positiveInt(decoder.path("head_size")),
                positiveInt(decoder.path("hidden_size")),
                positiveInt(model.path("context_length")),
                positiveInt(model.path("vocab_size")),
                textOr(inputs.path("input_ids"), "input_ids"),
                textOr(inputs.path("attention_mask"), "attention_mask"),
                textOr(inputs.path("position_ids"), "position_ids"),
                textOr(inputs.path("past_key_names"), "past_key_values.%d.key"),
                textOr(inputs.path("past_value_names"), "past_key_values.%d.value"),
                textOr(outputs.path("logits"), "logits"),
                textOr(outputs.path("present_key_names"), "present.%d.key"),
                textOr(outputs.path("present_value_names"), "present.%d.value"));
    }

    private static int positiveInt(JsonNode node) {
        if (node == null || !node.isNumber()) {
            return 0;
        }
        int value = node.asInt();
        return value > 0 ? value : 0;
    }

    private static String text(JsonNode node) {
        if (node == null || node.isNull() || node.isMissingNode()) {
            return null;
        }
        String value = node.asText(null);
        return value == null || value.isBlank() ? null : value;
    }

    private static String textOr(JsonNode node, String fallback) {
        String value = text(node);
        return value == null ? fallback : value;
    }

    private static List<String> stringArray(JsonNode node) {
        if (node == null || !node.isArray()) {
            return List.of();
        }
        List<String> values = new ArrayList<>();
        for (JsonNode item : node) {
            String value = text(item);
            if (value != null) {
                values.add(value);
            }
        }
        return List.copyOf(values);
    }

    private static String fileName(Path path) {
        return path == null || path.getFileName() == null ? "" : path.getFileName().toString();
    }

    public record GraphInfo(
            String relativePath,
            String role,
            long sizeBytes,
            boolean externalData,
            String quantization,
            List<TensorIoInfo> inputs,
            List<TensorIoInfo> outputs,
            String ioStatus) {
    }

    public record TensorIoInfo(
            String name,
            String elementType,
            List<String> shape) {
    }

    public record GenAiInfo(
            String type,
            int layers,
            int attentionHeads,
            int keyValueHeads,
            int headSize,
            int hiddenSize,
            int contextLength,
            int vocabSize,
            String inputIdsName,
            String attentionMaskName,
            String positionIdsName,
            String pastKeyNameTemplate,
            String pastValueNameTemplate,
            String logitsName,
            String presentKeyNameTemplate,
            String presentValueNameTemplate) {
    }

    private record GraphIoInfo(
            List<TensorIoInfo> inputs,
            List<TensorIoInfo> outputs,
            String status) {
    }

    private record TensorTypeInfo(
            String elementType,
            List<String> shape) {
        static final TensorTypeInfo UNKNOWN = new TensorTypeInfo("unknown", List.of());
    }

    private static final class ProtoReader {
        static final int WIRE_VARINT = 0;
        static final int WIRE_FIXED64 = 1;
        static final int WIRE_LENGTH_DELIMITED = 2;
        static final int WIRE_START_GROUP = 3;
        static final int WIRE_END_GROUP = 4;
        static final int WIRE_FIXED32 = 5;

        private final byte[] data;
        private int position;
        private final int end;

        private ProtoReader(byte[] data, int start, int end) {
            this.data = data;
            this.position = start;
            this.end = end;
        }

        boolean hasRemaining() {
            return position < end;
        }

        int readTag() throws IOException {
            if (!hasRemaining()) {
                return 0;
            }
            long tag = readVarint();
            if (tag > Integer.MAX_VALUE) {
                throw new IOException("protobuf tag is too large");
            }
            return (int) tag;
        }

        long readVarint() throws IOException {
            long value = 0L;
            int shift = 0;
            while (shift < 64) {
                if (position >= end) {
                    throw new IOException("truncated protobuf varint");
                }
                int b = data[position++] & 0xff;
                value |= (long) (b & 0x7f) << shift;
                if ((b & 0x80) == 0) {
                    return value;
                }
                shift += 7;
            }
            throw new IOException("protobuf varint is too long");
        }

        ProtoReader readMessage() throws IOException {
            int length = readLength();
            int start = position;
            skipBytes(length);
            return new ProtoReader(data, start, start + length);
        }

        String readString() throws IOException {
            int length = readLength();
            int start = position;
            skipBytes(length);
            return new String(data, start, length, StandardCharsets.UTF_8);
        }

        void skip(int wireType) throws IOException {
            switch (wireType) {
                case WIRE_VARINT -> readVarint();
                case WIRE_FIXED64 -> skipBytes(8);
                case WIRE_LENGTH_DELIMITED -> skipBytes(readLength());
                case WIRE_START_GROUP -> skipGroup();
                case WIRE_END_GROUP -> {
                    return;
                }
                case WIRE_FIXED32 -> skipBytes(4);
                default -> throw new IOException("unsupported protobuf wire type " + wireType);
            }
        }

        private int readLength() throws IOException {
            long length = readVarint();
            if (length < 0L || length > Integer.MAX_VALUE || length > end - position) {
                throw new IOException("invalid protobuf length " + length);
            }
            return (int) length;
        }

        private void skipBytes(int count) throws IOException {
            if (count < 0 || count > end - position) {
                throw new IOException("truncated protobuf field");
            }
            position += count;
        }

        private void skipGroup() throws IOException {
            while (hasRemaining()) {
                int tag = readTag();
                int wire = tag & 7;
                if (wire == WIRE_END_GROUP) {
                    return;
                }
                skip(wire);
            }
        }
    }

    public record Report(
            Path modelPath,
            Path modelDir,
            String pipelineType,
            String modelType,
            List<String> architectures,
            GenAiInfo genAiInfo,
            List<GraphInfo> graphs,
            List<String> sidecars,
            List<String> capabilities,
            List<String> warnings,
            List<String> recommendations) {

        public boolean isPaddleOcrVl() {
            return pipelineType != null && pipelineType.startsWith("PaddleOCR-VL");
        }

        public boolean requiresCustomPipelineOrchestration() {
            return isPaddleOcrVl();
        }
    }
}
