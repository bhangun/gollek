/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.route;
import tech.kayys.gollek.sdk.route.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ComponentReadiness;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.HeaderInspection;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ProblemDetail;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.TensorInventory;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Header-only contract check for Gemma 4 unified decoder and projector tensors.
 */
final class Gemma4UnifiedSafetensorPreflight {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final long MAX_HEADER_BYTES = 100L * 1024L * 1024L;
    private static final int MAX_REPORTED_PROBLEMS = 12;

    private Gemma4UnifiedSafetensorPreflight() {
    }

    public record Result(
            boolean allowed,
            List<String> messages,
            ProjectorSummary projectors,
            HeaderSummary header,
            TensorInventorySummary inventory,
            ComponentReadinessSummary readiness) {
        public static Result pass() {
            return pass(
                    ProjectorSummary.empty(),
                    HeaderSummary.empty(),
                    TensorInventorySummary.empty(),
                    ComponentReadinessSummary.empty());
        }

        public static Result pass(
                ProjectorSummary projectors,
                HeaderSummary header,
                TensorInventorySummary inventory,
                ComponentReadinessSummary readiness) {
            return new Result(true, List.of(), projectors, header, inventory, readiness);
        }

        public static Result invalid(List<String> messages) {
            return invalid(
                    messages,
                    ProjectorSummary.empty(),
                    HeaderSummary.empty(),
                    TensorInventorySummary.empty(),
                    ComponentReadinessSummary.empty());
        }

        public static Result invalid(
                List<String> messages,
                ProjectorSummary projectors,
                HeaderSummary header,
                TensorInventorySummary inventory,
                ComponentReadinessSummary readiness) {
            return new Result(false, List.copyOf(messages), projectors, header, inventory, readiness);
        }

        Map<String, Object> diagnosticDetails() {
            return Map.of(
                    ProblemDetail.HEADER_INSPECTION, header.diagnosticDetails(),
                    ProblemDetail.TENSOR_INVENTORY, inventory.diagnosticDetails(),
                    ProblemDetail.COMPONENT_READINESS, readiness.diagnosticDetails());
        }
    }

    public static Result validate(Path modelPath, ModelConfig config, String modelLabel) {
        return inspect(modelPath, config, modelLabel).text();
    }

    public static Inspection inspect(Path modelPath, ModelConfig config, String modelLabel) {
        if (modelPath == null || config == null) {
            return Inspection.pass(
                    ProjectorSummary.empty(),
                    HeaderSummary.empty(),
                    TensorInventorySummary.empty(),
                    ComponentReadinessSummary.empty());
        }
        String label = modelLabel == null || modelLabel.isBlank()
                ? modelPath.getFileName().toString()
                : modelLabel;
        try {
            List<Path> files = safetensorFiles(modelPath);
            if (files.isEmpty()) {
                return Inspection.invalid(
                        invalidMessages(label, List.of("no .safetensors file or model.safetensors.index.json was found")),
                        ProjectorSummary.empty(),
                        HeaderSummary.empty(),
                        TensorInventorySummary.empty(),
                        ComponentReadinessSummary.empty());
            }

            HeaderLoad headers = loadTensorHeaders(files);
            Map<String, TensorMeta> tensors = headers.tensors();
            HeaderSummary header = headers.summary();
            TensorInventorySummary inventory = TensorInventorySummary.from(tensors);
            ProjectorSummary projectors = ProjectorSummary.from(tensors, config);
            if (tensors.isEmpty()) {
                ComponentReadinessSummary readiness = ComponentReadinessSummary.from(false, projectors, inventory);
                return Inspection.invalid(
                        invalidMessages(label, List.of("SafeTensors headers did not contain any tensor entries")),
                        projectors,
                        header,
                        inventory,
                        readiness);
            }

            List<String> problems = new ArrayList<>();
            validateTextTowerContract(config, tensors, problems);
            ComponentReadinessSummary readiness =
                    ComponentReadinessSummary.from(problems.isEmpty(), projectors, inventory);
            if (!problems.isEmpty()) {
                return Inspection.invalid(invalidMessages(label, problems), projectors, header, inventory, readiness);
            }
            return Inspection.pass(projectors, header, inventory, readiness);
        } catch (IOException | RuntimeException e) {
            return Inspection.invalid(
                    invalidMessages(label, List.of("could not inspect SafeTensors headers: " + e.getMessage())),
                    ProjectorSummary.empty(),
                    HeaderSummary.empty(),
                    TensorInventorySummary.empty(),
                    ComponentReadinessSummary.empty());
        }
    }

    private static List<String> invalidMessages(String modelLabel, List<String> problems) {
        List<String> messages = new ArrayList<>();
        messages.add("Error: Gemma 4 unified safetensor text preflight failed.");
        messages.add("Checkpoint " + modelLabel
                + " is recognized as gemma4_unified text, but required text decoder SafeTensors headers are missing or incompatible.");
        for (String problem : problems) {
            messages.add("  - " + problem);
        }
        messages.add("This check reads only SafeTensors headers; no 12B weight payload was loaded.");
        return messages;
    }

    private static void validateTextTowerContract(
            ModelConfig config,
            Map<String, TensorMeta> tensors,
            List<String> problems) {
        int hiddenSize = config.hiddenSize();
        int intermediateSize = config.intermediateSize();
        int vocabSize = config.vocabSize();
        int layers = config.numHiddenLayers();
        if (hiddenSize <= 0 || intermediateSize <= 0 || vocabSize <= 0 || layers <= 0) {
            addProblem(problems, "invalid Gemma 4 text config dimensions: hidden=%d, intermediate=%d, vocab=%d, layers=%d"
                    .formatted(hiddenSize, intermediateSize, vocabSize, layers));
            return;
        }

        TensorMeta embed = requireTensor(
                tensors,
                problems,
                textCandidates("embed_tokens.weight"));
        if (embed != null) {
            requireShape(problems, embed, vocabSize, hiddenSize);
        }

        TensorMeta finalNorm = requireTensor(
                tensors,
                problems,
                textCandidates("norm.weight"));
        if (finalNorm != null) {
            requireShape(problems, finalNorm, hiddenSize);
        }

        TensorMeta lmHead = findTensor(
                tensors,
                List.of("lm_head.weight", "model.lm_head.weight", "model.language_model.lm_head.weight"));
        if (lmHead == null && !config.tieWordEmbeddings()) {
            addProblem(problems, "missing tensor: lm_head.weight for untied output embeddings");
        } else if (lmHead != null) {
            requireShape(problems, lmHead, vocabSize, hiddenSize);
        }

        for (int layer = 0; layer < layers; layer++) {
            validateLayer(config, tensors, problems, layer, hiddenSize, intermediateSize);
            if (problems.size() > MAX_REPORTED_PROBLEMS) {
                break;
            }
        }
    }

    private static void validateLayer(
            ModelConfig config,
            Map<String, TensorMeta> tensors,
            List<String> problems,
            int layer,
            int hiddenSize,
            int intermediateSize) {
        String prefix = "layers.%d.".formatted(layer);
        int headDim = config.resolvedHeadDimForLayer(layer);
        int queryRows = config.numAttentionHeads() > 0 && headDim > 0
                ? config.numAttentionHeads() * headDim
                : 0;
        int kvRows = config.resolvedNumKvHeadsForLayer(layer) > 0 && headDim > 0
                ? config.resolvedNumKvHeadsForLayer(layer) * headDim
                : 0;
        requireVector(tensors, problems, prefix + "layer_scalar", 1);
        requireVector(tensors, problems, prefix + "input_layernorm.weight", hiddenSize);
        requireVector(tensors, problems, prefix + "post_attention_layernorm.weight", hiddenSize);
        requireVector(tensors, problems, prefix + "pre_feedforward_layernorm.weight", hiddenSize);
        requireVector(tensors, problems, prefix + "post_feedforward_layernorm.weight", hiddenSize);

        TensorMeta q = requireMatrixInput(tensors, problems, prefix + "self_attn.q_proj.weight", hiddenSize);
        TensorMeta k = requireMatrixInput(tensors, problems, prefix + "self_attn.k_proj.weight", hiddenSize);
        TensorMeta o = requireMatrixInput(tensors, problems, prefix + "self_attn.o_proj.weight", -1);
        if (q != null && queryRows > 0) {
            requireDim(problems, q, 0, queryRows);
        }
        if (k != null && kvRows > 0) {
            requireDim(problems, k, 0, kvRows);
        }

        boolean valueProjectionMayBeTiedToKey =
                config.usesAlternativeAttentionForLayer(layer) || config.usesSharedKvCache(layer);
        TensorMeta v = findTensor(tensors, textCandidates(prefix + "self_attn.v_proj.weight"));
        if (v == null && !valueProjectionMayBeTiedToKey) {
            addProblem(problems, "missing tensor: " + preferredName(prefix + "self_attn.v_proj.weight"));
        } else if (v != null) {
            requireRank(problems, v, 2);
            requireDim(problems, v, 1, hiddenSize);
            if (kvRows > 0) {
                requireDim(problems, v, 0, kvRows);
            } else if (k != null && k.rank() == 2) {
                requireDim(problems, v, 0, k.dim(0));
            }
        }

        if (o != null && queryRows > 0) {
            requireShape(problems, o, hiddenSize, queryRows);
        } else if (o != null && q != null && q.rank() == 2) {
            requireShape(problems, o, hiddenSize, q.dim(0));
        }

        TensorMeta qNorm = requireTensor(tensors, problems, textCandidates(prefix + "self_attn.q_norm.weight"));
        TensorMeta kNorm = requireTensor(tensors, problems, textCandidates(prefix + "self_attn.k_norm.weight"));
        validateAttentionNorm(problems, qNorm, q, headDim);
        validateAttentionNorm(problems, kNorm, k, headDim);

        TensorMeta gate = requireMatrixInput(tensors, problems, prefix + "mlp.gate_proj.weight", hiddenSize);
        TensorMeta up = requireMatrixInput(tensors, problems, prefix + "mlp.up_proj.weight", hiddenSize);
        TensorMeta down = requireMatrixInput(tensors, problems, prefix + "mlp.down_proj.weight", -1);
        if (gate != null) {
            requireDim(problems, gate, 0, intermediateSize);
        }
        if (up != null) {
            requireDim(problems, up, 0, intermediateSize);
        }
        if (down != null) {
            requireShape(problems, down, hiddenSize, intermediateSize);
        }
    }

    private static TensorMeta requireVector(
            Map<String, TensorMeta> tensors,
            List<String> problems,
            String suffix,
            long expectedDim) {
        TensorMeta tensor = requireTensor(tensors, problems, textCandidates(suffix));
        if (tensor != null) {
            requireShape(problems, tensor, expectedDim);
        }
        return tensor;
    }

    private static TensorMeta requireMatrixInput(
            Map<String, TensorMeta> tensors,
            List<String> problems,
            String suffix,
            long expectedInputDim) {
        TensorMeta tensor = requireTensor(tensors, problems, textCandidates(suffix));
        if (tensor != null) {
            requireRank(problems, tensor, 2);
            if (expectedInputDim > 0) {
                requireDim(problems, tensor, 1, expectedInputDim);
            }
        }
        return tensor;
    }

    private static void validateAttentionNorm(
            List<String> problems,
            TensorMeta norm,
            TensorMeta projection,
            int expectedHeadDim) {
        if (norm == null) {
            return;
        }
        requireRank(problems, norm, 1);
        if (expectedHeadDim > 0) {
            requireDim(problems, norm, 0, expectedHeadDim);
            return;
        }
        if (projection == null || projection.rank() != 2 || norm.rank() != 1 || norm.dim(0) <= 0) {
            return;
        }
        long projectionRows = projection.dim(0);
        long normDim = norm.dim(0);
        if (projectionRows % normDim != 0) {
            addProblem(problems, "%s shape %s is incompatible with %s rows=%d"
                    .formatted(norm.name(), norm.shapeString(), projection.name(), projectionRows));
        }
    }

    private static TensorMeta requireTensor(
            Map<String, TensorMeta> tensors,
            List<String> problems,
            List<String> candidates) {
        TensorMeta tensor = findTensor(tensors, candidates);
        if (tensor == null) {
            addProblem(problems, "missing tensor: one of " + String.join(", ", candidates));
            return null;
        }
        if (!isSupportedFloatingDtype(tensor.dtype())) {
            addProblem(problems, "unsupported dtype for " + tensor.name() + ": " + tensor.dtype());
        }
        return tensor;
    }

    private static TensorMeta findTensor(Map<String, TensorMeta> tensors, List<String> candidates) {
        for (String candidate : candidates) {
            TensorMeta tensor = tensors.get(candidate);
            if (tensor != null) {
                return tensor;
            }
        }
        return null;
    }

    private static void requireShape(List<String> problems, TensorMeta tensor, long... expected) {
        requireRank(problems, tensor, expected.length);
        if (tensor.rank() != expected.length) {
            return;
        }
        for (int i = 0; i < expected.length; i++) {
            requireDim(problems, tensor, i, expected[i]);
        }
    }

    private static void requireRank(List<String> problems, TensorMeta tensor, int expectedRank) {
        if (tensor.rank() != expectedRank) {
            addProblem(problems, "%s shape %s must have rank %d"
                    .formatted(tensor.name(), tensor.shapeString(), expectedRank));
        }
    }

    private static void requireDim(List<String> problems, TensorMeta tensor, int dim, long expected) {
        if (dim < 0 || dim >= tensor.rank() || tensor.dim(dim) == expected) {
            return;
        }
        addProblem(problems, "%s shape %s has dim[%d]=%d, expected %d"
                .formatted(tensor.name(), tensor.shapeString(), dim, tensor.dim(dim), expected));
    }

    private static void addProblem(List<String> problems, String problem) {
        if (problems.size() < MAX_REPORTED_PROBLEMS) {
            problems.add(problem);
        } else if (problems.size() == MAX_REPORTED_PROBLEMS) {
            problems.add("additional tensor contract problems omitted");
        }
    }

    private static boolean isSupportedFloatingDtype(String dtype) {
        String normalized = dtype == null ? "" : dtype.trim().toUpperCase(Locale.ROOT);
        return normalized.equals("BF16")
                || normalized.equals("F16")
                || normalized.equals("F32")
                || normalized.equals("FLOAT16")
                || normalized.equals("FLOAT32")
                || normalized.equals("BFLOAT16");
    }

    private static List<String> textCandidates(String suffix) {
        return List.of(
                "model.language_model." + suffix,
                "language_model." + suffix,
                "model." + suffix,
                suffix);
    }

    private static String preferredName(String suffix) {
        return "model.language_model." + suffix;
    }

    private static List<Path> safetensorFiles(Path modelPath) throws IOException {
        if (Files.isRegularFile(modelPath) && isSafetensorFile(modelPath)) {
            return List.of(modelPath.toAbsolutePath().normalize());
        }
        Path modelDir = Files.isDirectory(modelPath) ? modelPath : modelPath.getParent();
        if (modelDir == null || !Files.isDirectory(modelDir)) {
            return List.of();
        }

        Optional<List<Path>> indexed = indexedSafetensorFiles(modelDir);
        if (indexed.isPresent()) {
            return indexed.get();
        }

        Path modelSafetensors = modelDir.resolve("model.safetensors");
        if (Files.isRegularFile(modelSafetensors)) {
            return List.of(modelSafetensors.toAbsolutePath().normalize());
        }

        try (var stream = Files.list(modelDir)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(Gemma4UnifiedSafetensorPreflight::isSafetensorFile)
                    .map(path -> path.toAbsolutePath().normalize())
                    .sorted(Comparator.comparing(Path::toString))
                    .toList();
        }
    }

    private static Optional<List<Path>> indexedSafetensorFiles(Path modelDir) throws IOException {
        Path index = modelDir.resolve("model.safetensors.index.json");
        if (!Files.isRegularFile(index)) {
            return Optional.empty();
        }
        JsonNode weightMap = OBJECT_MAPPER.readTree(index.toFile()).path("weight_map");
        if (!weightMap.isObject()) {
            return Optional.of(List.of());
        }
        Set<Path> shards = new LinkedHashSet<>();
        for (JsonNode shardNode : weightMap) {
            String shardName = shardNode.asText("");
            if (!shardName.isBlank()) {
                Path shard = modelDir.resolve(shardName).toAbsolutePath().normalize();
                if (Files.isRegularFile(shard) && isSafetensorFile(shard)) {
                    shards.add(shard);
                }
            }
        }
        return Optional.of(shards.stream()
                .sorted(Comparator.comparing(Path::toString))
                .toList());
    }

    private static boolean isSafetensorFile(Path path) {
        String name = path == null || path.getFileName() == null
                ? ""
                : path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".safetensors") || name.endsWith(".safetensor");
    }

    private static HeaderLoad loadTensorHeaders(List<Path> files) throws IOException {
        Map<String, TensorMeta> tensors = new LinkedHashMap<>();
        long headerBytes = 0;
        for (Path file : files) {
            HeaderFileLoad header = loadTensorHeader(file);
            tensors.putAll(header.tensors());
            headerBytes += header.headerBytes();
        }
        return new HeaderLoad(
                tensors,
                new HeaderSummary(files.size(), tensors.size(), headerBytes, 0));
    }

    private static HeaderFileLoad loadTensorHeader(Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            ByteBuffer headerLengthBuffer = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, headerLengthBuffer, 0);
            headerLengthBuffer.flip();
            long headerLength = headerLengthBuffer.getLong();
            if (headerLength <= 0 || headerLength > MAX_HEADER_BYTES || headerLength > Integer.MAX_VALUE) {
                throw new IOException("invalid SafeTensors header length " + headerLength + " for " + file);
            }

            ByteBuffer headerBuffer = ByteBuffer.allocate((int) headerLength);
            readFully(channel, headerBuffer, Long.BYTES);
            headerBuffer.flip();
            String headerJson = StandardCharsets.UTF_8.decode(headerBuffer).toString();
            JsonNode root = OBJECT_MAPPER.readTree(headerJson);
            Map<String, TensorMeta> tensors = new LinkedHashMap<>();
            for (Map.Entry<String, JsonNode> field : root.properties()) {
                if ("__metadata__".equals(field.getKey())) {
                    continue;
                }
                JsonNode node = field.getValue();
                if (!node.isObject()) {
                    continue;
                }
                tensors.put(field.getKey(), new TensorMeta(
                        field.getKey(),
                        node.path("dtype").asText(""),
                        shape(node.path("shape"))));
            }
            return new HeaderFileLoad(tensors, headerLength);
        }
    }

    private static void readFully(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        long offset = position;
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, offset);
            if (read < 0) {
                throw new IOException("unexpected end of file while reading SafeTensors header");
            }
            offset += read;
        }
    }

    private static long[] shape(JsonNode shapeNode) {
        if (!shapeNode.isArray()) {
            return new long[0];
        }
        long[] shape = new long[shapeNode.size()];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = shapeNode.get(i).asLong();
        }
        return shape;
    }

    private record TensorMeta(String name, String dtype, long[] shape) {
        public int rank() {
            return shape.length;
        }

        public long dim(int dim) {
            return shape[dim];
        }

        public String shapeString() {
            StringBuilder builder = new StringBuilder("[");
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) {
                    builder.append(',');
                }
                builder.append(shape[i]);
            }
            return builder.append(']').toString();
        }
    }

    private record HeaderLoad(Map<String, TensorMeta> tensors, HeaderSummary summary) {
    }

    private record HeaderFileLoad(Map<String, TensorMeta> tensors, long headerBytes) {
    }

    public record HeaderSummary(
            int safetensorFileCount,
            int tensorCount,
            long headerBytesRead,
            long payloadBytesLoaded) {
        public static HeaderSummary empty() {
            return new HeaderSummary(0, 0, 0, 0);
        }

        Map<String, Object> diagnosticDetails() {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put(HeaderInspection.SAFETENSOR_FILE_COUNT, safetensorFileCount);
            details.put(HeaderInspection.TENSOR_COUNT, tensorCount);
            details.put(HeaderInspection.HEADER_BYTES_READ, headerBytesRead);
            details.put(HeaderInspection.PAYLOAD_BYTES_LOADED, payloadBytesLoaded);
            return details;
        }
    }

    public record Inspection(
            Result text,
            ProjectorSummary projectors,
            HeaderSummary header,
            TensorInventorySummary inventory,
            ComponentReadinessSummary readiness) {
        public static Inspection pass(
                ProjectorSummary projectors,
                HeaderSummary header,
                TensorInventorySummary inventory,
                ComponentReadinessSummary readiness) {
            return new Inspection(
                    Result.pass(projectors, header, inventory, readiness),
                    projectors,
                    header,
                    inventory,
                    readiness);
        }

        public static Inspection invalid(
                List<String> messages,
                ProjectorSummary projectors,
                HeaderSummary header,
                TensorInventorySummary inventory,
                ComponentReadinessSummary readiness) {
            return new Inspection(
                    Result.invalid(messages, projectors, header, inventory, readiness),
                    projectors,
                    header,
                    inventory,
                    readiness);
        }
    }

    /**
     * Header-only readiness signals for the next Gemma 4 runtime implementation steps.
     */
    public record ComponentReadinessSummary(
            boolean textDecoderReady,
            boolean visionProjectorReady,
            boolean audioProjectorReady,
            boolean videoProjectorReady,
            boolean multimodalProjectorReady,
            boolean packedMoeRouterReady,
            boolean packedMoeExpertsReady,
            boolean packedMoeHeaderReady) {
        public static ComponentReadinessSummary empty() {
            return new ComponentReadinessSummary(false, false, false, false, false, false, false, false);
        }

        public static ComponentReadinessSummary from(
                boolean textDecoderReady,
                ProjectorSummary projectors,
                TensorInventorySummary inventory) {
            boolean visionReady = projectors != null && projectors.visionProjectorReady();
            boolean audioReady = projectors != null && projectors.audioProjectorReady();
            boolean videoReady = projectors != null && projectors.videoProjectorReady();
            boolean routerReady = projectors != null && projectors.packedMoeRouterReady();
            boolean expertsReady = projectors != null && projectors.packedMoeExpertsReady();
            return new ComponentReadinessSummary(
                    textDecoderReady,
                    visionReady,
                    audioReady,
                    videoReady,
                    visionReady || audioReady || videoReady,
                    routerReady,
                    expertsReady,
                    routerReady && expertsReady);
        }

        Map<String, Object> diagnosticDetails() {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put(ComponentReadiness.TEXT_DECODER_READY, textDecoderReady);
            details.put(ComponentReadiness.VISION_PROJECTOR_READY, visionProjectorReady);
            details.put(ComponentReadiness.AUDIO_PROJECTOR_READY, audioProjectorReady);
            details.put(ComponentReadiness.VIDEO_PROJECTOR_READY, videoProjectorReady);
            details.put(ComponentReadiness.MULTIMODAL_PROJECTOR_READY, multimodalProjectorReady);
            details.put(ComponentReadiness.PACKED_MOE_ROUTER_READY, packedMoeRouterReady);
            details.put(ComponentReadiness.PACKED_MOE_EXPERTS_READY, packedMoeExpertsReady);
            details.put(ComponentReadiness.PACKED_MOE_HEADER_READY, packedMoeHeaderReady);
            return details;
        }
    }

    /**
     * Header-only component inventory for Gemma 4 unified SafeTensor checkpoints.
     */
    public record TensorInventorySummary(
            int textDecoderTensors,
            int embeddingTensors,
            int logitsHeadTensors,
            int multimodalProjectorTensors,
            int visionTowerTensors,
            int audioTowerTensors,
            int videoTowerTensors,
            int packedMoeRouterTensors,
            int packedMoeExpertTensors,
            int unclassifiedTensors) {
        public static TensorInventorySummary empty() {
            return new TensorInventorySummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        public static TensorInventorySummary from(Map<String, TensorMeta> tensors) {
            if (tensors == null || tensors.isEmpty()) {
                return empty();
            }
            int textDecoder = 0;
            int embeddings = 0;
            int logitsHead = 0;
            int projectors = 0;
            int visionTower = 0;
            int audioTower = 0;
            int videoTower = 0;
            int moeRouters = 0;
            int moeExperts = 0;
            int unclassified = 0;
            for (String rawName : tensors.keySet()) {
                String name = normalizeTensorName(rawName);
                if (isMultimodalProjectorTensor(name)) {
                    projectors++;
                } else if (isVisionTowerTensor(name)) {
                    visionTower++;
                } else if (isAudioTowerTensor(name)) {
                    audioTower++;
                } else if (isVideoTowerTensor(name)) {
                    videoTower++;
                } else if (isPackedMoeRouterTensor(name)) {
                    moeRouters++;
                } else if (isPackedMoeExpertTensor(name)) {
                    moeExperts++;
                } else if (isTextEmbeddingTensor(name)) {
                    embeddings++;
                } else if (isLogitsHeadTensor(name)) {
                    logitsHead++;
                } else if (isTextDecoderTensor(name)) {
                    textDecoder++;
                } else {
                    unclassified++;
                }
            }
            return new TensorInventorySummary(
                    textDecoder,
                    embeddings,
                    logitsHead,
                    projectors,
                    visionTower,
                    audioTower,
                    videoTower,
                    moeRouters,
                    moeExperts,
                    unclassified);
        }

        Map<String, Object> diagnosticDetails() {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put(TensorInventory.TEXT_DECODER_TENSORS, textDecoderTensors);
            details.put(TensorInventory.EMBEDDING_TENSORS, embeddingTensors);
            details.put(TensorInventory.LOGITS_HEAD_TENSORS, logitsHeadTensors);
            details.put(TensorInventory.MULTIMODAL_PROJECTOR_TENSORS, multimodalProjectorTensors);
            details.put(TensorInventory.VISION_TOWER_TENSORS, visionTowerTensors);
            details.put(TensorInventory.AUDIO_TOWER_TENSORS, audioTowerTensors);
            details.put(TensorInventory.VIDEO_TOWER_TENSORS, videoTowerTensors);
            details.put(TensorInventory.PACKED_MOE_ROUTER_TENSORS, packedMoeRouterTensors);
            details.put(TensorInventory.PACKED_MOE_EXPERT_TENSORS, packedMoeExpertTensors);
            details.put(TensorInventory.UNCLASSIFIED_TENSORS, unclassifiedTensors);
            return details;
        }

        private static String normalizeTensorName(String rawName) {
            return rawName == null ? "" : rawName.toLowerCase(Locale.ROOT);
        }

        private static boolean isTextEmbeddingTensor(String name) {
            return name.endsWith("embed_tokens.weight")
                    || name.endsWith("embed_tokens_per_layer.weight")
                    || name.endsWith("per_layer_model_projection.weight")
                    || name.endsWith("per_layer_projection_norm.weight");
        }

        private static boolean isLogitsHeadTensor(String name) {
            return name.endsWith("lm_head.weight") || name.endsWith("logits_proj.weight");
        }

        private static boolean isTextDecoderTensor(String name) {
            return name.contains(".language_model.layers.")
                    || name.contains(".text_model.layers.")
                    || name.contains(".layers.")
                    || name.startsWith("layers.")
                    || name.endsWith("norm.weight");
        }

        private static boolean isMultimodalProjectorTensor(String name) {
            return name.contains("embedding_projection")
                    || name.contains("vision_projection")
                    || name.contains("image_projection")
                    || name.contains("audio_projection")
                    || name.contains("video_projection")
                    || name.contains("multi_modal_projector");
        }

        private static boolean isVisionTowerTensor(String name) {
            return (name.contains("vision_tower")
                    || name.contains("vision_model")
                    || name.contains("image_encoder")
                    || name.contains("embed_vision"))
                    && !isMultimodalProjectorTensor(name);
        }

        private static boolean isAudioTowerTensor(String name) {
            return (name.contains("audio_tower")
                    || name.contains("audio_model")
                    || name.contains("audio_encoder")
                    || name.contains("embed_audio"))
                    && !isMultimodalProjectorTensor(name);
        }

        private static boolean isVideoTowerTensor(String name) {
            return (name.contains("video_tower")
                    || name.contains("video_model")
                    || name.contains("video_encoder")
                    || name.contains("embed_video"))
                    && !isMultimodalProjectorTensor(name);
        }

        private static boolean isPackedMoeRouterTensor(String name) {
            return name.endsWith(".mlp.router.weight")
                    || name.endsWith(".mlp.router.bias")
                    || name.endsWith(".mlp.gate.weight")
                    || name.endsWith(".mlp.gate.bias")
                    || name.contains(".mlp.router.");
        }

        private static boolean isPackedMoeExpertTensor(String name) {
            return name.contains(".mlp.experts.");
        }
    }

    public record ProjectorSummary(
            boolean visionProjection,
            boolean audioProjection,
            boolean videoProjection,
            boolean visionProjectorReady,
            boolean audioProjectorReady,
            boolean videoProjectorReady,
            boolean packedMoeRouter,
            boolean packedMoeExperts,
            boolean packedMoeRouterReady,
            boolean packedMoeExpertsReady) {
        public static ProjectorSummary empty() {
            return new ProjectorSummary(false, false, false, false, false, false, false, false, false, false);
        }

        public static ProjectorSummary from(Map<String, TensorMeta> tensors, ModelConfig config) {
            if (tensors == null || tensors.isEmpty()) {
                return empty();
            }
            int hiddenSize = config == null ? 0 : config.hiddenSize();
            int numExperts = config == null ? 0 : config.numLocalExperts();
            int moeIntermediateSize = config == null ? 0 : config.moeIntermediateSize();
            TensorMeta vision = findAny(tensors, "model.embed_vision.embedding_projection.weight",
                    "embed_vision.embedding_projection.weight",
                    "model.vision_projection.weight",
                    "vision_projection.weight");
            TensorMeta audio = findAny(tensors, "model.embed_audio.embedding_projection.weight",
                    "embed_audio.embedding_projection.weight",
                    "model.audio_projection.weight",
                    "audio_projection.weight");
            TensorMeta video = findAny(tensors, "model.embed_video.embedding_projection.weight",
                    "embed_video.embedding_projection.weight",
                    "model.video_projection.weight",
                    "video_projection.weight");
            boolean packedMoeRouter = hasSuffix(tensors, ".mlp.router.weight",
                    ".mlp.router.bias",
                    ".mlp.gate.weight",
                    ".mlp.gate.bias");
            boolean packedMoeExperts = hasContains(tensors, ".mlp.experts.");
            TensorMeta packedRouterWeight = findSuffix(tensors, ".mlp.router.weight", ".mlp.gate.weight");
            return new ProjectorSummary(
                    vision != null,
                    audio != null,
                    video != null,
                    projectionCompatibleWithTextHiddenSize(vision, hiddenSize),
                    projectionCompatibleWithTextHiddenSize(audio, hiddenSize),
                    projectionCompatibleWithTextHiddenSize(video, hiddenSize),
                    packedMoeRouter,
                    packedMoeExperts,
                    routerCompatibleWithConfig(packedRouterWeight, hiddenSize, numExperts),
                    expertTripletCompatibleWithConfig(tensors, hiddenSize, moeIntermediateSize, numExperts));
        }

        public String display() {
            String projectors = projectorDisplay();
            String moe = moeDisplay();
            if (!projectors.isBlank() && !moe.isBlank()) {
                return projectors + "; " + moe;
            }
            if (!projectors.isBlank()) {
                return projectors;
            }
            if (!moe.isBlank()) {
                return moe;
            }
            return "vision/audio/video projector or packed MoE tensors were not detected";
        }

        public List<String> detectedProjectors() {
            List<String> values = new ArrayList<>();
            if (visionProjection) {
                values.add("vision");
            }
            if (audioProjection) {
                values.add("audio");
            }
            if (videoProjection) {
                values.add("video");
            }
            return List.copyOf(values);
        }

        public List<String> detectedPackedMoe() {
            List<String> values = new ArrayList<>();
            if (packedMoeRouter) {
                values.add("router");
            }
            if (packedMoeExperts) {
                values.add("experts");
            }
            return List.copyOf(values);
        }

        private String projectorDisplay() {
            if (projectorsDetectedButNotReady()) {
                return incompatibleProjectorDisplay();
            }
            if (visionProjectorReady && audioProjectorReady && videoProjectorReady) {
                return "vision/audio/video projector tensors detected";
            }
            if (visionProjectorReady && audioProjectorReady) {
                return "vision/audio projector tensors detected";
            }
            if (visionProjectorReady && videoProjectorReady) {
                return "vision/video projector tensors detected";
            }
            if (audioProjectorReady && videoProjectorReady) {
                return "audio/video projector tensors detected";
            }
            if (visionProjectorReady) {
                return "vision projector tensor detected";
            }
            if (audioProjectorReady) {
                return "audio projector tensor detected";
            }
            if (videoProjectorReady) {
                return "video projector tensor detected";
            }
            return "";
        }

        private boolean projectorsDetectedButNotReady() {
            return (visionProjection && !visionProjectorReady)
                    || (audioProjection && !audioProjectorReady)
                    || (videoProjection && !videoProjectorReady);
        }

        private String incompatibleProjectorDisplay() {
            List<String> incompatible = new ArrayList<>();
            if (visionProjection && !visionProjectorReady) {
                incompatible.add("vision");
            }
            if (audioProjection && !audioProjectorReady) {
                incompatible.add("audio");
            }
            if (videoProjection && !videoProjectorReady) {
                incompatible.add("video");
            }
            String subject = String.join("/", incompatible);
            String suffix = incompatible.size() == 1 ? "shape is" : "shapes are";
            return subject + " projector tensor detected but " + suffix
                    + " incompatible with text hidden size";
        }

        private String moeDisplay() {
            if (packedMoeDetectedButNotReady()) {
                return incompatibleMoeDisplay();
            }
            if (packedMoeRouterReady && packedMoeExpertsReady) {
                return "packed MoE router and expert tensors detected";
            }
            if (packedMoeRouterReady) {
                return "packed MoE router tensor detected";
            }
            if (packedMoeExpertsReady) {
                return "packed MoE expert tensors detected";
            }
            return "";
        }

        private boolean packedMoeDetectedButNotReady() {
            return (packedMoeRouter && !packedMoeRouterReady)
                    || (packedMoeExperts && !packedMoeExpertsReady);
        }

        private String incompatibleMoeDisplay() {
            List<String> incompatible = new ArrayList<>();
            if (packedMoeRouter && !packedMoeRouterReady) {
                incompatible.add("router");
            }
            if (packedMoeExperts && !packedMoeExpertsReady) {
                incompatible.add("experts");
            }
            String subject = String.join(" and ", incompatible);
            return "packed MoE " + subject + " tensor coverage is"
                    + " incomplete or incompatible with config dimensions";
        }

        private static TensorMeta findAny(Map<String, TensorMeta> tensors, String... names) {
            for (String name : names) {
                TensorMeta tensor = tensors.get(name);
                if (tensor != null) {
                    return tensor;
                }
            }
            return null;
        }

        private static boolean projectionCompatibleWithTextHiddenSize(TensorMeta tensor, int hiddenSize) {
            return tensor != null
                    && hiddenSize > 0
                    && isSupportedFloatingDtype(tensor.dtype())
                    && tensor.rank() == 2
                    && (tensor.dim(0) == hiddenSize || tensor.dim(1) == hiddenSize);
        }

        private static boolean routerCompatibleWithConfig(TensorMeta tensor, int hiddenSize, int numExperts) {
            return tensor != null
                    && hiddenSize > 0
                    && isSupportedFloatingDtype(tensor.dtype())
                    && tensor.rank() == 2
                    && tensor.dim(1) == hiddenSize
                    && (numExperts <= 0 || tensor.dim(0) == numExperts);
        }

        private static boolean expertTripletCompatibleWithConfig(
                Map<String, TensorMeta> tensors,
                int hiddenSize,
                int moeIntermediateSize,
                int numExperts) {
            if (tensors == null || tensors.isEmpty() || hiddenSize <= 0) {
                return false;
            }
            int expectedExperts = numExperts > 0 ? numExperts : 1;
            Map<String, Map<String, TensorMeta[]>> expertsByLayer = new LinkedHashMap<>();
            for (Map.Entry<String, TensorMeta> entry : tensors.entrySet()) {
                String name = entry.getKey();
                int marker = name.indexOf(".mlp.experts.");
                if (marker < 0) {
                    continue;
                }
                int expertStart = marker + ".mlp.experts.".length();
                int expertEnd = name.indexOf('.', expertStart);
                if (expertEnd <= expertStart) {
                    continue;
                }
                String layerKey = name.substring(0, marker);
                String expertKey = name.substring(expertStart, expertEnd);
                Map<String, TensorMeta[]> layerExperts =
                        expertsByLayer.computeIfAbsent(layerKey, ignored -> new LinkedHashMap<>());
                TensorMeta[] triplet = null;
                int projectionIndex = -1;
                if (name.endsWith(".gate_proj.weight")) {
                    projectionIndex = 0;
                    triplet = layerExperts.computeIfAbsent(expertKey, ignored -> new TensorMeta[3]);
                } else if (name.endsWith(".up_proj.weight")) {
                    projectionIndex = 1;
                    triplet = layerExperts.computeIfAbsent(expertKey, ignored -> new TensorMeta[3]);
                } else if (name.endsWith(".down_proj.weight")) {
                    projectionIndex = 2;
                    triplet = layerExperts.computeIfAbsent(expertKey, ignored -> new TensorMeta[3]);
                }
                if (triplet != null) {
                    triplet[projectionIndex] = entry.getValue();
                }
            }
            for (Map<String, TensorMeta[]> layerExperts : expertsByLayer.values()) {
                int compatibleExperts = 0;
                for (TensorMeta[] triplet : layerExperts.values()) {
                    if (expertGateOrUpCompatible(triplet[0], hiddenSize, moeIntermediateSize)
                            && expertGateOrUpCompatible(triplet[1], hiddenSize, moeIntermediateSize)
                            && expertDownCompatible(triplet[2], hiddenSize, moeIntermediateSize)) {
                        compatibleExperts++;
                    }
                }
                if (compatibleExperts >= expectedExperts) {
                    return true;
                }
            }
            return false;
        }

        private static boolean expertGateOrUpCompatible(TensorMeta tensor, int hiddenSize, int moeIntermediateSize) {
            return tensor != null
                    && isSupportedFloatingDtype(tensor.dtype())
                    && tensor.rank() == 2
                    && tensor.dim(1) == hiddenSize
                    && (moeIntermediateSize <= 0 || tensor.dim(0) == moeIntermediateSize);
        }

        private static boolean expertDownCompatible(TensorMeta tensor, int hiddenSize, int moeIntermediateSize) {
            return tensor != null
                    && isSupportedFloatingDtype(tensor.dtype())
                    && tensor.rank() == 2
                    && tensor.dim(0) == hiddenSize
                    && (moeIntermediateSize <= 0 || tensor.dim(1) == moeIntermediateSize);
        }

        private static boolean hasSuffix(Map<String, TensorMeta> tensors, String... suffixes) {
            for (String tensorName : tensors.keySet()) {
                for (String suffix : suffixes) {
                    if (tensorName.endsWith(suffix)) {
                        return true;
                    }
                }
            }
            return false;
        }

        private static TensorMeta findSuffix(Map<String, TensorMeta> tensors, String... suffixes) {
            for (Map.Entry<String, TensorMeta> entry : tensors.entrySet()) {
                for (String suffix : suffixes) {
                    if (entry.getKey().endsWith(suffix)) {
                        return entry.getValue();
                    }
                }
            }
            return null;
        }

        private static boolean hasContains(Map<String, TensorMeta> tensors, String token) {
            for (String tensorName : tensors.keySet()) {
                if (tensorName.contains(token)) {
                    return true;
                }
            }
            return false;
        }
    }
}
