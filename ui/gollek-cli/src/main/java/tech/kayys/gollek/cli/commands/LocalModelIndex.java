package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import tech.kayys.gollek.sdk.util.GollekHome;
import tech.kayys.gollek.cli.util.CLIUtils;
import tech.kayys.gollek.cli.util.QuantSuggestionDetector;

final class LocalModelIndex {

    static final class Entry {
        public String id;
        public String shortId;
        public String name;
        public String architecture;
        public String parameterCount;
        public String format;
        public boolean runnable;
        public long sizeBytes;
        public String path;
        public String updatedAt;
        public String source;

        // ── Quantization metadata ────────────────────────────────────
        /** Quantization strategy (bnb, turbo, awq, gptq, autoround), null if not quantized */
        public String quantStrategy;
        /** Bit width (4, 8, etc.), 0 if not quantized */
        public int quantBits;
        /** Group size for block quantization, 0 if not quantized */
        public int quantGroupSize;
        public String quantSourceModel;
        /** Whether the model is enabled for inference */
        public boolean enabled = true;

        // ── Modality / task grouping ─────────────────────────────────
        /**
         * High-level task category derived from the HuggingFace pipeline_tag.
         * Values: text, vision, tts, stt, ocr, multimodal, embedding, unknown.
         */
        public String taskType;
    }

    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    private LocalModelIndex() {
    }

    static synchronized List<Entry> refreshFromDisk() {
        List<Entry> entries = scanDiskEntries();
        normalizeShortIds(entries);
        write(entries);
        return entries;
    }

    static synchronized Optional<Entry> find(String ref) {
        if (ref == null || ref.isBlank()) {
            return Optional.empty();
        }
        String needle = stripProviderPrefix(ref.trim());
        List<Entry> entries = readOrRefresh();
        Optional<Entry> cached = findBestMatch(entries, needle);
        if (cached.isPresent()) {
            return cached;
        }
        List<Entry> refreshed = refreshFromDisk();
        return findBestMatch(refreshed, needle);
    }

    static synchronized Optional<Entry> find(String ref, String preferredFormat) {
        if (preferredFormat == null || preferredFormat.isBlank()) {
            return find(ref);
        }
        if (ref == null || ref.isBlank()) {
            return Optional.empty();
        }
        String needle = stripProviderPrefix(ref.trim());
        List<Entry> entries = readOrRefresh();
        Optional<Entry> cached = findBestMatch(entries.stream()
                .filter(e -> formatMatches(e, preferredFormat))
                .toList(), needle);
        if (cached.isPresent()) {
            return cached;
        }
        List<Entry> refreshed = refreshFromDisk();
        return findBestMatch(refreshed.stream()
                .filter(e -> formatMatches(e, preferredFormat))
                .toList(), needle);
    }

    static synchronized List<Entry> entries() {
        return new ArrayList<>(readOrRefresh());
    }

    private static Optional<Entry> findBestMatch(List<Entry> entries, String ref) {
        if (entries == null || entries.isEmpty()) {
            return Optional.empty();
        }
        Optional<Entry> exact = entries.stream().filter(e -> matchesExact(e, ref)).findFirst();
        if (exact.isPresent()) {
            return exact;
        }
        Optional<Entry> legacyShort = entries.stream().filter(e -> matchesLegacyShortId(e, ref)).findFirst();
        if (legacyShort.isPresent()) {
            return legacyShort;
        }
        return entries.stream().filter(e -> matchesPathSuffix(e, ref)).findFirst();
    }

    private static boolean matchesExact(Entry e, String ref) {
        if (e == null) {
            return false;
        }
        String normalizedRef = stripProviderPrefix(ref);
        return equalsRef(ref, e.id)
                || equalsRef(ref, e.shortId)
                || equalsRef(ref, e.name)
                || equalsRef(ref, e.path)
                || equalsRef(normalizedRef, e.id)
                || equalsRef(normalizedRef, e.shortId)
                || equalsRef(normalizedRef, e.name)
                || equalsRef(normalizedRef, e.path);
    }

    private static boolean matchesPathSuffix(Entry e, String ref) {
        if (e == null || ref == null || ref.isBlank() || looksLikeShortId(ref)) {
            return false;
        }
        String normalizedRef = stripProviderPrefix(ref);
        if (looksLikeShortId(normalizedRef)) {
            return false;
        }
        String normalized = normalizedRef.replace("\\", "/").toLowerCase(Locale.ROOT);
        String path = e.path != null ? e.path.replace("\\", "/").toLowerCase(Locale.ROOT) : "";
        return !normalized.isBlank() && path.endsWith(normalized);
    }

    private static boolean matchesLegacyShortId(Entry e, String ref) {
        String normalizedRef = stripProviderPrefix(ref);
        if (e == null || !looksLikeShortId(normalizedRef)) {
            return false;
        }
        for (String legacyBasis : legacyShortIdBases(e)) {
            if (equalsRef(normalizedRef, CLIUtils.generateShortId(legacyBasis))) {
                return true;
            }
        }
        return false;
    }

    private static List<String> legacyShortIdBases(Entry e) {
        List<String> bases = new ArrayList<>();
        addIfPresent(bases, e.id);
        addIfPresent(bases, e.name);
        addIfPresent(bases, e.path);
        addIfPresent(bases, repositoryRelativePath(e.path));
        return bases;
    }

    private static void addIfPresent(List<String> values, String value) {
        if (value == null || value.isBlank() || values.contains(value)) {
            return;
        }
        values.add(value);
    }

    private static String repositoryRelativePath(String absolutePath) {
        if (absolutePath == null || absolutePath.isBlank()) {
            return null;
        }
        try {
            Path path = Path.of(absolutePath).toAbsolutePath().normalize();
            Path root = GollekHome.path("models").toAbsolutePath().normalize();
            if (!path.startsWith(root)) {
                return null;
            }
            Path relative = root.relativize(path);
            if (relative.getNameCount() < 3) {
                return null;
            }
            return relative.subpath(1, relative.getNameCount()).toString().replace("\\", "/");
        } catch (Exception ignored) {
            return null;
        }
    }

    private static boolean equalsRef(String left, String right) {
        if (left == null || right == null) {
            return false;
        }
        return left.equals(right) || left.equalsIgnoreCase(right);
    }

    private static boolean looksLikeShortId(String ref) {
        if (ref == null) {
            return false;
        }
        String value = ref.trim();
        if (value.length() < 4 || value.length() > 12) {
            return false;
        }
        for (int i = 0; i < value.length(); i++) {
            if (!isHexChar(value.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    private static boolean matches(Entry e, String ref) {
        if (matchesExact(e, ref)) {
            return true;
        }
        return matchesPathSuffix(e, ref);
    }

    private static boolean formatMatches(Entry entry, String preferredFormat) {
        if (entry == null || preferredFormat == null || preferredFormat.isBlank()) {
            return false;
        }
        String entryFormat = canonicalFormat(entry.format);
        String requestedFormat = canonicalFormat(preferredFormat);
        return !entryFormat.isBlank() && entryFormat.equals(requestedFormat);
    }

    private static String canonicalFormat(String value) {
        if (value == null) {
            return "";
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "safetensor" -> "safetensors";
            case "tflite", "task", "litertlm" -> "litert";
            case "pt", "pth", "pts", "pytorch" -> "torchscript";
            default -> normalized;
        };
    }

    private static String stripProviderPrefix(String ref) {
        if (ref == null) {
            return "";
        }
        String value = ref.trim();
        if (value.startsWith("hf:")) {
            return value.substring("hf:".length()).trim();
        }
        if (value.startsWith("huggingface:")) {
            return value.substring("huggingface:".length()).trim();
        }
        return value;
    }

    private static List<Entry> readOrRefresh() {
        Path indexPath = indexPath();
        try {
            if (Files.exists(indexPath)) {
                byte[] bytes = Files.readAllBytes(indexPath);
                if (bytes.length > 0) {
                    Entry[] parsed = JSON.readValue(bytes, Entry[].class);
                    List<Entry> entries = new ArrayList<>(List.of(parsed));
                    normalizeShortIds(entries);
                    return entries;
                }
            }
        } catch (Exception ignored) {
            // fallback to refresh
        }
        return refreshFromDisk();
    }

    private static void write(List<Entry> entries) {
        Path indexPath = indexPath();
        try {
            Files.createDirectories(indexPath.getParent());
            JSON.writeValue(indexPath.toFile(), entries);
        } catch (Exception ignored) {
            // best effort cache/index only
        }
    }

    private static Path indexPath() {
        return GollekHome.path("models", "index.json");
    }

    private static List<Entry> scanDiskEntries() {
        List<Entry> out = new ArrayList<>();
        Set<String> seenPaths = new HashSet<>();
        Path root = GollekHome.path("models");
        if (!Files.isDirectory(root)) {
            return out;
        }
        scanManifestEntries(root.resolve("manifests"), out, seenPaths);
        scanFlat(root.resolve("gguf"), "gguf", true, out, seenPaths);
        scanFlat(root.resolve("libtorchscript"), "libtorchscript", true, out, seenPaths);
        scanFlat(root.resolve("torchscript"), "torchscript", true, out, seenPaths);
        scanFlat(root.resolve("safetensors"), "safetensors", true, out, seenPaths);
        scanFlat(root.resolve("litert"), "litert", true, out, seenPaths);
        scanFlat(root.resolve("onnx"), "onnx", true, out, seenPaths);

        out.sort(Comparator.comparing((Entry e) -> parseInstant(e.updatedAt)).reversed());
        return out;
    }

    private static void scanFlat(Path base, String fallbackFormat, boolean runnable, List<Entry> out, Set<String> seenPaths) {
        if (!Files.isDirectory(base)) {
            return;
        }
        Map<String, Entry> discovered = new LinkedHashMap<>();
        try (var files = Files.walk(base, 4)) {
            files.filter(Files::isRegularFile)
                    .filter(p -> !p.getFileName().toString().startsWith("."))
                    .filter(LocalModelIndex::isLikelyWeightFile)
                    .forEach(p -> {
                        String normalized = normalizePath(p);
                        if (seenPaths.add(normalized)) {
                            Entry entry = toEntry(base, p, fallbackFormat, runnable);
                            discovered.putIfAbsent(entryKey(entry), entry);
                        }
                    });
        } catch (Exception ignored) {
            // best effort
        }
        out.addAll(discovered.values());
    }

    private static void scanManifestEntries(Path manifestsDir, List<Entry> out, Set<String> seenPaths) {
        if (!Files.isDirectory(manifestsDir)) {
            return;
        }
        try (var files = Files.list(manifestsDir)) {
            files.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().endsWith(".json"))
                    .forEach(path -> {
                        try {
                            var node = JSON.readTree(path.toFile());
                            String blobPath = text(node, "blobPath");
                            if (blobPath == null || blobPath.isBlank()) {
                                return;
                            }
                            Path resolved = Path.of(blobPath).toAbsolutePath().normalize();
                            if (!Files.exists(resolved)) {
                                return;
                            }
                            if (!seenPaths.add(normalizePath(resolved))) {
                                return;
                            }

                            Entry e = new Entry();
                            e.id = text(node, "modelId");
                            if (e.id == null || e.id.isBlank()) {
                                e.id = text(node, "id");
                            }
                            e.shortId = normalizeShortId(
                                    text(node, "shortId"),
                                    e.id != null ? e.id : resolved.getFileName().toString());
                            e.name = text(node, "name");
                            if (e.name == null || e.name.isBlank()) {
                                e.name = resolved.getFileName().toString();
                            }
                            e.architecture = text(node, "architecture");
                            e.format = text(node, "format");
                            e.path = resolved.toString();
                            e.source = text(node, "source");
                            e.updatedAt = text(node, "updatedAt");
                            e.runnable = isRunnableFormat(e.format);
                            e.enabled = node.path("enabled").asBoolean(true);

                            long sizeBytes = node.path("sizeBytes").asLong(0L);
                            if (sizeBytes <= 0L) {
                                sizeBytes = computeSize(resolved);
                            }
                            e.sizeBytes = sizeBytes;

                            // ── Task type / modality ─────────────────────────────
                            e.taskType = text(node, "taskType");
                            if (e.taskType == null || e.taskType.isBlank()) {
                                // fall back to metadata.pipeline_tag
                                var metaNode = node.path("metadata");
                                if (!metaNode.isMissingNode() && metaNode.isObject()) {
                                    String tag = metaNode.path("pipeline_tag").asText(null);
                                    e.taskType = pipelineTagToTaskType(tag);
                                }
                            }
                            if (e.taskType == null || e.taskType.isBlank() || "text".equals(e.taskType)) {
                                String lowerId = (e.id != null ? e.id : "").toLowerCase(Locale.ROOT);
                                String lowerName = (e.name != null ? e.name : "").toLowerCase(Locale.ROOT);
                                if (lowerId.contains("ocr") || lowerName.contains("ocr") || lowerId.contains("pix2text") || lowerName.contains("pix2text") || lowerId.contains("trocr") || lowerId.contains("nougat")) {
                                    e.taskType = "ocr";
                                } else if (e.taskType == null || e.taskType.isBlank()) {
                                    e.taskType = "text";
                                }
                            }

                            if (isUnknownArchitecture(e.architecture)) {
                                detectArchitecture(e, Files.isDirectory(resolved) ? resolved : resolved.getParent());
                            }
                            populateQuantMetadata(e, Files.isDirectory(resolved) ? resolved : resolved.getParent());
                            out.add(e);
                        } catch (Exception ignored) {
                            // best effort
                        }
                    });
        } catch (Exception ignored) {
            // best effort
        }
    }

    private static Entry toEntry(Path base, Path file, String fallbackFormat, boolean runnable) {
        Entry e = new Entry();
        Path relative = base.relativize(file);
        e.id = deriveModelId(relative, file);
        e.shortId = normalizeShortId(null, e.id);
        e.name = deriveDisplayName(relative, file);
        e.format = detectFormat(file, fallbackFormat);
        e.runnable = runnable && !e.format.equalsIgnoreCase("safetensors") && !e.format.equalsIgnoreCase("bin");
        Path listedPath = shouldAggregateRepositoryFile(relative, file) ? file.getParent() : file;
        e.path = listedPath.toAbsolutePath().toString();
        e.source = "local";
        double pc = QuantSuggestionDetector.parseParamCount(e.name);
        e.parameterCount = pc > 0 ? String.format("%.1fB", pc) : null;
        
        try {
            e.sizeBytes = computeSize(listedPath);
            e.updatedAt = latestModified(listedPath).toString();
        } catch (Exception ignored) {
            e.sizeBytes = 0L;
        }

        // Try to detect architecture from config.json
        detectArchitecture(e, file.getParent());

        // Populate quantization metadata from gollek_quant.json if present
        populateQuantMetadata(e, file.getParent());
        return e;
    }

    private static void normalizeShortIds(List<Entry> entries) {
        if (entries == null) {
            return;
        }
        for (Entry entry : entries) {
            if (entry == null) {
                continue;
            }
            String basis = entry.id;
            if (basis == null || basis.isBlank()) {
                basis = entry.path;
            }
            if (basis == null || basis.isBlank()) {
                basis = entry.name;
            }
            entry.shortId = normalizeShortId(entry.shortId, basis);
        }
    }

    private static String normalizeShortId(String candidate, String basis) {
        if (isHexShortId(candidate)) {
            return candidate.toLowerCase(Locale.ROOT);
        }
        return CLIUtils.generateShortId(basis);
    }

    private static boolean isHexShortId(String value) {
        return value != null
                && value.length() == 6
                && value.toLowerCase(Locale.ROOT).chars().allMatch(LocalModelIndex::isHexChar);
    }

    private static boolean isHexChar(int ch) {
        return (ch >= '0' && ch <= '9')
                || (ch >= 'a' && ch <= 'f')
                || (ch >= 'A' && ch <= 'F');
    }

    private static String deriveModelId(Path relative, Path file) {
        if (relative == null) {
            return "";
        }
        String normalized = relative.toString().replace("\\", "/");
        if (relative.getNameCount() >= 3) {
            String fileName = file.getFileName().toString();
            if (!shouldUseRepositoryDisplayName(fileName)) {
                return relative.getName(0) + "/" + fileName;
            }
            return relative.getName(0) + "/" + relative.getName(1);
        }
        return normalized;
    }

    private static String entryKey(Entry entry) {
        String id = entry.id != null ? entry.id : "";
        String format = entry.format != null ? entry.format : "";
        String path = entry.path != null ? entry.path : "";
        return id + "|" + format + "|" + path;
    }

    private static String deriveDisplayName(Path relative, Path file) {
        if (relative != null && relative.getNameCount() >= 3 && shouldUseRepositoryDisplayName(file.getFileName().toString())) {
            return relative.getName(1).toString();
        }
        String fileName = file.getFileName().toString();
        String lower = fileName.toLowerCase(Locale.ROOT);
        if ((lower.equals("model.safetensors")
                || lower.equals("model.safetensor")
                || lower.equals("model.onnx")
                || lower.equals("model.tflite")
                || lower.equals("model.litert")
                || lower.equals("model.litertlm")
                || lower.equals("model.task"))
                && file.getParent() != null
                && file.getParent().getFileName() != null) {
            return file.getParent().getFileName().toString();
        }
        return fileName;
    }

    private static boolean shouldUseRepositoryDisplayName(String fileName) {
        if (fileName == null || fileName.isBlank()) {
            return false;
        }
        String lower = fileName.toLowerCase(Locale.ROOT);
        return lower.endsWith(".onnx")
                || lower.equals("model.safetensors")
                || lower.equals("model.safetensor")
                || lower.equals("model.tflite")
                || lower.equals("model.litert")
                || lower.equals("model.litertlm")
                || lower.equals("model.task");
    }

    private static boolean shouldAggregateRepositoryFile(Path relative, Path file) {
        if (relative == null || relative.getNameCount() < 3 || file == null) {
            return false;
        }
        String lower = file.getFileName().toString().toLowerCase(Locale.ROOT);
        return lower.endsWith(".onnx");
    }

    /**
     * Maps a HuggingFace {@code pipeline_tag} value to a gollek task-type label.
     * Returns {@code null} when the tag is blank or unrecognised (caller decides default).
     */
    static String pipelineTagToTaskType(String pipelineTag) {
        if (pipelineTag == null || pipelineTag.isBlank()) {
            return null;
        }
        return switch (pipelineTag.trim().toLowerCase(Locale.ROOT)) {
            case "text-generation",
                 "text2text-generation",
                 "conversational",
                 "fill-mask"             -> "text";
            case "automatic-speech-recognition",
                 "audio-classification"  -> "stt";
            case "text-to-audio",
                 "text-to-speech"        -> "tts";
            case "ocr",
                 "optical-character-recognition" -> "ocr";
            case "image-to-text",
                 "document-question-answering",
                 "visual-question-answering",
                 "image-text-to-text"    -> "vision";
            case "image-classification",
                 "object-detection",
                 "image-segmentation",
                 "image-to-image"        -> "vision";
            case "feature-extraction",
                 "sentence-similarity"   -> "embedding";
            case "time-series-forecasting",
                 "time-series-classification",
                 "tabular-regression",
                 "tabular-classification" -> "timeseries";
            default                      -> null;
        };
    }

    private static boolean isRunnableFormat(String format) {
        if (format == null || format.isBlank()) {
            return true;
        }
        String normalized = format.toLowerCase(Locale.ROOT);
        return normalized.equals("gguf")
                || normalized.equals("safetensors")
                || normalized.equals("safetensor")
                || normalized.equals("litert")
                || normalized.equals("onnx")
                || normalized.equals("task")
                || normalized.equals("tflite");
    }

    private static long computeSize(Path path) {
        try {
            if (Files.isRegularFile(path)) {
                return Files.size(path);
            }
            if (Files.isDirectory(path)) {
                try (var walk = Files.walk(path)) {
                    return walk.filter(Files::isRegularFile)
                            .mapToLong(p -> {
                                try {
                                    return Files.size(p);
                                } catch (Exception ignored) {
                                    return 0L;
                                }
                            })
                            .sum();
                }
            }
        } catch (Exception ignored) {
            // best effort
        }
        return 0L;
    }

    private static Instant latestModified(Path path) {
        try {
            if (Files.isRegularFile(path)) {
                return Files.getLastModifiedTime(path).toInstant();
            }
            if (Files.isDirectory(path)) {
                try (var walk = Files.walk(path)) {
                    return walk.filter(Files::isRegularFile)
                            .map(p -> {
                                try {
                                    return Files.getLastModifiedTime(p).toInstant();
                                } catch (Exception ignored) {
                                    return Instant.EPOCH;
                                }
                            })
                            .max(Comparator.naturalOrder())
                            .orElse(Instant.EPOCH);
                }
            }
        } catch (Exception ignored) {
            // best effort
        }
        return Instant.EPOCH;
    }

    private static String normalizePath(Path path) {
        return path.toAbsolutePath().normalize().toString();
    }

    private static String text(com.fasterxml.jackson.databind.JsonNode node, String field) {
        if (node == null || field == null || !node.hasNonNull(field)) {
            return null;
        }
        String value = node.get(field).asText();
        return value == null || value.isBlank() ? null : value;
    }

    private static void detectArchitecture(Entry e, Path modelDir) {
        if (modelDir == null) return;
        if (Files.isRegularFile(modelDir.resolve("tts_browser_onnx_meta.json"))) {
            e.architecture = "moss-tts";
            return;
        }
        if (Files.isRegularFile(modelDir.resolve("codec_browser_onnx_meta.json"))) {
            e.architecture = "moss-audio-tokenizer";
            return;
        }
        Path configJson = modelDir.resolve("config.json");
        if (!Files.isRegularFile(configJson)) return;
        try {
            var node = JSON.readTree(configJson.toFile());
            if (node.has("architectures") && node.get("architectures").isArray() && node.get("architectures").size() > 0) {
                e.architecture = node.get("architectures").get(0).asText();
            } else if (node.has("model_type")) {
                e.architecture = node.get("model_type").asText();
            }
        } catch (Exception ignored) {
            // best effort
        }
    }

    private static boolean isUnknownArchitecture(String architecture) {
        return architecture == null
                || architecture.isBlank()
                || "unknown".equalsIgnoreCase(architecture);
    }

    @SuppressWarnings("unchecked")
    private static void populateQuantMetadata(Entry e, Path modelDir) {
        if (modelDir == null) return;
        Path quantMeta = modelDir.resolve("gollek_quant.json");
        if (!Files.isRegularFile(quantMeta)) return;
        try {
            var map = JSON.readValue(quantMeta.toFile(), java.util.Map.class);
            var quant = (java.util.Map<String, Object>) map.get("gollek_quant");
            if (quant != null) {
                e.quantStrategy = (String) quant.get("strategy");
                e.quantBits = quant.get("bits") instanceof Number n ? n.intValue() : 0;
                e.quantGroupSize = quant.get("group_size") instanceof Number n ? n.intValue() : 0;
                e.quantSourceModel = (String) quant.get("source_model");
            }
        } catch (Exception ignored) {
            // best effort — skip if malformed
        }
    }

    private static boolean isLikelyWeightFile(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".gguf")
                || name.endsWith(".safetensors")
                || name.endsWith(".safetensor")
                || name.endsWith(".onnx")
                || name.endsWith(".tflite")
                || name.endsWith(".litert")
                || name.endsWith(".pb")
                || name.endsWith(".ckpt")
                || name.endsWith(".pt")
                || name.endsWith(".pth")
                || name.endsWith(".bin")
                || name.endsWith(".litertlm")
                || name.endsWith(".task")
                || !name.contains(".");
    }

    private static String detectFormat(Path file, String fallback) {
        String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".gguf")) {
            return "gguf";
        }
        if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
            return "safetensors";
        }
        if (name.endsWith(".onnx")) {
            return "onnx";
        }
        if (name.endsWith(".tflite") || name.endsWith(".litert") || name.endsWith(".litertlm") || name.endsWith(".task")) {
            return "litert";
        }
        if (name.endsWith(".pt") || name.endsWith(".pth")) {
            return "torchscript";
        }
        if (name.endsWith(".bin")) {
            return "bin";
        }
        return fallback;
    }

    static Instant parseInstant(String value) {
        if (value == null || value.isBlank()) {
            return Instant.EPOCH;
        }
        try {
            return Instant.parse(value);
        } catch (Exception ignored) {
            return Instant.EPOCH;
        }
    }
}
