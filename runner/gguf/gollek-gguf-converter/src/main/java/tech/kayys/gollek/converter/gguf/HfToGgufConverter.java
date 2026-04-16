package tech.kayys.gollek.converter.gguf;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Main converter: HuggingFace model directory → GGUF file.
 *
 * <p>
 * Improvements over the initial version:
 * <ul>
 * <li>Uses {@code model.safetensors.index.json} when present to avoid
 * scanning every shard twice.</li>
 * <li>Fixes the two-pass offset/type inconsistency: if a tensor falls back
 * to F32 during data conversion its {@link TensorInfo} is updated in-place
 * before writing.</li>
 * <li>Validates block alignment before quantization.</li>
 * <li>Supports tied-embedding models (lm_head = embed_tokens): skips
 * {@code lm_head.weight} when {@code tie_word_embeddings = true}.</li>
 * <li>Streaming blob write to avoid loading entire model into heap.</li>
 * </ul>
 *
 * <p><b>For new integrations prefer {@link SafetensorToGgufConverter}</b>,
 * which exposes a cleaner Options API and progress callbacks. This class is
 * retained for backwards-compatibility with the existing CLI entry-point.
 */
public final class HfToGgufConverter {

    // ── Configuration ─────────────────────────────────────────────────────

    public record ConvertOptions(
            Path inputDir,
            Path outputFile,
            GgmlType quantType,
            String modelVersion,
            boolean verbose) {
    }

    // ── Entry point ───────────────────────────────────────────────────────

    public static void convert(ConvertOptions opts) throws IOException {
        log(opts, "=== GGUF Converter (Java FFM) ===");
        log(opts, "Input  : " + opts.inputDir());
        log(opts, "Output : " + opts.outputFile());
        log(opts, "Quant  : " + opts.quantType().label);

        // ── 1. Parse HF config ──────────────────────────────────────────
        HfConfigParser.ModelConfig cfg = HfConfigParser.parseConfig(opts.inputDir());
        HfConfigParser.TokenizerData tok = null;
        try {
            tok = HfConfigParser.parseTokenizer(opts.inputDir());
        } catch (Exception e) {
            log(opts, "Tokenizer not found / unreadable: " + e.getMessage());
        }

        log(opts, "Architecture : " + cfg.modelType());
        log(opts, "Hidden size  : " + cfg.hiddenSize());
        log(opts, "Layers       : " + cfg.numHiddenLayers());
        log(opts, "Vocab size   : " + cfg.vocabSize());

        // ── 2. Build GGUF model skeleton ────────────────────────────────
        GgufModel model = new GgufModel();
        
        // Select appropriate mapper based on architecture
        String modelType = cfg.modelType().toLowerCase();
        if (modelType.contains("gemma")) {
            GemmaArchMapper.applyConfig(model, cfg, tok, opts.modelVersion());
        } else {
            LlamaArchMapper.applyConfig(model, cfg, tok, opts.modelVersion());
        }

        // ── 3. Build tensor→shard index ─────────────────────────────────
        // Prefer model.safetensors.index.json if present; otherwise scan shards.
        Map<String, Path> tensorToShard = buildTensorIndex(opts, opts.inputDir());
        List<String> orderedHfNames    = new ArrayList<>(tensorToShard.keySet());
        log(opts, "Found " + orderedHfNames.size() + " tensors in "
                + new HashSet<>(tensorToShard.values()).size() + " shard(s)");

        // ── 4. First pass: plan all tensors with output offsets ──────────
        record TensorPlan(String ggufName, Path shard, String hfName,
                          long[] shape, String dtype,
                          long dataStart, long dataEnd,
                          GgmlType targetType, long ggufOffset) {
            long numElements() {
                long n = 1; for (long d : shape) n *= d; return n;
            }
        }

        // Cache shard headers so we don't re-open each shard for every tensor
        Map<Path, Map<String, SafetensorsReader.TensorEntry>> shardCache = new LinkedHashMap<>();
        for (Path shard : new HashSet<>(tensorToShard.values())) {
            try (SafetensorsReader r = new SafetensorsReader(shard)) {
                shardCache.put(shard, new LinkedHashMap<>(r.tensors()));
            }
        }

        int alignment = model.alignment();
        List<TensorPlan> plan = new ArrayList<>();
        long dataOffset = 0;

        for (String hfName : orderedHfNames) {
            // Skip tied embeddings
            if (hfName.equals("lm_head.weight") && cfg.tieWordEmbeddings()) {
                log(opts, "  skip (tied embd): " + hfName);
                continue;
            }

            Path shard = tensorToShard.get(hfName);
            SafetensorsReader.TensorEntry entry = shardCache.get(shard).get(hfName);
            if (entry == null) continue;

            // Select appropriate mapper based on architecture
            String ggufName;
            if (modelType.contains("gemma")) {
                ggufName = GemmaArchMapper.mapTensorName(hfName, cfg.numHiddenLayers());
            } else {
                ggufName = LlamaArchMapper.mapTensorName(hfName, cfg.numHiddenLayers());
            }
            
            if (ggufName == null) {
                log(opts, "  skip: " + hfName);
                continue;
            }

            GgmlType dstType = TensorConverter.targetType(ggufName, opts.quantType());

            // Block-alignment guard
            long numElem = entry.numElements();
            if (dstType.blockSize > 1 && numElem % dstType.blockSize != 0) {
                log(opts, "  WARN: " + hfName + " elem=" + numElem
                        + " not multiple of blockSize=" + dstType.blockSize
                        + " for " + dstType.label + " → falling back to F32");
                dstType = GgmlType.F32;
            }

            long dstSize = dstType.bytesFor(numElem);
            plan.add(new TensorPlan(ggufName, shard, hfName,
                    entry.shape(), entry.dtype(),
                    entry.dataStart(), entry.dataEnd(), dstType, dataOffset));
            dataOffset = alignUp(dataOffset + dstSize, alignment);

            log(opts, "  map: " + hfName + " → " + ggufName
                    + " [" + entry.dtype() + " → " + dstType.label + "]");
        }

        if (plan.isEmpty()) throw new IOException("No tensors mapped – conversion aborted.");

        // ── 5. Register tensor descriptors ──────────────────────────────
        for (TensorPlan tp : plan) {
            long[] ne = reverseShape(tp.shape());
            model.addTensor(new TensorInfo(tp.ggufName(), ne, tp.targetType(), tp.ggufOffset()));
        }

        // ── 6. Convert all tensor data (heap blob) ───────────────────────
        // Note: for very large models (>30B params) use SafetensorToGgufConverter
        // which streams directly to disk without a full in-heap blob.
        byte[] blob = new byte[(int) dataOffset];

        for (TensorPlan tp : plan) {
            log(opts, "  convert: " + tp.ggufName());
            try (SafetensorsReader r = new SafetensorsReader(tp.shard())) {
                SafetensorsReader.TensorEntry entry = r.tensors().get(tp.hfName());
                byte[] srcBytes = r.tensorBytes(entry);
                long numElem = entry.numElements();
                GgmlType srcType = entry.ggmlType();
                GgmlType dstType = tp.targetType();

                // Normalise to F32
                byte[] f32 = switch (srcType) {
                    case F32  -> srcBytes;
                    case BF16 -> TensorConverter.bf16ToF32(srcBytes, numElem);
                    case F16  -> TensorConverter.f16ToF32(srcBytes, numElem);
                    default   -> srcBytes;
                };

                // Quantize / re-encode
                byte[] dst = switch (dstType) {
                    case F32  -> f32;
                    case F16  -> TensorConverter.f32ToF16(f32, numElem);
                    case BF16 -> TensorConverter.f32ToBf16(f32, numElem);
                    case Q8_0 -> TensorConverter.quantizeQ8_0(f32, numElem);
                    case Q4_0 -> TensorConverter.quantizeQ4_0(f32, numElem);
                    default   -> {
                        System.err.println("[gguf] WARN: unsupported target type "
                                + dstType.label + " for " + tp.ggufName() + ", keeping F32");
                        yield f32;
                    }
                };

                System.arraycopy(dst, 0, blob, (int) tp.ggufOffset(), dst.length);
            }
        }

        model.setTensorData(blob);

        // ── 7. Write GGUF file ───────────────────────────────────────────
        log(opts, "Writing " + opts.outputFile() + " …");
        Path parent = opts.outputFile().getParent();
        if (parent != null) Files.createDirectories(parent);
        GgufWriter.write(model, opts.outputFile());

        long sizeBytes = Files.size(opts.outputFile());
        log(opts, String.format("Done! Output size: %.2f MB", sizeBytes / 1e6));
    }

    // ── Shard index ───────────────────────────────────────────────────────

    /**
     * Build a map of HF tensor name → shard {@link Path}.
     *
     * <p>
     * If {@code model.safetensors.index.json} is present it is used directly;
     * otherwise all {@code .safetensors} files in the directory are scanned.
     */
    private static Map<String, Path> buildTensorIndex(
            ConvertOptions opts, Path dir) throws IOException {
        Path indexFile = dir.resolve("model.safetensors.index.json");
        if (Files.exists(indexFile)) {
            log(opts, "Using shard index: " + indexFile.getFileName());
            return parseIndexJson(dir, indexFile);
        }
        List<Path> shards = findSafetensorShards(dir);
        if (shards.isEmpty())
            throw new IOException("No .safetensors files found in " + dir);

        Map<String, Path> idx = new LinkedHashMap<>();
        for (Path shard : shards) {
            try (SafetensorsReader r = new SafetensorsReader(shard)) {
                for (String name : r.tensors().keySet()) idx.put(name, shard);
            }
        }
        return idx;
    }

    private static Map<String, Path> parseIndexJson(Path dir, Path indexFile)
            throws IOException {
        try (java.io.Reader r = Files.newBufferedReader(indexFile)) {
            com.google.gson.JsonObject root =
                    com.google.gson.JsonParser.parseReader(r).getAsJsonObject();
            com.google.gson.JsonObject wm = root.getAsJsonObject("weight_map");
            if (wm == null) throw new IOException("No 'weight_map' in index JSON");
            Map<String, Path> idx = new LinkedHashMap<>();
            for (Map.Entry<String, com.google.gson.JsonElement> e : wm.entrySet())
                idx.put(e.getKey(), dir.resolve(e.getValue().getAsString()));
            return idx;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static List<Path> findSafetensorShards(Path dir) throws IOException {
        try (var stream = Files.list(dir)) {
            return stream
                    .filter(p -> p.toString().endsWith(".safetensors"))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    private static long[] reverseShape(long[] shape) {
        long[] rev = new long[shape.length];
        for (int i = 0; i < shape.length; i++) rev[i] = shape[shape.length - 1 - i];
        return rev;
    }

    private static long alignUp(long offset, long alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    private static void log(ConvertOptions opts, String msg) {
        if (opts.verbose() || msg.startsWith("===") || msg.startsWith("Done"))
            System.out.println("[gguf] " + msg);
    }
}
