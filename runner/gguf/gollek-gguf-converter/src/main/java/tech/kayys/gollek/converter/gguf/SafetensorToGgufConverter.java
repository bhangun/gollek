package tech.kayys.gollek.converter.gguf;

import com.google.gson.*;

import java.io.*;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

/**
 * High-level converter: one or more {@code .safetensors} shards → GGUF.
 *
 * <p>
 * Differences from {@link HfToGgufConverter}:
 * <ul>
 * <li>Reads {@code model.safetensors.index.json} to get the authoritative
 * shard-to-tensor mapping (avoids re-scanning every shard for each
 * tensor).</li>
 * <li>Writes tensor data <em>streaming</em> to disk instead of
 * accumulating a single in-heap blob – safe for 70B+ models.</li>
 * <li>Exposes a progress callback so callers can render a progress bar.</li>
 * <li>Validates tensor element counts against expected block alignment
 * before writing.</li>
 * <li>Supports {@code generation_config.json} for additional metadata
 * (pad_token_id, chat_template).</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * var opts = new SafetensorToGgufConverter.Options.Builder()
 *         .inputDir(Path.of("/models/llama-3-8b"))
 *         .outputFile(Path.of("llama3-8b-f16.gguf"))
 *         .quantType(GgmlType.F16)
 *         .modelVersion("3.0")
 *         .verbose(true)
 *         .onProgress((done, total) -> System.out.printf("\r%d/%d tensors", done, total))
 *         .build();
 * SafetensorToGgufConverter.convert(opts);
 * }</pre>
 */
public final class SafetensorToGgufConverter {

    // ── Options ───────────────────────────────────────────────────────────

    public static final class Options {
        public final Path inputDir;
        public final Path outputFile;
        public final GgmlType quantType;
        public final String modelVersion;
        public final boolean verbose;
        /** Called with (tensorsWritten, totalTensors) after each tensor. May be null. */
        public final BiConsumer<Integer, Integer> onProgress;

        private Options(Builder b) {
            this.inputDir     = Objects.requireNonNull(b.inputDir, "inputDir");
            this.outputFile   = Objects.requireNonNull(b.outputFile, "outputFile");
            this.quantType    = b.quantType != null ? b.quantType : GgmlType.F16;
            this.modelVersion = b.modelVersion != null ? b.modelVersion : "1.0";
            this.verbose      = b.verbose;
            this.onProgress   = b.onProgress;
        }

        public static final class Builder {
            Path inputDir, outputFile;
            GgmlType quantType;
            String modelVersion;
            boolean verbose;
            BiConsumer<Integer, Integer> onProgress;

            public Builder inputDir(Path p)    { this.inputDir = p; return this; }
            public Builder outputFile(Path p)  { this.outputFile = p; return this; }
            public Builder quantType(GgmlType t){ this.quantType = t; return this; }
            public Builder modelVersion(String v){ this.modelVersion = v; return this; }
            public Builder verbose(boolean v)  { this.verbose = v; return this; }
            public Builder onProgress(BiConsumer<Integer, Integer> cb) { this.onProgress = cb; return this; }
            public Options build() { return new Options(this); }
        }
    }

    // ── Internal plan entry ───────────────────────────────────────────────

    private record TensorPlan(
            String ggufName,
            Path shard,
            String srcName,       // original HF tensor name
            long[] shape,         // HF shape (outermost→innermost)
            String dtype,         // e.g. "BF16"
            long dataStart,       // offset inside shard data blob
            long dataEnd,
            GgmlType targetType,
            long ggufOffset       // byte offset in GGUF tensor-data section
    ) {
        long numElements() {
            long n = 1;
            for (long d : shape) n *= d;
            return n;
        }
        long srcBytes() { return dataEnd - dataStart; }
    }

    // ── Entry point ───────────────────────────────────────────────────────

    public static void convert(Options opts) throws IOException {
        log(opts, "=== SafetensorToGgufConverter ===");
        log(opts, "Input  : " + opts.inputDir);
        log(opts, "Output : " + opts.outputFile);
        log(opts, "Quant  : " + opts.quantType.label);

        // 1. Parse HF config + tokenizer
        HfConfigParser.ModelConfig cfg   = HfConfigParser.parseConfig(opts.inputDir);
        HfConfigParser.TokenizerData tok = tryParseTokenizer(opts, cfg);

        log(opts, "Architecture : " + cfg.modelType());
        log(opts, "Layers       : " + cfg.numHiddenLayers());
        log(opts, "Vocab size   : " + cfg.vocabSize());

        // 2. Build GGUF model skeleton (metadata)
        GgufModel model = new GgufModel();
        
        // Select appropriate mapper based on architecture
        String modelType = cfg.modelType().toLowerCase();
        if (modelType.contains("gemma")) {
            GemmaArchMapper.applyConfig(model, cfg, tok, opts.modelVersion);
        } else {
            LlamaArchMapper.applyConfig(model, cfg, tok, opts.modelVersion);
        }
        
        applyGenerationConfig(opts, model);

        // 3. Discover shards + build tensor→shard index
        Map<String, Path> tensorToShard = buildTensorShardIndex(opts);
        List<String> orderedHfNames     = orderedTensorNames(opts, tensorToShard);

        log(opts, "Total tensors to map: " + orderedHfNames.size());

        // 4. First pass: build the conversion plan (no data loaded yet)
        List<TensorPlan> plan = buildPlan(opts, cfg, tensorToShard, orderedHfNames);

        log(opts, "Tensors in plan: " + plan.size());
        if (plan.isEmpty()) throw new IOException("No tensors could be mapped – aborting.");

        // 5. Register tensor descriptors in the model
        for (TensorPlan tp : plan) {
            long[] ne = reverseShape(tp.shape());
            model.addTensor(new TensorInfo(tp.ggufName(), ne, tp.targetType(), tp.ggufOffset()));
        }

        // 6. Write GGUF header + metadata + tensor-info to a temp file,
        //    then stream tensor data, then assemble final file.
        streamingWrite(opts, model, plan);

        long sizeBytes = Files.size(opts.outputFile);
        log(opts, String.format("Done! Output size: %.2f MB", sizeBytes / 1e6));
    }

    // ── Shard discovery ───────────────────────────────────────────────────

    /**
     * Returns a map of HF tensor name → shard path.
     *
     * <p>If {@code model.safetensors.index.json} exists it is used directly;
     * otherwise every {@code .safetensors} file is scanned to build the index.
     */
    private static Map<String, Path> buildTensorShardIndex(Options opts) throws IOException {
        Path indexFile = opts.inputDir.resolve("model.safetensors.index.json");
        if (Files.exists(indexFile)) {
            return parseIndexJson(opts, indexFile);
        }
        // Single-shard or no index: scan all .safetensors files
        List<Path> shards = findShards(opts.inputDir);
        if (shards.isEmpty())
            throw new IOException("No .safetensors files found in " + opts.inputDir);
        Map<String, Path> idx = new LinkedHashMap<>();
        for (Path shard : shards) {
            try (SafetensorsReader r = new SafetensorsReader(shard)) {
                for (String name : r.tensors().keySet()) {
                    idx.put(name, shard);
                }
            }
        }
        log(opts, "Built shard index from " + shards.size() + " file(s), "
                + idx.size() + " tensors");
        return idx;
    }

    /** Parse the HuggingFace shard index JSON. */
    private static Map<String, Path> parseIndexJson(Options opts, Path indexFile)
            throws IOException {
        try (Reader r = Files.newBufferedReader(indexFile)) {
            JsonObject root = JsonParser.parseReader(r).getAsJsonObject();
            JsonObject weightMap = root.getAsJsonObject("weight_map");
            if (weightMap == null)
                throw new IOException("model.safetensors.index.json has no 'weight_map'");

            Map<String, Path> idx = new LinkedHashMap<>();
            for (Map.Entry<String, JsonElement> e : weightMap.entrySet()) {
                String filename = e.getValue().getAsString();
                idx.put(e.getKey(), opts.inputDir.resolve(filename));
            }
            log(opts, "Loaded shard index: " + idx.size() + " tensors across "
                    + weightMap.entrySet().stream()
                      .map(e -> e.getValue().getAsString()).collect(Collectors.toSet()).size()
                    + " shard(s)");
            return idx;
        }
    }

    /**
     * Returns tensor names in a stable order: shard-declaration order, then
     * alphabetical within each shard.
     */
    private static List<String> orderedTensorNames(
            Options opts, Map<String, Path> tensorToShard) throws IOException {
        // Group by shard, preserve shard file order, within-shard order from header
        Map<Path, List<String>> byShard = new LinkedHashMap<>();
        for (Map.Entry<String, Path> e : tensorToShard.entrySet()) {
            byShard.computeIfAbsent(e.getValue(), k -> new ArrayList<>()).add(e.getKey());
        }
        List<String> ordered = new ArrayList<>();
        for (Path shard : byShard.keySet()) {
            // Re-read the shard header to get the original declaration order
            try (SafetensorsReader r = new SafetensorsReader(shard)) {
                List<String> shardNames = new ArrayList<>(r.tensors().keySet());
                // Only keep names that belong to this shard per the index
                Set<String> inThisShard = new HashSet<>(byShard.get(shard));
                shardNames.removeIf(n -> !inThisShard.contains(n));
                ordered.addAll(shardNames);
            }
        }
        return ordered;
    }

    // ── Plan construction ─────────────────────────────────────────────────

    private static List<TensorPlan> buildPlan(
            Options opts,
            HfConfigParser.ModelConfig cfg,
            Map<String, Path> tensorToShard,
            List<String> orderedHfNames) throws IOException {

        int alignment = GgufModel.DEFAULT_ALIGNMENT;

        // Cache open readers to avoid re-opening the same shard for metadata
        Map<Path, Map<String, SafetensorsReader.TensorEntry>> shardMeta = new LinkedHashMap<>();
        for (Map.Entry<String, Path> e : tensorToShard.entrySet()) {
            Path shard = e.getValue();
            if (!shardMeta.containsKey(shard)) {
                try (SafetensorsReader r = new SafetensorsReader(shard)) {
                    shardMeta.put(shard, new LinkedHashMap<>(r.tensors()));
                }
            }
        }

        List<TensorPlan> plan = new ArrayList<>();
        long dataOffset = 0;

        for (String hfName : orderedHfNames) {
            Path shard = tensorToShard.get(hfName);
            if (shard == null) continue;

            SafetensorsReader.TensorEntry entry = shardMeta.get(shard).get(hfName);
            if (entry == null) continue;

            // Select appropriate mapper based on architecture
            String modelType = cfg.modelType().toLowerCase();
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

            GgmlType srcType = entry.ggmlType();
            GgmlType dstType = TensorConverter.targetType(ggufName, opts.quantType);

            // Validate element count for block-aligned quantization types
            long numElem = entry.numElements();
            if (dstType.blockSize > 1 && numElem % dstType.blockSize != 0) {
                log(opts, "  WARN: " + hfName + " has " + numElem +
                        " elements, not multiple of " + dstType.blockSize +
                        " for " + dstType.label + " – falling back to F32");
                dstType = GgmlType.F32;
            }

            long dstSize = dstType.bytesFor(numElem);
            long ggufOffset = dataOffset;
            dataOffset = alignUp(dataOffset + dstSize, alignment);

            log(opts, "  map: " + hfName + " → " + ggufName
                    + " [" + entry.dtype() + " → " + dstType.label
                    + ", shape=" + Arrays.toString(entry.shape()) + "]");

            plan.add(new TensorPlan(
                    ggufName, shard, hfName, entry.shape(), entry.dtype(),
                    entry.dataStart(), entry.dataEnd(), dstType, ggufOffset));
        }

        // Close cached metadata (no open resources held – SafetensorsReader was
        // used in try-with-resources above and the map just holds the entries).
        return plan;
    }

    // ── Streaming write ───────────────────────────────────────────────────

    /**
     * Two-step write:
     * <ol>
     * <li>Serialize GGUF header + KV + tensor-info to a temp file.</li>
     * <li>Open final output, write the meta segment, then stream each
     * tensor's converted data directly from its mmap'd shard.</li>
     * </ol>
     * This keeps heap usage proportional to a single (largest) tensor,
     * not the entire model.
     */
    private static void streamingWrite(Options opts, GgufModel model,
            List<TensorPlan> plan) throws IOException {

        int alignment = model.alignment();
        
        log(opts, "=== STREAMING WRITE STARTED ===");
        log(opts, "Plan size: " + plan.size() + " tensors");
        log(opts, "Output file: " + opts.outputFile);

        // Step 1: write metadata to a temp file to get its byte length
        Path metaTemp = Files.createTempFile("gguf-meta-", ".bin");
        try {
            // Write with empty tensor data blob so GgufWriter builds the header
            model.setTensorData(new byte[0]);
            GgufWriter.write(model, metaTemp);
            long metaLen = Files.size(metaTemp);
            
            log(opts, "Metadata written: " + metaLen + " bytes");

            // Step 2: create final output and stream
            Path parent = opts.outputFile.getParent();
            if (parent != null) Files.createDirectories(parent);

            log(opts, "Writing GGUF (streaming) …");
            try (FileOutputStream fos = new FileOutputStream(opts.outputFile.toFile())) {

                // Copy meta section
                Files.copy(metaTemp, fos);

                // Stream each tensor
                int total  = plan.size();
                long written = metaLen; // track position for alignment padding

                // We need data starting at alignUp(metaLen, alignment)
                // GgufWriter already pads the meta section; the offsets in
                // TensorInfo are relative to after the padding, so we just
                // write tensors in offset order with per-tensor alignment.
                long dataBase = alignUp(metaLen, alignment);
                long padNeeded = dataBase - metaLen;
                for (long p = 0; p < padNeeded; p++) fos.write(0);
                written = dataBase;

                for (int i = 0; i < plan.size(); i++) {
                    TensorPlan tp = plan.get(i);
                    
                    if (i < 3 || i % 50 == 0) {
                        log(opts, String.format("  Converting tensor [%d/%d]: %s", 
                                i + 1, total, tp.ggufName()));
                    }
                    
                    byte[] converted = convertTensor(tp);
                    
                    if (converted.length == 0) {
                        log(opts, String.format("  WARNING: tensor %s converted to 0 bytes, skipping", 
                                tp.ggufName()));
                        continue;
                    }

                    fos.write(converted);
                    written += converted.length;

                    // Pad to alignment
                    long nextAligned = alignUp(written - dataBase, alignment) + dataBase;
                    long pad = nextAligned - written;
                    for (long p = 0; p < pad; p++) fos.write(0);
                    written = nextAligned;

                    if (opts.onProgress != null) opts.onProgress.accept(i + 1, total);
                    
                    if (i < 3 || i % 50 == 0) {
                        log(opts, String.format("  [%d/%d] wrote %s (%.1f KB, total: %.1f MB)",
                                i + 1, total, tp.ggufName(), 
                                converted.length / 1024.0, written / 1024.0 / 1024.0));
                    }
                }
            }
        } finally {
            Files.deleteIfExists(metaTemp);
        }
    }

    // ── Tensor conversion ─────────────────────────────────────────────────

    /** Open shard, copy tensor bytes via mmap, convert in-memory. */
    private static byte[] convertTensor(TensorPlan tp) throws IOException {
        try (Arena arena = Arena.ofConfined();
             FileChannel fc = FileChannel.open(tp.shard(), StandardOpenOption.READ)) {

            MemorySegment fileSeg = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size(), arena);

            // Re-read JSON header to find dataOffset within the shard
            long jsonLen = fileSeg.get(
                    ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 0);
            long dataOffset = 8 + jsonLen;

            long numElem = tp.numElements();
            
            // Validate tensor size before reading
            long srcBytesCount = tp.srcBytes();
            if (srcBytesCount == 0 && numElem > 0) {
                System.err.printf("[gguf] WARN: tensor %s has %d elements but 0 bytes, skipping%n",
                        tp.ggufName(), numElem);
                return new byte[0];
            }
            
            // Log large tensors
            if (srcBytesCount > 10_000_000) { // > 10MB
                System.out.printf("[gguf] Large tensor: %s, %.1f MB, %d elements%n",
                        tp.ggufName(), srcBytesCount / 1024.0 / 1024.0, numElem);
            }
            
            byte[] srcBytes = new byte[(int) srcBytesCount];
            try {
                MemorySegment.copy(fileSeg, ValueLayout.JAVA_BYTE,
                        dataOffset + tp.dataStart(), srcBytes, 0, srcBytes.length);
            } catch (Exception e) {
                System.err.printf("[gguf] ERROR: Failed to read tensor %s from file: %s%n",
                        tp.ggufName(), e.getMessage());
                throw new IOException("Failed to read tensor " + tp.ggufName(), e);
            }

            GgmlType srcType = ggmlTypeForDtype(tp.dtype());
            GgmlType dstType = tp.targetType();

            // Step 1: normalise to F32
            byte[] f32;
            try {
                f32 = switch (srcType) {
                    case F32  -> srcBytes;
                    case BF16 -> TensorConverter.bf16ToF32(srcBytes, numElem);
                    case F16  -> TensorConverter.f16ToF32(srcBytes, numElem);
                    case F64  -> f64ToF32(srcBytes, numElem);
                    default   -> srcBytes; // integer types passed through
                };
            } catch (Exception e) {
                System.err.printf("[gguf] ERROR: Failed to convert %s from %s to F32: %s%n",
                        tp.ggufName(), srcType.label, e.getMessage());
                throw new IOException("F32 conversion failed for " + tp.ggufName(), e);
            }
            
            // Validate F32 conversion
            long expectedF32Size = numElem * 4; // 4 bytes per F32
            if (f32.length != expectedF32Size) {
                System.err.printf("[gguf] ERROR: F32 size mismatch for %s: expected %d, got %d%n",
                        tp.ggufName(), expectedF32Size, f32.length);
            }

            // Step 2: quantize / re-encode
            byte[] result;
            try {
                result = switch (dstType) {
                    case F32  -> f32;
                    case F16  -> TensorConverter.f32ToF16(f32, numElem);
                    case BF16 -> TensorConverter.f32ToBf16(f32, numElem);
                    case Q8_0 -> TensorConverter.quantizeQ8_0(f32, numElem);
                    case Q4_0 -> TensorConverter.quantizeQ4_0(f32, numElem);
                    case Q2_K -> TensorConverter.quantizeQ2_K(f32, numElem);
                    case Q4_K -> TensorConverter.quantizeQ4_K(f32, numElem);
                    case Q5_K -> TensorConverter.quantizeQ5_K(f32, numElem);
                    case Q6_K -> TensorConverter.quantizeQ6_K(f32, numElem);
                    default   -> {
                        System.err.printf("[gguf] WARN: unsupported dst type %s for %s – keeping F32%n",
                                dstType.label, tp.ggufName());
                        yield f32;
                    }
                };
            } catch (Exception e) {
                System.err.printf("[gguf] ERROR: Failed to quantize %s to %s: %s%n",
                        tp.ggufName(), dstType.label, e.getMessage());
                e.printStackTrace();
                throw new IOException("Quantization failed for " + tp.ggufName(), e);
            }
            
            // Log first tensor to verify
            if (tp.ggufName().equals("token_embd.weight")) {
                System.out.printf("[gguf] Sample tensor: %s, src=%s, dst=%s, srcBytes=%d, dstBytes=%d, numElem=%d%n",
                        tp.ggufName(), srcType.label, dstType.label, srcBytes.length, result.length, numElem);
            }
            
            return result;
        }
    }

    // ── generation_config.json support ───────────────────────────────────

    /** Reads optional {@code generation_config.json} and adds pad/eos/bos ids if absent. */
    private static void applyGenerationConfig(Options opts, GgufModel model) {
        Path genCfg = opts.inputDir.resolve("generation_config.json");
        if (!Files.exists(genCfg)) return;
        try (Reader r = Files.newBufferedReader(genCfg)) {
            JsonObject obj = JsonParser.parseReader(r).getAsJsonObject();
            if (obj.has("pad_token_id") && !obj.get("pad_token_id").isJsonNull()) {
                long padId = obj.get("pad_token_id").getAsLong();
                model.getMeta("tokenizer.ggml.padding_token_id")
                     .ifPresentOrElse(
                             v -> { /* already set */ },
                             () -> model.addMeta("tokenizer.ggml.padding_token_id",
                                     GgufMetaValue.ofUInt32(padId)));
            }
            if (obj.has("chat_template") && !obj.get("chat_template").isJsonNull()) {
                String tmpl = obj.get("chat_template").getAsString();
                model.getMeta("tokenizer.chat_template")
                     .ifPresentOrElse(v -> {}, () ->
                             model.addMeta("tokenizer.chat_template",
                                     GgufMetaValue.ofString(tmpl)));
                log(opts, "Applied chat_template from generation_config.json");
            }
        } catch (IOException e) {
            log(opts, "WARN: could not read generation_config.json: " + e.getMessage());
        }
    }

    // ── Utilities ─────────────────────────────────────────────────────────

    private static HfConfigParser.TokenizerData tryParseTokenizer(
            Options opts, HfConfigParser.ModelConfig cfg) {
        try {
            HfConfigParser.TokenizerData tok = HfConfigParser.parseTokenizer(opts.inputDir);
            if (tok != null) log(opts, "Tokenizer model : " + tok.tokenizerModel());
            return tok;
        } catch (Exception e) {
            log(opts, "Tokenizer not found / unreadable: " + e.getMessage());
            return null;
        }
    }

    private static List<Path> findShards(Path dir) throws IOException {
        try (var stream = Files.list(dir)) {
            return stream
                    .filter(p -> p.toString().endsWith(".safetensors"))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    /** HF shape is [outermost, …, innermost]; GGUF needs [innermost, …, outermost]. */
    private static long[] reverseShape(long[] shape) {
        long[] rev = new long[shape.length];
        for (int i = 0; i < shape.length; i++) rev[i] = shape[shape.length - 1 - i];
        return rev;
    }

    private static long alignUp(long offset, long alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    private static GgmlType ggmlTypeForDtype(String dtype) {
        return switch (dtype.toUpperCase()) {
            case "F32"  -> GgmlType.F32;
            case "F16"  -> GgmlType.F16;
            case "BF16" -> GgmlType.BF16;
            case "F64"  -> GgmlType.F64;
            case "I8"   -> GgmlType.I8;
            case "I16"  -> GgmlType.I16;
            case "I32"  -> GgmlType.I32;
            case "I64"  -> GgmlType.I64;
            default     -> throw new UnsupportedOperationException("Unknown dtype: " + dtype);
        };
    }

    /** Convert F64 tensor bytes to F32 (truncates precision). */
    private static byte[] f64ToF32(byte[] src, long numElements) {
        byte[] dst = new byte[(int) (numElements * 4)];
        java.nio.ByteBuffer in  = java.nio.ByteBuffer.wrap(src)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        java.nio.ByteBuffer out = java.nio.ByteBuffer.wrap(dst)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (long i = 0; i < numElements; i++)
            out.putFloat((float) in.getDouble());
        return dst;
    }

    private static void log(Options opts, String msg) {
        if (opts.verbose || msg.startsWith("===") || msg.startsWith("Done"))
            System.out.println("[gguf] " + msg);
    }
}
