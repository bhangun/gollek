package tech.kayys.gollek.provider.litert;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.loader.SafetensorFFMLoader;
import tech.kayys.alkhawarizm.safetensor.loader.SafetensorLoadResult;
import tech.kayys.alkhawarizm.safetensor.loader.SafetensorTensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LiteRTGemmaDebugTest {

    @Test
    public void dumpSelectedWeightSizes() throws Exception {
        Path modelPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");
        Map<String, LiteRTContainerParser.WeightEntry> weights = LiteRTContainerParser.extractWeightMap(modelPath);

        dumpSize(weights, "transformer.embedder.input_embedding.w");
        dumpSize(weights, "transformer.embedder.input_embedding.w_quantized_scale");
        dumpSize(weights, "transformer.embedder.per_layer_model_projection.w");
        dumpSize(weights, "transformer.embedder.per_layer_model_projection.w_quantized_scale");
        dumpSize(weights, "transformer.embedder.per_layer_model_projection.w.sum_i");
        dumpSize(weights, "transformer.layer_0.attn.q.w");
        dumpSize(weights, "transformer.layer_0.attn.q.w_quantized_scale");
        dumpSize(weights, "transformer.layer_0.attn.q.w.sum_i");
        dumpSize(weights, "transformer.layer_0.attn.k.w");
        dumpSize(weights, "transformer.layer_0.attn.k.w_quantized_scale");
        dumpSize(weights, "transformer.layer_0.attn.k.w.sum_i");
        dumpSize(weights, "transformer.layer_0.attn.attn_vec_einsum.w");
        dumpSize(weights, "transformer.layer_0.attn.attn_vec_einsum.w_quantized_scale");
        dumpSize(weights, "transformer.layer_0.attn.attn_vec_einsum.w.sum_i");
        dumpSize(weights, "transformer.layer_0.mlp.ff_gate.w");
        dumpSize(weights, "transformer.layer_0.mlp.ff_gate.w_quantized_scale");
        dumpSize(weights, "transformer.layer_0.mlp.ff_gate.w.sum_i");
        dumpSize(weights, "transformer.layer_0.mlp.ff1.w");
        dumpSize(weights, "transformer.layer_0.mlp.ff1.w_quantized_scale");
        dumpSize(weights, "transformer.layer_0.mlp.ff1.w.sum_i");
        dumpSize(weights, "transformer.layer_0.mlp.linear.w");
        dumpSize(weights, "transformer.layer_0.mlp.linear.w_quantized_scale");
        dumpSize(weights, "transformer.layer_0.mlp.linear.w.sum_i");
        dumpSize(weights, "transformer.layer_0.skip.scale");
    }

    @Test
    public void dumpQuantMetadataSamples() throws Exception {
        Path modelPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");
        Map<String, LiteRTContainerParser.WeightEntry> weights = LiteRTContainerParser.extractWeightMap(modelPath);

        try (Arena arena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            dumpFloatSamples(channel, arena, weights, "transformer.layer_0.attn.q.w_quantized_scale");
            dumpIntSamples(channel, arena, weights, "transformer.layer_0.attn.q.w.sum_i");
            dumpFloatSamples(channel, arena, weights, "transformer.layer_0.mlp.ff_gate.w_quantized_scale");
            dumpIntSamples(channel, arena, weights, "transformer.layer_0.mlp.ff_gate.w.sum_i");
            comparePackedRowSums(channel, arena, weights, "transformer.layer_0.attn.q.w",
                    "transformer.layer_0.attn.q.w.sum_i", 1536);
        }
    }

    @Test
    public void compareTaskDecodeAgainstSafetensorRow() throws Exception {
        Path taskPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");
        Path safetensorPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "google", "gemma-4-E2B-it", "google__gemma-4-E2B-it", "model.safetensors");

        Map<String, LiteRTContainerParser.WeightEntry> taskWeights = LiteRTContainerParser.extractWeightMap(taskPath);

        try (Arena arena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(taskPath, StandardOpenOption.READ);
             SafetensorLoadResult result = loadSafetensorFile(safetensorPath)) {
            MemorySegment packed = map(channel, arena, taskWeights.get("transformer.layer_0.attn.q.w"));
            MemorySegment scale = map(channel, arena, taskWeights.get("transformer.layer_0.attn.q.w_quantized_scale"));

            String safetensorName = result.tensorNames().stream()
                    .filter(name -> name.contains("layers.0.self_attn.q_proj.weight"))
                    .findFirst()
                    .orElseThrow();
            System.out.println("Using safetensor tensor: " + safetensorName);
            SafetensorTensor tensor = result.tensor(safetensorName);
            float[] reference = tensor.toFloatArray();

            for (int row = 0; row < 4; row++) {
                float[] signedExt = decodeTaskRow(packed, scale, row, 1536, false);
                float[] centered = decodeTaskRow(packed, scale, row, 1536, true);
                float[] signedExtInputMajor = decodeTaskOutputRowInputMajor(packed, scale, row, 1536, 2048, false);
                float[] centeredInputMajor = decodeTaskOutputRowInputMajor(packed, scale, row, 1536, 2048, true);
                float[] centeredInputMajorSwapped = decodeTaskOutputRowInputMajorSwapped(
                        packed, scale, row, 1536, 2048, true);
                float[] refRow = sliceRow(reference, row, 1536);
                System.out.println("row " + row
                        + " signedExt cosine=" + cosineSimilarity(signedExt, refRow)
                        + " mse=" + mse(signedExt, refRow));
                System.out.println("row " + row
                        + " centered cosine=" + cosineSimilarity(centered, refRow)
                        + " mse=" + mse(centered, refRow));
                System.out.println("row " + row
                        + " signedExtInputMajor cosine=" + cosineSimilarity(signedExtInputMajor, refRow)
                        + " mse=" + mse(signedExtInputMajor, refRow));
                System.out.println("row " + row
                        + " centeredInputMajor cosine=" + cosineSimilarity(centeredInputMajor, refRow)
                        + " mse=" + mse(centeredInputMajor, refRow));
                System.out.println("row " + row
                        + " centeredInputMajorSwapped cosine=" + cosineSimilarity(centeredInputMajorSwapped, refRow)
                        + " mse=" + mse(centeredInputMajorSwapped, refRow));
            }
        }
    }

    @Test
    public void inspectQProjRowAlignment() throws Exception {
        Path taskPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");
        Path safetensorPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "google", "gemma-4-E2B-it", "google__gemma-4-E2B-it", "model.safetensors");

        Map<String, LiteRTContainerParser.WeightEntry> taskWeights = LiteRTContainerParser.extractWeightMap(taskPath);

        try (Arena arena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(taskPath, StandardOpenOption.READ);
             SafetensorLoadResult result = loadSafetensorFile(safetensorPath)) {
            MemorySegment packed = map(channel, arena, taskWeights.get("transformer.layer_0.attn.q.w"));
            MemorySegment scale = map(channel, arena, taskWeights.get("transformer.layer_0.attn.q.w_quantized_scale"));

            String safetensorName = result.tensorNames().stream()
                    .filter(name -> name.contains("layers.0.self_attn.q_proj.weight"))
                    .findFirst()
                    .orElseThrow();
            SafetensorTensor tensor = result.tensor(safetensorName);
            System.out.println("Q projection shape: " + java.util.Arrays.toString(tensor.shape()));
            float[] reference = tensor.toFloatArray();
            int outDim = (int) tensor.shape()[0];
            int inDim = (int) tensor.shape()[1];

            for (int row = 0; row < 4; row++) {
                float[] taskRow = decodeTaskOutputRowInputMajor(packed, scale, row, inDim, outDim, true);
                Match best = bestRowMatch(taskRow, reference, outDim, inDim);
                double sameRow = cosineSimilarity(taskRow, sliceRow(reference, row, inDim));
                System.out.println("row " + row
                        + " bestRefRow=" + best.index
                        + " bestCosine=" + best.cosine
                        + " sameRowCosine=" + sameRow);
            }
        }
    }

    @Test
    public void compareInputEmbeddingDecodeAgainstSafetensorRow() throws Exception {
        Path taskPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");
        Path safetensorPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "google", "gemma-4-E2B-it", "google__gemma-4-E2B-it", "model.safetensors");

        Map<String, LiteRTContainerParser.WeightEntry> taskWeights = LiteRTContainerParser.extractWeightMap(taskPath);

        try (Arena arena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(taskPath, StandardOpenOption.READ);
             SafetensorLoadResult result = loadSafetensorFile(safetensorPath)) {
            MemorySegment packed = map(channel, arena, taskWeights.get("transformer.embedder.input_embedding.w"));
            MemorySegment scale = map(channel, arena, taskWeights.get("transformer.embedder.input_embedding.w_quantized_scale"));

            String safetensorName = result.tensorNames().stream()
                    .filter(name -> name.contains("model.language_model.embed_tokens.weight"))
                    .findFirst()
                    .orElseThrow();
            SafetensorTensor tensor = result.tensor(safetensorName);
            System.out.println("Input embedding shape: " + java.util.Arrays.toString(tensor.shape()));
            float[] reference = tensor.toFloatArray();
            int vocabSize = (int) tensor.shape()[0];
            int dim = (int) tensor.shape()[1];

            for (int tokenId : new int[] { 0, 1, 42, 506 }) {
                float[] refRow = sliceRow(reference, tokenId, dim);
                float[] centeredTwoBit = decodePacked2BitEmbeddingRow(packed, scale, tokenId, dim, TwoBitMode.CENTERED_HALF);
                float[] signedTwoBit = decodePacked2BitEmbeddingRow(packed, scale, tokenId, dim, TwoBitMode.SIGNED_OFFSET_TWO);
                float[] unsignedTwoBit = decodePacked2BitEmbeddingRow(packed, scale, tokenId, dim, TwoBitMode.UNSIGNED);
                float[] reversedCentered = decodePacked2BitEmbeddingRow(packed, scale, tokenId, dim, TwoBitMode.CENTERED_HALF_REVERSED);
                float[] inputMajorCentered = decodePacked2BitEmbeddingRowInputMajor(
                        packed, scale, tokenId, dim, vocabSize, TwoBitMode.CENTERED_HALF);
                float[] inputMajorReversed = decodePacked2BitEmbeddingRowInputMajor(
                        packed, scale, tokenId, dim, vocabSize, TwoBitMode.CENTERED_HALF_REVERSED);
                System.out.println("token " + tokenId
                        + " centeredHalf cosine=" + cosineSimilarity(centeredTwoBit, refRow)
                        + " mse=" + mse(centeredTwoBit, refRow));
                System.out.println("token " + tokenId
                        + " signedOffsetTwo cosine=" + cosineSimilarity(signedTwoBit, refRow)
                        + " mse=" + mse(signedTwoBit, refRow));
                System.out.println("token " + tokenId
                        + " unsigned cosine=" + cosineSimilarity(unsignedTwoBit, refRow)
                        + " mse=" + mse(unsignedTwoBit, refRow));
                System.out.println("token " + tokenId
                        + " reversedCentered cosine=" + cosineSimilarity(reversedCentered, refRow)
                        + " mse=" + mse(reversedCentered, refRow));
                System.out.println("token " + tokenId
                        + " inputMajorCentered cosine=" + cosineSimilarity(inputMajorCentered, refRow)
                        + " mse=" + mse(inputMajorCentered, refRow));
                System.out.println("token " + tokenId
                        + " inputMajorReversed cosine=" + cosineSimilarity(inputMajorReversed, refRow)
                        + " mse=" + mse(inputMajorReversed, refRow));
            }
            System.out.println("Input embedding vocab size: " + vocabSize);
        }
    }

    @Test
    public void inferInputEmbeddingCodebook() throws Exception {
        Path taskPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");
        Path safetensorPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "google", "gemma-4-E2B-it", "google__gemma-4-E2B-it", "model.safetensors");

        Map<String, LiteRTContainerParser.WeightEntry> taskWeights = LiteRTContainerParser.extractWeightMap(taskPath);

        try (Arena arena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(taskPath, StandardOpenOption.READ);
             SafetensorLoadResult result = loadSafetensorFile(safetensorPath)) {
            MemorySegment packed = map(channel, arena, taskWeights.get("transformer.embedder.input_embedding.w"));
            MemorySegment scale = map(channel, arena, taskWeights.get("transformer.embedder.input_embedding.w_quantized_scale"));

            String safetensorName = result.tensorNames().stream()
                    .filter(name -> name.contains("model.language_model.embed_tokens.weight"))
                    .findFirst()
                    .orElseThrow();
            SafetensorTensor tensor = result.tensor(safetensorName);
            float[] reference = tensor.toFloatArray();
            int vocabSize = (int) tensor.shape()[0];
            int dim = (int) tensor.shape()[1];

            for (boolean inputMajor : new boolean[] { false, true }) {
                double[] sums = new double[4];
                int[] counts = new int[4];
                for (int tokenId : new int[] { 0, 1, 42, 506 }) {
                    float rowScale = scale.getAtIndex(ValueLayout.JAVA_FLOAT, tokenId);
                    float[] refRow = sliceRow(reference, tokenId, dim);
                    for (int i = 0; i < dim; i++) {
                        int q = inputMajor
                                ? read2BitInputMajor(packed, tokenId, i, vocabSize)
                                : read2BitRowMajor(packed, tokenId, i, dim);
                        sums[q] += refRow[i] / rowScale;
                        counts[q]++;
                    }
                }
                System.out.println((inputMajor ? "inputMajor" : "rowMajor") + " inferred codebook:");
                for (int q = 0; q < 4; q++) {
                    System.out.println("  q=" + q + " mean=" + (sums[q] / Math.max(1, counts[q])) + " count=" + counts[q]);
                }
            }
        }
    }

    @Test
    public void scanBlockedInputEmbeddingLayouts() throws Exception {
        Path taskPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");
        Path safetensorPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "google", "gemma-4-E2B-it", "google__gemma-4-E2B-it", "model.safetensors");

        Map<String, LiteRTContainerParser.WeightEntry> taskWeights = LiteRTContainerParser.extractWeightMap(taskPath);

        try (Arena arena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(taskPath, StandardOpenOption.READ);
             SafetensorLoadResult result = loadSafetensorFile(safetensorPath)) {
            MemorySegment packed = map(channel, arena, taskWeights.get("transformer.embedder.input_embedding.w"));
            MemorySegment scale = map(channel, arena, taskWeights.get("transformer.embedder.input_embedding.w_quantized_scale"));

            String safetensorName = result.tensorNames().stream()
                    .filter(name -> name.contains("model.language_model.embed_tokens.weight"))
                    .findFirst()
                    .orElseThrow();
            SafetensorTensor tensor = result.tensor(safetensorName);
            float[] reference = tensor.toFloatArray();
            int vocabSize = (int) tensor.shape()[0];
            int dim = (int) tensor.shape()[1];
            int[] probeTokens = new int[] { 0, 1, 42, 506 };

            for (int blockSize : new int[] { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 }) {
                if (vocabSize % blockSize != 0) {
                    continue;
                }
                double avgCosine = 0.0;
                double avgMse = 0.0;
                for (int tokenId : probeTokens) {
                    float[] refRow = sliceRow(reference, tokenId, dim);
                    float[] candidate = decodePacked2BitEmbeddingRowBlocked(
                            packed, scale, tokenId, dim, vocabSize, blockSize, TwoBitMode.CENTERED_HALF);
                    avgCosine += cosineSimilarity(candidate, refRow);
                    avgMse += mse(candidate, refRow);
                }
                avgCosine /= probeTokens.length;
                avgMse /= probeTokens.length;
                System.out.println("blockSize=" + blockSize
                        + " avgCosine=" + avgCosine
                        + " avgMse=" + avgMse);
            }
        }
    }

    @Test
    public void compareManualEmbeddingToNativeEmbedder() throws Exception {
        Path taskPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");

        Map<String, LiteRTContainerParser.WeightEntry> taskWeights = LiteRTContainerParser.extractWeightMap(taskPath);

        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);
        try (Arena arena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(taskPath, StandardOpenOption.READ);
             LiteRTTokenizer tokenizer = LiteRTTokenizer.create(taskPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            List<Long> offsets = LiteRTContainerParser.findTfl3SegmentsForInspection(litertlmPath);
            long embedderOffset = offsets.get(0);
            long embedderSize = offsets.get(1) - offsets.get(0);

            Arena sharedArena = (Arena) getField(nativeRunner, "arena");
            MemorySegment environment = bindings.createEnvironment(sharedArena);
            MemorySegment options = bindings.createOptions(sharedArena);
            bindings.setOptionsHardwareAccelerators(options, LiteRTNativeBindings.kLiteRtHwAcceleratorCpu);
            setField(nativeRunner, "environment", environment);
            setField(nativeRunner, "options", options);

            Method loadCompiledSegment = LiteRTGemmaNativeRunner.class.getDeclaredMethod(
                    "loadCompiledSegment", String.class, long.class, long.class);
            loadCompiledSegment.setAccessible(true);
            Object embedderSegment = loadCompiledSegment.invoke(nativeRunner, "raw-embedder", embedderOffset, embedderSize);

            @SuppressWarnings("unchecked")
            Map<String, Object> signatures = (Map<String, Object>) getField(embedderSegment, "signatures");
            Object embedderSignature = signatures.get("embedder");
            @SuppressWarnings("unchecked")
            List<Object> outputSpecs = (List<Object>) getField(embedderSignature, "outputs");
            for (Object outputSpec : outputSpecs) {
                System.out.println("embedder output spec name=" + getField(outputSpec, "name")
                        + " typeId=" + getField(outputSpec, "typeId")
                        + " dims=" + Arrays.toString((int[]) getField(outputSpec, "dims"))
                        + " reqBytes=" + getField(outputSpec, "reqBytes"));
            }
            Method runMethod = embedderSegment.getClass().getDeclaredMethod(
                    "run",
                    LiteRTNativeBindings.class,
                    MemorySegment.class,
                    byte[].class,
                    embedderSignature.getClass());
            runMethod.setAccessible(true);

            MemorySegment packed = map(channel, arena, taskWeights.get("transformer.embedder.input_embedding.w"));
            MemorySegment scale = map(channel, arena, taskWeights.get("transformer.embedder.input_embedding.w_quantized_scale"));

            for (int tokenId : new int[] { 0, 1, 42, 506 }) {
                @SuppressWarnings("unchecked")
                Map<String, byte[]> outputs = (Map<String, byte[]>) runMethod.invoke(
                        embedderSegment, bindings, environment, encodeInt32(tokenId), embedderSignature);
                byte[] embeddingBytes = outputs.get("embeddings");
                float[] nativeEmbedding = decodeNativeFloatTensor(embeddingBytes);
                float[] rowMajorCentered = decodePacked2BitEmbeddingRow(packed, scale, tokenId, nativeEmbedding.length, TwoBitMode.CENTERED_HALF);
                float[] inputMajorCentered = decodePacked2BitEmbeddingRowInputMajor(
                        packed, scale, tokenId, nativeEmbedding.length,
                        (int) (scale.byteSize() / Float.BYTES),
                        TwoBitMode.CENTERED_HALF);
                System.out.println("token " + tokenId
                        + " manualRowMajorVsNative cosine=" + cosineSimilarity(rowMajorCentered, nativeEmbedding)
                        + " mse=" + mse(rowMajorCentered, nativeEmbedding));
                System.out.println("token " + tokenId
                        + " manualInputMajorVsNative cosine=" + cosineSimilarity(inputMajorCentered, nativeEmbedding)
                        + " mse=" + mse(inputMajorCentered, nativeEmbedding));
            }
        }
    }

    @Test
    public void inspectRawLitertlmSegments() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            Method inspectSegment = LiteRTGemmaNativeRunner.class.getDeclaredMethod(
                    "inspectSegment", String.class, long.class, long.class);
            inspectSegment.setAccessible(true);
            List<Long> offsets = LiteRTContainerParser.findTfl3SegmentsForInspection(litertlmPath);
            long fileSize = Files.size(litertlmPath);
            for (int i = 0; i < offsets.size(); i++) {
                long off = offsets.get(i);
                long next = i + 1 < offsets.size() ? offsets.get(i + 1) : fileSize;
                long size = next - off;
                Object candidate = inspectSegment.invoke(nativeRunner, "raw-" + i, off, size);
                @SuppressWarnings("unchecked")
                java.util.Set<String> signatures = (java.util.Set<String>) getField(candidate, "signatureKeys");
                System.out.println("raw segment " + i + " off=0x" + Long.toHexString(off)
                        + " size=" + size + " signatures=" + signatures);
            }
        }
    }

    @Test
    public void inspectRawDecodeSignatureSpecs() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            List<Long> offsets = LiteRTContainerParser.findTfl3SegmentsForInspection(litertlmPath);
            long fileSize = Files.size(litertlmPath);
            Arena sharedArena = (Arena) getField(nativeRunner, "arena");
            MemorySegment environment = bindings.createEnvironment(sharedArena);
            MemorySegment options = bindings.createOptions(sharedArena);
            bindings.setOptionsHardwareAccelerators(options, LiteRTNativeBindings.kLiteRtHwAcceleratorCpu);
            setField(nativeRunner, "environment", environment);
            setField(nativeRunner, "options", options);

            Method loadCompiledSegment = LiteRTGemmaNativeRunner.class.getDeclaredMethod(
                    "loadCompiledSegment", String.class, long.class, long.class);
            loadCompiledSegment.setAccessible(true);

            int[] interestingSegments = new int[] { 0, 1, 8 };
            for (int index : interestingSegments) {
                long off = offsets.get(index);
                long next = index + 1 < offsets.size() ? offsets.get(index + 1) : fileSize;
                Object segment = loadCompiledSegment.invoke(nativeRunner, "raw-" + index, off, next - off);
                @SuppressWarnings("unchecked")
                Map<String, Object> signatures = (Map<String, Object>) getField(segment, "signatures");
                System.out.println("segment " + index + " off=0x" + Long.toHexString(off));
                for (Map.Entry<String, Object> entry : signatures.entrySet()) {
                    System.out.println("  signature " + entry.getKey());
                    dumpSignatureSpec(entry.getValue());
                }
            }
        }
    }

    @Test
    public void smokeRawDecodeWithZeroParamTensor() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            List<Long> offsets = LiteRTContainerParser.findTfl3SegmentsForInspection(litertlmPath);
            long fileSize = Files.size(litertlmPath);
            Arena sharedArena = (Arena) getField(nativeRunner, "arena");
            MemorySegment environment = bindings.createEnvironment(sharedArena);
            MemorySegment options = bindings.createOptions(sharedArena);
            bindings.setOptionsHardwareAccelerators(options, LiteRTNativeBindings.kLiteRtHwAcceleratorCpu);
            setField(nativeRunner, "environment", environment);
            setField(nativeRunner, "options", options);

            Method loadCompiledSegment = LiteRTGemmaNativeRunner.class.getDeclaredMethod(
                    "loadCompiledSegment", String.class, long.class, long.class);
            loadCompiledSegment.setAccessible(true);

            Object embedderSegment = loadCompiledSegment.invoke(nativeRunner, "raw-embedder",
                    offsets.get(0), offsets.get(1) - offsets.get(0));
            Object perLayerSegment = loadCompiledSegment.invoke(nativeRunner, "raw-per-layer",
                    offsets.get(1), offsets.get(2) - offsets.get(1));
            Object decodeSegment = loadCompiledSegment.invoke(nativeRunner, "raw-decode",
                    offsets.get(8), offsets.get(9) - offsets.get(8));

            @SuppressWarnings("unchecked")
            Object embedderSignature = ((Map<String, Object>) getField(embedderSegment, "signatures")).get("embedder");
            @SuppressWarnings("unchecked")
            Object perLayerSignature = ((Map<String, Object>) getField(perLayerSegment, "signatures")).get("per_layer_embedder");
            @SuppressWarnings("unchecked")
            Object decodeSignature = ((Map<String, Object>) getField(decodeSegment, "signatures")).get("decode");

            Method runEmbedder = embedderSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, byte[].class, embedderSignature.getClass());
            Method runPerLayer = perLayerSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, byte[].class, perLayerSignature.getClass());
            Method runDecode = decodeSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, Map.class, decodeSignature.getClass());
            runEmbedder.setAccessible(true);
            runPerLayer.setAccessible(true);
            runDecode.setAccessible(true);

            int tokenId = LiteRTTokenizer.BOS_TOKEN_ID;
            @SuppressWarnings("unchecked")
            Map<String, byte[]> embedderOutputs = (Map<String, byte[]>) runEmbedder.invoke(
                    embedderSegment, bindings, environment, encodeInt32(tokenId), embedderSignature);
            @SuppressWarnings("unchecked")
            Map<String, byte[]> perLayerOutputs = (Map<String, byte[]>) runPerLayer.invoke(
                    perLayerSegment, bindings, environment, encodeInt32(tokenId), perLayerSignature);

            @SuppressWarnings("unchecked")
            List<Object> decodeInputs = (List<Object>) getField(decodeSignature, "inputs");
            Map<String, byte[]> inputs = new java.util.HashMap<>();
            for (Object inputSpec : decodeInputs) {
                String name = (String) getField(inputSpec, "name");
                int reqBytes = (Integer) getField(inputSpec, "reqBytes");
                switch (name) {
                    case "embeddings" -> inputs.put(name, embedderOutputs.get("embeddings"));
                    case "per_layer_embeddings" -> inputs.put(name, perLayerOutputs.get("embeddings"));
                    case "input_pos" -> inputs.put(name, encodeInt32(0));
                    case "mask" -> inputs.put(name, buildRawMask(reqBytes));
                    case "param_tensor" -> inputs.put(name, new byte[reqBytes]);
                    default -> {
                        if (name.startsWith("kv_cache_")) {
                            inputs.put(name, new byte[reqBytes]);
                        } else {
                            throw new IllegalStateException("Unhandled decode input: " + name);
                        }
                    }
                }
            }

            @SuppressWarnings("unchecked")
            Map<String, byte[]> decodeOutputs = (Map<String, byte[]>) runDecode.invoke(
                    decodeSegment, bindings, environment, inputs, decodeSignature);
            float[] logits = littleEndianFloats(decodeOutputs.get("logits"));
            int[] top = topKIndices(logits, 5);
            System.out.println("Top logits with zero param tensor:");
            for (int id : top) {
                System.out.println("  id=" + id + " logit=" + logits[id] + " text=[" + tokenizer.decodeToken(id) + "]");
            }
        }
    }

    @Test
    public void smokeRawDecodePromptLoopWithZeroParamTensor() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            List<Long> offsets = LiteRTContainerParser.findTfl3SegmentsForInspection(litertlmPath);
            Arena sharedArena = (Arena) getField(nativeRunner, "arena");
            MemorySegment environment = bindings.createEnvironment(sharedArena);
            MemorySegment options = bindings.createOptions(sharedArena);
            bindings.setOptionsHardwareAccelerators(options, LiteRTNativeBindings.kLiteRtHwAcceleratorCpu);
            setField(nativeRunner, "environment", environment);
            setField(nativeRunner, "options", options);

            Method loadCompiledSegment = LiteRTGemmaNativeRunner.class.getDeclaredMethod(
                    "loadCompiledSegment", String.class, long.class, long.class);
            loadCompiledSegment.setAccessible(true);

            Object embedderSegment = loadCompiledSegment.invoke(nativeRunner, "raw-embedder",
                    offsets.get(0), offsets.get(1) - offsets.get(0));
            Object perLayerSegment = loadCompiledSegment.invoke(nativeRunner, "raw-per-layer",
                    offsets.get(1), offsets.get(2) - offsets.get(1));
            Object decodeSegment = loadCompiledSegment.invoke(nativeRunner, "raw-decode",
                    offsets.get(8), offsets.get(9) - offsets.get(8));

            @SuppressWarnings("unchecked")
            Object embedderSignature = ((Map<String, Object>) getField(embedderSegment, "signatures")).get("embedder");
            @SuppressWarnings("unchecked")
            Object perLayerSignature = ((Map<String, Object>) getField(perLayerSegment, "signatures")).get("per_layer_embedder");
            @SuppressWarnings("unchecked")
            Object decodeSignature = ((Map<String, Object>) getField(decodeSegment, "signatures")).get("decode");

            Method runEmbedder = embedderSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, byte[].class, embedderSignature.getClass());
            Method runPerLayer = perLayerSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, byte[].class, perLayerSignature.getClass());
            Method runDecode = decodeSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, Map.class, decodeSignature.getClass());
            runEmbedder.setAccessible(true);
            runPerLayer.setAccessible(true);
            runDecode.setAccessible(true);

            @SuppressWarnings("unchecked")
            List<Object> decodeInputs = (List<Object>) getField(decodeSignature, "inputs");
            Map<String, byte[]> state = new java.util.HashMap<>();
            for (Object inputSpec : decodeInputs) {
                String name = (String) getField(inputSpec, "name");
                int reqBytes = (Integer) getField(inputSpec, "reqBytes");
                if (name.startsWith("kv_cache_")) {
                    state.put(name, new byte[reqBytes]);
                }
            }

            int[] prompt = tokenizer.encodeChatPrompt("where is jakarta");
            byte[] lastLogits = null;
            for (int position = 0; position < prompt.length; position++) {
                int tokenId = prompt[position];
                @SuppressWarnings("unchecked")
                Map<String, byte[]> embedderOutputs = (Map<String, byte[]>) runEmbedder.invoke(
                        embedderSegment, bindings, environment, encodeInt32(tokenId), embedderSignature);
                @SuppressWarnings("unchecked")
                Map<String, byte[]> perLayerOutputs = (Map<String, byte[]>) runPerLayer.invoke(
                        perLayerSegment, bindings, environment, encodeInt32(tokenId), perLayerSignature);

                Map<String, byte[]> inputs = new java.util.HashMap<>(state);
                inputs.put("embeddings", embedderOutputs.get("embeddings"));
                inputs.put("per_layer_embeddings", perLayerOutputs.get("embeddings"));
                inputs.put("input_pos", encodeInt32(position));
                inputs.put("mask", buildCausalRawMask((Integer) getField(findInputSpec(decodeInputs, "mask"), "reqBytes"), position));
                inputs.put("param_tensor", new byte[(Integer) getField(findInputSpec(decodeInputs, "param_tensor"), "reqBytes")]);

                @SuppressWarnings("unchecked")
                Map<String, byte[]> outputs = (Map<String, byte[]>) runDecode.invoke(
                        decodeSegment, bindings, environment, inputs, decodeSignature);
                for (Map.Entry<String, byte[]> entry : outputs.entrySet()) {
                    if (entry.getKey().startsWith("kv_cache_")) {
                        state.put(entry.getKey(), entry.getValue());
                    }
                }
                lastLogits = outputs.get("logits");
            }

            float[] logits = littleEndianFloats(lastLogits);
            int[] top = topKIndices(logits, 5);
            System.out.println("Prompt-loop top logits:");
            for (int id : top) {
                System.out.println("  id=" + id + " logit=" + logits[id] + " text=[" + tokenizer.decodeToken(id) + "]");
            }
        }
    }

    @Test
    @Disabled("Experimental raw LiteRT-LM path is disabled by default until prompt-echo behavior is fixed")
    public void smokeNativeRunnerGenerate() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            nativeRunner.initialize();

            StringBuilder out = new StringBuilder();
            nativeRunner.generate(tokenizer.encodeChatPrompt("where is jakarta"),
                    1,
                    0.0d,
                    1,
                    1.0d,
                    1.0d,
                    out::append);
            System.out.println("native runner output=[" + out + "]");
        }
    }

    @Test
    @Disabled("Experimental raw LiteRT-LM path is disabled by default until prompt-echo behavior is fixed")
    public void smokeNativeRunnerGenerateTenTokensCpu() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            long startedAt = System.nanoTime();
            nativeRunner.initialize();

            StringBuilder out = new StringBuilder();
            nativeRunner.generate(tokenizer.encodeChatPrompt("where is jakarta"),
                    10,
                    0.0d,
                    1,
                    1.0d,
                    1.0d,
                    out::append);
            long elapsedMs = (System.nanoTime() - startedAt) / 1_000_000L;
            System.out.println("native runner 10-token output=[" + out + "] elapsedMs=" + elapsedMs);
        }
    }

    @Test
    @Disabled("Experimental raw LiteRT-LM path is disabled by default until prompt-echo behavior is fixed")
    public void nativeRunnerGenerateTenTokensCpuMentionsJakarta() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            nativeRunner.initialize();

            StringBuilder out = new StringBuilder();
            nativeRunner.generate(tokenizer.encodeChatPrompt("where is jakarta"),
                    10,
                    0.0d,
                    1,
                    1.0d,
                    1.0d,
                    out::append);
            String normalized = out.toString().toLowerCase(Locale.ROOT);
            assertTrue(normalized.contains("jakarta"),
                    "Expected native runner output to mention Jakarta, got: " + out);
            assertTrue(normalized.contains("capital"),
                    "Expected native runner output to answer with a capital-city fact, got: " + out);
            assertTrue(normalized.contains("indonesia"),
                    "Expected native runner output to mention Indonesia, got: " + out);
        }
    }

    @Test
    public void nativeRunnerRejectsRawLiteRtLmWhenDisabled() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        System.setProperty(LiteRTGemmaNativeRunner.DISABLE_RAW_LITERTLM_PROPERTY, "true");
        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            RuntimeException thrown = assertThrows(RuntimeException.class, nativeRunner::initialize);

            assertTrue(thrown.getMessage().contains("Raw Gemma LiteRT-LM signatures were disabled"),
                     "Expected raw LiteRT-LM guard message, got: " + thrown.getMessage());
        } finally {
            System.clearProperty(LiteRTGemmaNativeRunner.DISABLE_RAW_LITERTLM_PROPERTY);
        }
    }

    @Test
    public void gemma4TaskRunnerRejectsSlowRepeatedTokenPathByDefault() throws Exception {
        Path taskPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(taskPath);
             LiteRTInferenceRunner runner = new LiteRTInferenceRunner(null, taskPath, tokenizer, true, 4)) {
            runner.initialize();
            IllegalStateException thrown = assertThrows(
                    IllegalStateException.class,
                    () -> runner.generate("where is jakarta", ignored -> { }));

            assertTrue(thrown.getMessage().contains("Gemma 4 LiteRT .task runner is disabled by default"),
                    "Expected Gemma 4 task guard message, got: " + thrown.getMessage());
        }
    }

    @Test
    @Disabled("Diagnostic probe for the experimental raw LiteRT-LM path")
    public void nativeRunnerGenerateTenTokensCpuPlainPromptMentionsJakarta() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            nativeRunner.initialize();

            StringBuilder out = new StringBuilder();
            nativeRunner.generate(tokenizer.encodeWithBos("where is jakarta"),
                    10,
                    0.0d,
                    1,
                    1.0d,
                    1.0d,
                    out::append);
            String normalized = out.toString().toLowerCase(Locale.ROOT);
            System.out.println("native runner plain-prompt 10-token output=[" + out + "]");
            assertTrue(normalized.contains("jakarta"),
                    "Expected native runner plain-prompt output to mention Jakarta, got: " + out);
        }
    }

    @Test
    @Disabled("Diagnostic probe for the experimental raw LiteRT-LM path")
    public void exploreNativeRunnerPromptVariantsForJakarta() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            nativeRunner.initialize();

            List<String> prompts = List.of(
                    "where is jakarta",
                    "where is jakarta?",
                    "Where is Jakarta?",
                    "Question: where is jakarta\nAnswer:",
                    "Question: Where is Jakarta?\nAnswer:");
            double[] penalties = new double[] { 1.0d, 1.05d, 1.1d };

            for (String prompt : prompts) {
                for (double repeatPenalty : penalties) {
                    StringBuilder out = new StringBuilder();
                    nativeRunner.generate(tokenizer.encodeWithBos(prompt),
                            10,
                            0.0d,
                            1,
                            1.0d,
                            repeatPenalty,
                            out::append);
                    System.out.println("plain variant prompt=[" + prompt.replace("\n", "\\n")
                            + "] repeatPenalty=" + repeatPenalty + " output=[" + out + "]");
                }
            }
        }
    }

    @Test
    @Disabled("Diagnostic probe for the experimental raw LiteRT-LM path")
    public void explorePromptLoopPositionOffsetsForJakarta() throws Exception {
        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm"))) {
            for (int offset : new int[] { -1, 0, 1, 2 }) {
                int[] chatTop = promptLoopTopKForTokens(
                        tokenizer.encodeChatPrompt("where is jakarta"), offset, 5);
                int[] plainTop = promptLoopTopKForTokens(
                        tokenizer.encodeWithBos("where is jakarta"), offset, 5);
                System.out.println("positionOffset=" + offset
                        + " chatTop=" + describeTokens(tokenizer, chatTop)
                        + " plainTop=" + describeTokens(tokenizer, plainTop));
            }
        }
    }

    @Test
    @Disabled("Diagnostic probe for the experimental raw LiteRT-LM path")
    public void prefill128AbsoluteKeepsPromptLoopTopTokenInTopThree() throws Exception {
        int[] promptLoopTop = promptLoopTopKForPrompt("where is jakarta", 3);
        int[] prefillTop = prefill128TopKForPrompt("where is jakarta", "absolute", 3);

        assertEquals(174187, promptLoopTop[0], "Prompt loop top token changed unexpectedly");
        assertTrue(prefillTop[0] == promptLoopTop[0]
                        || prefillTop[1] == promptLoopTop[0]
                        || prefillTop[2] == promptLoopTop[0],
                "Absolute prefill should keep the prompt-loop top token in the top three");
    }

    @Test
    public void probeRawPrefill128MaskLayouts() throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            List<Long> offsets = LiteRTContainerParser.findTfl3SegmentsForInspection(litertlmPath);
            Arena sharedArena = (Arena) getField(nativeRunner, "arena");
            MemorySegment environment = bindings.createEnvironment(sharedArena);
            MemorySegment options = bindings.createOptions(sharedArena);
            bindings.setOptionsHardwareAccelerators(options, LiteRTNativeBindings.kLiteRtHwAcceleratorCpu);
            setField(nativeRunner, "environment", environment);
            setField(nativeRunner, "options", options);

            Method loadCompiledSegment = LiteRTGemmaNativeRunner.class.getDeclaredMethod(
                    "loadCompiledSegment", String.class, long.class, long.class);
            loadCompiledSegment.setAccessible(true);

            Object embedderSegment = loadCompiledSegment.invoke(nativeRunner, "raw-embedder",
                    offsets.get(0), offsets.get(1) - offsets.get(0));
            Object perLayerSegment = loadCompiledSegment.invoke(nativeRunner, "raw-per-layer",
                    offsets.get(1), offsets.get(2) - offsets.get(1));
            Object decodeSegment = loadCompiledSegment.invoke(nativeRunner, "raw-decode",
                    offsets.get(8), offsets.get(9) - offsets.get(8));

            @SuppressWarnings("unchecked")
            Object embedderSignature = ((Map<String, Object>) getField(embedderSegment, "signatures")).get("embedder");
            @SuppressWarnings("unchecked")
            Object perLayerSignature = ((Map<String, Object>) getField(perLayerSegment, "signatures")).get("per_layer_embedder");
            @SuppressWarnings("unchecked")
            Map<String, Object> decodeSignatures = (Map<String, Object>) getField(decodeSegment, "signatures");
            Object decodeSignature = decodeSignatures.get("decode");
            Object prefillSignature = decodeSignatures.get("prefill_128");

            Method runEmbedder = embedderSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, byte[].class, embedderSignature.getClass());
            Method runPerLayer = perLayerSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, byte[].class, perLayerSignature.getClass());
            Method runModel = decodeSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, Map.class, decodeSignature.getClass());
            runEmbedder.setAccessible(true);
            runPerLayer.setAccessible(true);
            runModel.setAccessible(true);

            @SuppressWarnings("unchecked")
            List<Object> decodeInputs = (List<Object>) getField(decodeSignature, "inputs");
            @SuppressWarnings("unchecked")
            List<Object> prefillInputs = (List<Object>) getField(prefillSignature, "inputs");

            int[] prompt = tokenizer.encodeChatPrompt("where is jakarta");
            int prefillCount = prompt.length - 1;
            int lastToken = prompt[prompt.length - 1];
            int lastPosition = prefillCount;

            byte[] zeroParam = new byte[(Integer) getField(findInputSpec(prefillInputs, "param_tensor"), "reqBytes")];
            int prefillMaskBytes = (Integer) getField(findInputSpec(prefillInputs, "mask"), "reqBytes");
            int prefillRows = ((int[]) getField(findInputSpec(prefillInputs, "mask"), "dims"))[2];
            int prefillCols = ((int[]) getField(findInputSpec(prefillInputs, "mask"), "dims"))[3];

            @SuppressWarnings("unchecked")
            Map<String, byte[]> prefillEmbeddings = packNativeEmbeddings(
                    prompt, prefillCount, runEmbedder, runPerLayer,
                    embedderSegment, perLayerSegment, embedderSignature, perLayerSignature, bindings, environment,
                    findInputSpec(prefillInputs, "embeddings"),
                    findInputSpec(prefillInputs, "per_layer_embeddings"));

            for (String maskMode : List.of("absolute", "tail", "past_plus_tail")) {
                Map<String, byte[]> state = new HashMap<>();
                for (Object inputSpec : prefillInputs) {
                    String name = (String) getField(inputSpec, "name");
                    int reqBytes = (Integer) getField(inputSpec, "reqBytes");
                    if (name.startsWith("kv_cache_")) {
                        state.put(name, new byte[reqBytes]);
                    }
                }

                Map<String, byte[]> prefillInputsMap = new HashMap<>(state);
                prefillInputsMap.put("embeddings", prefillEmbeddings.get("embeddings"));
                prefillInputsMap.put("per_layer_embeddings", prefillEmbeddings.get("per_layer_embeddings"));
                prefillInputsMap.put("input_pos", encodeIntRange(prefillRows, prefillCount, 0));
                prefillInputsMap.put("mask", buildPrefillMask(maskMode, prefillMaskBytes, prefillRows, prefillCols, prefillCount, 0));
                prefillInputsMap.put("param_tensor", zeroParam);

                @SuppressWarnings("unchecked")
                Map<String, byte[]> prefillOutputs = (Map<String, byte[]>) runModel.invoke(
                        decodeSegment, bindings, environment, prefillInputsMap, prefillSignature);
                for (Map.Entry<String, byte[]> entry : prefillOutputs.entrySet()) {
                    if (entry.getKey().startsWith("kv_cache_")) {
                        state.put(entry.getKey(), entry.getValue());
                    }
                }

                @SuppressWarnings("unchecked")
                Map<String, byte[]> embedderOutputs = (Map<String, byte[]>) runEmbedder.invoke(
                        embedderSegment, bindings, environment, encodeInt32(lastToken), embedderSignature);
                @SuppressWarnings("unchecked")
                Map<String, byte[]> perLayerOutputs = (Map<String, byte[]>) runPerLayer.invoke(
                        perLayerSegment, bindings, environment, encodeInt32(lastToken), perLayerSignature);

                Map<String, byte[]> decodeInputsMap = new HashMap<>(state);
                decodeInputsMap.put("embeddings", embedderOutputs.get("embeddings"));
                decodeInputsMap.put("per_layer_embeddings", perLayerOutputs.get("embeddings"));
                decodeInputsMap.put("input_pos", encodeInt32(lastPosition));
                decodeInputsMap.put("mask", buildCausalRawMask((Integer) getField(findInputSpec(decodeInputs, "mask"), "reqBytes"), lastPosition));
                decodeInputsMap.put("param_tensor", new byte[(Integer) getField(findInputSpec(decodeInputs, "param_tensor"), "reqBytes")]);

                @SuppressWarnings("unchecked")
                Map<String, byte[]> outputs = (Map<String, byte[]>) runModel.invoke(
                        decodeSegment, bindings, environment, decodeInputsMap, decodeSignature);
                float[] logits = littleEndianFloats(outputs.get("logits"));
                int[] top = topKIndices(logits, 5);
                System.out.println("prefill128 mask=" + maskMode);
                for (int id : top) {
                    System.out.println("  id=" + id + " logit=" + logits[id] + " text=[" + tokenizer.decodeToken(id) + "]");
                }
            }
        }
    }

    @Test
    public void searchNativeConsistentEmbeddingLayout() throws Exception {
        Path taskPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");

        Map<String, LiteRTContainerParser.WeightEntry> taskWeights = LiteRTContainerParser.extractWeightMap(taskPath);
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);
        try (Arena arena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(taskPath, StandardOpenOption.READ);
             LiteRTTokenizer tokenizer = LiteRTTokenizer.create(taskPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            NativeEmbedderProbe probe = openNativeEmbedder(nativeRunner, bindings, litertlmPath);
            MemorySegment packed = map(channel, arena, taskWeights.get("transformer.embedder.input_embedding.w"));
            MemorySegment scale = map(channel, arena, taskWeights.get("transformer.embedder.input_embedding.w_quantized_scale"));
            int vocabSize = (int) (scale.byteSize() / Float.BYTES);
            int dim = 1536;
            int[] trainTokens = new int[] { 0, 1, 42 };
            int[] testTokens = new int[] { 506, 667 };

            for (LayoutCandidate layout : buildLayoutCandidates()) {
                float[] codebook = inferEmbeddingCodebookFromNative(
                        packed, scale, vocabSize, dim, layout, probe, trainTokens);
                double trainMse = 0.0;
                double testMse = 0.0;
                double testCosine = 0.0;

                for (int tokenId : trainTokens) {
                    float[] nativeEmbedding = probe.embeddingForToken(tokenId);
                    float[] candidate = decodePacked2BitEmbeddingRowWithCodebook(
                            packed, scale, tokenId, dim, vocabSize, layout, codebook);
                    trainMse += mse(candidate, nativeEmbedding);
                }
                for (int tokenId : testTokens) {
                    float[] nativeEmbedding = probe.embeddingForToken(tokenId);
                    float[] candidate = decodePacked2BitEmbeddingRowWithCodebook(
                            packed, scale, tokenId, dim, vocabSize, layout, codebook);
                    testMse += mse(candidate, nativeEmbedding);
                    testCosine += cosineSimilarity(candidate, nativeEmbedding);
                }

                trainMse /= trainTokens.length;
                testMse /= testTokens.length;
                testCosine /= testTokens.length;
                System.out.println(layout.label
                        + " codebook=" + Arrays.toString(codebook)
                        + " trainMse=" + trainMse
                        + " testMse=" + testMse
                        + " testCosine=" + testCosine);
            }
        }
    }

    @Test
    public void verifyInputMajorRowSums() throws Exception {
        Path taskPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");
        Map<String, LiteRTContainerParser.WeightEntry> weights = LiteRTContainerParser.extractWeightMap(taskPath);

        try (Arena arena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(taskPath, StandardOpenOption.READ)) {
            MemorySegment packed = map(channel, arena, weights.get("transformer.layer_0.attn.q.w"));
            MemorySegment sums = map(channel, arena, weights.get("transformer.layer_0.attn.q.w.sum_i"));
            int inDim = 1536;
            int outDim = 2048;
            for (int row = 0; row < 8; row++) {
                int decodedSum = sumInputMajorOutputRow(packed, row, inDim, outDim);
                int stored = sums.getAtIndex(ValueLayout.JAVA_INT, row);
                System.out.println("row " + row + " stored=" + stored + " inputMajorCentered=" + decodedSum);
            }
        }
    }

    @Test
    public void dumpNextTokensForPrompt() throws Exception {
        Path modelPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(modelPath);
             LiteRTGemmaMetalRunner runner = new LiteRTGemmaMetalRunner(modelPath, tokenizer)) {
            runner.initialize();

            Method forwardHiddenState = LiteRTGemmaMetalRunner.class.getDeclaredMethod(
                    "forwardHiddenState", int.class, int.class, Arena.class);
            forwardHiddenState.setAccessible(true);

            Method argmax = LiteRTGemmaMetalRunner.class.getDeclaredMethod(
                    "argmaxLogitsFromEmbeddingHead", MemorySegment.class, Arena.class);
            argmax.setAccessible(true);

            int[] promptIds = tokenizer.encodeChatPrompt("where is jakarta");
            int cacheLen = 0;
            int nextToken;

            for (int i = 0; i < promptIds.length - 1; i++) {
                try (Arena stepArena = Arena.ofConfined()) {
                    forwardHiddenState.invoke(runner, promptIds[i], cacheLen, stepArena);
                    cacheLen++;
                }
            }

            try (Arena stepArena = Arena.ofConfined()) {
                MemorySegment hiddenState = (MemorySegment) forwardHiddenState.invoke(
                        runner, promptIds[promptIds.length - 1], cacheLen, stepArena);
                nextToken = (int) argmax.invoke(runner, hiddenState, stepArena);
                cacheLen++;
            }

            List<String> emitted = new ArrayList<>();
            for (int i = 0; i < 10; i++) {
                String decoded = tokenizer.decodeToken(nextToken);
                emitted.add("#" + i + " token=" + nextToken + " text=[" + decoded + "]");
                try (Arena stepArena = Arena.ofConfined()) {
                    MemorySegment hiddenState = (MemorySegment) forwardHiddenState.invoke(
                            runner, nextToken, cacheLen, stepArena);
                    nextToken = (int) argmax.invoke(runner, hiddenState, stepArena);
                    cacheLen++;
                }
            }

            System.out.println("=== GENERATED TOKENS ===");
            emitted.forEach(System.out::println);
        }
    }

    @Test
    public void compareTokenizerVariants() throws Exception {
        Path modelPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it-web.task");

        try (LiteRTTokenizer hfTokenizer = LiteRTTokenizer.create(modelPath)) {
            LiteRTTokenizer spmTokenizer = new LiteRTTokenizer();
            Method loadFromSpmBytes = LiteRTTokenizer.class.getDeclaredMethod("loadFromSpmBytes", byte[].class);
            loadFromSpmBytes.setAccessible(true);
            byte[] spmBytes = LiteRTContainerParser.extractSpmVocab(modelPath).orElseThrow();
            loadFromSpmBytes.invoke(spmTokenizer, spmBytes);

            int[] hfPrompt = hfTokenizer.encodeChatPrompt("where is jakarta");
            int[] spmPrompt = spmTokenizer.encodeChatPrompt("where is jakarta");
            System.out.println("HF prompt ids=" + Arrays.toString(hfPrompt));
            System.out.println("SPM prompt ids=" + Arrays.toString(spmPrompt));
            for (int tokenId : new int[] { 0, 1, 2, 3, 42, 106, 506, 667 }) {
                System.out.println("token " + tokenId
                        + " hf=[" + hfTokenizer.decodeToken(tokenId) + "]"
                        + " spm=[" + spmTokenizer.decodeToken(tokenId) + "]");
            }

            spmTokenizer.close();
        }
    }

    private static void dumpSize(Map<String, LiteRTContainerParser.WeightEntry> weights, String name) {
        LiteRTContainerParser.WeightEntry entry = weights.get(name);
        if (entry == null) {
            System.out.println(name + " -> MISSING");
            return;
        }
        System.out.println(name + " -> " + entry.size() + " bytes");
    }

    private static void dumpFloatSamples(
            FileChannel channel,
            Arena arena,
            Map<String, LiteRTContainerParser.WeightEntry> weights,
            String name) throws Exception {
        LiteRTContainerParser.WeightEntry entry = weights.get(name);
        if (entry == null) {
            System.out.println(name + " -> MISSING");
            return;
        }
        MemorySegment seg = channel.map(FileChannel.MapMode.READ_ONLY, entry.offset(), entry.size(), arena);
        int count = (int) Math.min(8, entry.size() / Float.BYTES);
        StringBuilder sb = new StringBuilder(name).append(" ->");
        for (int i = 0; i < count; i++) {
            sb.append(' ').append(seg.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        }
        System.out.println(sb);
    }

    private static void dumpIntSamples(
            FileChannel channel,
            Arena arena,
            Map<String, LiteRTContainerParser.WeightEntry> weights,
            String name) throws Exception {
        LiteRTContainerParser.WeightEntry entry = weights.get(name);
        if (entry == null) {
            System.out.println(name + " -> MISSING");
            return;
        }
        MemorySegment seg = channel.map(FileChannel.MapMode.READ_ONLY, entry.offset(), entry.size(), arena);
        int count = (int) Math.min(8, entry.size() / Integer.BYTES);
        StringBuilder sb = new StringBuilder(name).append(" ->");
        for (int i = 0; i < count; i++) {
            sb.append(' ').append(seg.getAtIndex(ValueLayout.JAVA_INT, i));
        }
        System.out.println(sb);
    }

    private static void comparePackedRowSums(
            FileChannel channel,
            Arena arena,
            Map<String, LiteRTContainerParser.WeightEntry> weights,
            String weightName,
            String sumName,
            int rowDim) throws Exception {
        LiteRTContainerParser.WeightEntry weightEntry = weights.get(weightName);
        LiteRTContainerParser.WeightEntry sumEntry = weights.get(sumName);
        if (weightEntry == null || sumEntry == null) {
            System.out.println(weightName + " / " + sumName + " -> MISSING");
            return;
        }
        MemorySegment weightSeg = channel.map(FileChannel.MapMode.READ_ONLY, weightEntry.offset(), weightEntry.size(), arena);
        MemorySegment sumSeg = channel.map(FileChannel.MapMode.READ_ONLY, sumEntry.offset(), sumEntry.size(), arena);
        int rowBytes = rowDim / 2;
        for (int row = 0; row < 4; row++) {
            int signedExtSum = 0;
            int centeredUnsignedSum = 0;
            long rowOffset = (long) row * rowBytes;
            for (int i = 0; i < rowBytes; i++) {
                int packed = weightSeg.get(ValueLayout.JAVA_BYTE, rowOffset + i) & 0xFF;
                int lo = packed & 0x0F;
                int hi = (packed >>> 4) & 0x0F;

                int loSigned = lo >= 8 ? lo - 16 : lo;
                int hiSigned = hi >= 8 ? hi - 16 : hi;
                signedExtSum += loSigned + hiSigned;

                centeredUnsignedSum += (lo - 8) + (hi - 8);
            }
            int stored = sumSeg.getAtIndex(ValueLayout.JAVA_INT, row);
            System.out.println(weightName + " row " + row
                    + " stored=" + stored
                    + " signedExt=" + signedExtSum
                    + " centeredUnsigned=" + centeredUnsignedSum);
        }
    }

    private static SafetensorLoadResult loadSafetensorFile(Path path) throws Exception {
        SafetensorFFMLoader loader = new SafetensorFFMLoader();
        Field objectMapperField = SafetensorFFMLoader.class.getDeclaredField("objectMapper");
        objectMapperField.setAccessible(true);
        objectMapperField.set(loader, new ObjectMapper());
        Method loadMmap = SafetensorFFMLoader.class.getDeclaredMethod("loadMmap", Path.class);
        loadMmap.setAccessible(true);
        return (SafetensorLoadResult) loadMmap.invoke(loader, path);
    }

    private static MemorySegment map(
            FileChannel channel,
            Arena arena,
            LiteRTContainerParser.WeightEntry entry) throws Exception {
        return channel.map(FileChannel.MapMode.READ_ONLY, entry.offset(), entry.size(), arena);
    }

    private static float[] decodeTaskRow(
            MemorySegment packed,
            MemorySegment scale,
            int row,
            int rowDim,
            boolean centeredUnsigned) {
        float[] out = new float[rowDim];
        float s = scale.getAtIndex(ValueLayout.JAVA_FLOAT, row);
        long rowOffset = (long) row * (rowDim / 2);
        for (int i = 0; i < rowDim; i += 2) {
            int byteVal = packed.get(ValueLayout.JAVA_BYTE, rowOffset + (i / 2)) & 0xFF;
            int lo = byteVal & 0x0F;
            int hi = (byteVal >>> 4) & 0x0F;
            if (centeredUnsigned) {
                lo -= 8;
                hi -= 8;
            } else {
                if (lo >= 8) lo -= 16;
                if (hi >= 8) hi -= 16;
            }
            out[i] = lo * s;
            out[i + 1] = hi * s;
        }
        return out;
    }

    private static float[] sliceRow(float[] all, int row, int rowDim) {
        float[] out = new float[rowDim];
        System.arraycopy(all, row * rowDim, out, 0, rowDim);
        return out;
    }

    private static Match bestRowMatch(float[] candidate, float[] reference, int rows, int rowDim) {
        int bestIndex = -1;
        double bestCosine = Double.NEGATIVE_INFINITY;
        for (int row = 0; row < rows; row++) {
            double cosine = cosineSimilarity(candidate, reference, row * rowDim, rowDim);
            if (cosine > bestCosine) {
                bestCosine = cosine;
                bestIndex = row;
            }
        }
        return new Match(bestIndex, bestCosine);
    }

    private static Object getField(Object target, String name) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(target);
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static String candidateLabel(Object candidate) throws Exception {
        if (candidate == null) {
            return "null";
        }
        Object label = getField(candidate, "label");
        Object offset = getField(candidate, "offset");
        return label + "@0x" + Long.toHexString((Long) offset);
    }

    private static float[] decodeTaskOutputRowInputMajor(
            MemorySegment packed,
            MemorySegment scale,
            int outRow,
            int inDim,
            int outDim,
            boolean centeredUnsigned) {
        float[] out = new float[inDim];
        float s = scale.getAtIndex(ValueLayout.JAVA_FLOAT, outRow);
        for (int in = 0; in < inDim; in++) {
            long flatIndex = (long) in * outDim + outRow;
            int byteVal = packed.get(ValueLayout.JAVA_BYTE, flatIndex / 2) & 0xFF;
            int q = ((flatIndex & 1L) == 0L) ? (byteVal & 0x0F) : ((byteVal >>> 4) & 0x0F);
            if (centeredUnsigned) {
                q -= 8;
            } else if (q >= 8) {
                q -= 16;
            }
            out[in] = q * s;
        }
        return out;
    }

    private static float[] decodeTaskOutputRowInputMajorSwapped(
            MemorySegment packed,
            MemorySegment scale,
            int outRow,
            int inDim,
            int outDim,
            boolean centeredUnsigned) {
        float[] out = new float[inDim];
        float s = scale.getAtIndex(ValueLayout.JAVA_FLOAT, outRow);
        for (int in = 0; in < inDim; in++) {
            long flatIndex = (long) in * outDim + outRow;
            int byteVal = packed.get(ValueLayout.JAVA_BYTE, flatIndex / 2) & 0xFF;
            int q = ((flatIndex & 1L) == 0L) ? ((byteVal >>> 4) & 0x0F) : (byteVal & 0x0F);
            if (centeredUnsigned) {
                q -= 8;
            } else if (q >= 8) {
                q -= 16;
            }
            out[in] = q * s;
        }
        return out;
    }

    private static float[] decodePacked2BitEmbeddingRow(
            MemorySegment packed,
            MemorySegment scale,
            int tokenId,
            int dim,
            TwoBitMode mode) {
        float[] out = new float[dim];
        float s = scale.getAtIndex(ValueLayout.JAVA_FLOAT, tokenId);
        int rowBytes = dim / 4;
        long rowOffset = (long) tokenId * rowBytes;
        for (int i = 0; i < dim; i++) {
            int byteVal = packed.get(ValueLayout.JAVA_BYTE, rowOffset + (i / 4)) & 0xFF;
            int shift = (i & 0x03) * 2;
            int q = (byteVal >>> shift) & 0x03;
            out[i] = mode.decode(q) * s;
        }
        return out;
    }

    private static float[] decodePacked2BitEmbeddingRowInputMajor(
            MemorySegment packed,
            MemorySegment scale,
            int tokenId,
            int dim,
            int vocabSize,
            TwoBitMode mode) {
        float[] out = new float[dim];
        float s = scale.getAtIndex(ValueLayout.JAVA_FLOAT, tokenId);
        for (int i = 0; i < dim; i++) {
            int q = read2BitInputMajor(packed, tokenId, i, vocabSize);
            out[i] = mode.decode(q) * s;
        }
        return out;
    }

    private static float[] decodePacked2BitEmbeddingRowBlocked(
            MemorySegment packed,
            MemorySegment scale,
            int tokenId,
            int dim,
            int vocabSize,
            int blockSize,
            TwoBitMode mode) {
        float[] out = new float[dim];
        float s = scale.getAtIndex(ValueLayout.JAVA_FLOAT, tokenId);
        int blockId = tokenId / blockSize;
        int tokenInBlock = tokenId % blockSize;
        for (int i = 0; i < dim; i++) {
            long flatIndex = ((long) blockId * dim + i) * blockSize + tokenInBlock;
            int byteVal = packed.get(ValueLayout.JAVA_BYTE, flatIndex / 4) & 0xFF;
            int shift = (int) ((flatIndex & 0x03L) * 2);
            int q = (byteVal >>> shift) & 0x03;
            out[i] = mode.decode(q) * s;
        }
        return out;
    }

    private static int read2BitRowMajor(MemorySegment packed, int tokenId, int dimIndex, int dim) {
        int rowBytes = dim / 4;
        long rowOffset = (long) tokenId * rowBytes;
        int byteVal = packed.get(ValueLayout.JAVA_BYTE, rowOffset + (dimIndex / 4)) & 0xFF;
        int shift = (dimIndex & 0x03) * 2;
        return (byteVal >>> shift) & 0x03;
    }

    private static int read2BitInputMajor(MemorySegment packed, int tokenId, int dimIndex, int vocabSize) {
        long flatIndex = (long) dimIndex * vocabSize + tokenId;
        int byteVal = packed.get(ValueLayout.JAVA_BYTE, flatIndex / 4) & 0xFF;
        int shift = (int) ((flatIndex & 0x03L) * 2);
        return (byteVal >>> shift) & 0x03;
    }

    private static int read2BitBlocked(MemorySegment packed, int tokenId, int dimIndex, int dim,
                                       int vocabSize, LayoutCandidate layout) {
        long flatIndex;
        if (layout.blockSize == 1) {
            int rowBytes = dim / 4;
            long rowOffset = (long) tokenId * rowBytes;
            flatIndex = rowOffset * 4 + dimIndex;
        } else if (layout.blockSize == vocabSize) {
            flatIndex = (long) dimIndex * vocabSize + tokenId;
        } else {
            int blockId = tokenId / layout.blockSize;
            int tokenInBlock = tokenId % layout.blockSize;
            flatIndex = ((long) blockId * dim + dimIndex) * layout.blockSize + tokenInBlock;
        }
        int byteVal = packed.get(ValueLayout.JAVA_BYTE, flatIndex / 4) & 0xFF;
        int lane = (int) (flatIndex & 0x03L);
        int shift = layout.reverseBitGroups ? (3 - lane) * 2 : lane * 2;
        return (byteVal >>> shift) & 0x03;
    }

    private static int sumInputMajorOutputRow(
            MemorySegment packed,
            int outRow,
            int inDim,
            int outDim) {
        int sum = 0;
        for (int in = 0; in < inDim; in++) {
            long flatIndex = (long) in * outDim + outRow;
            int byteVal = packed.get(ValueLayout.JAVA_BYTE, flatIndex / 2) & 0xFF;
            int q = ((flatIndex & 1L) == 0L) ? (byteVal & 0x0F) : ((byteVal >>> 4) & 0x0F);
            sum += q - 8;
        }
        return sum;
    }

    private static double cosineSimilarity(float[] a, float[] b, int bOffset, int length) {
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < length; i++) {
            float bv = b[bOffset + i];
            dot += a[i] * bv;
            normA += a[i] * a[i];
            normB += bv * bv;
        }
        return dot / Math.max(1e-12, Math.sqrt(normA) * Math.sqrt(normB));
    }

    private static double cosineSimilarity(float[] a, float[] b) {
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / Math.max(1e-12, Math.sqrt(normA) * Math.sqrt(normB));
    }

    private static double mse(float[] a, float[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return sum / a.length;
    }

    private static NativeEmbedderProbe openNativeEmbedder(
            LiteRTGemmaNativeRunner nativeRunner,
            LiteRTNativeBindings bindings,
            Path litertlmPath) throws Exception {
        List<Long> offsets = LiteRTContainerParser.findTfl3SegmentsForInspection(litertlmPath);
        long embedderOffset = offsets.get(0);
        long embedderSize = offsets.get(1) - offsets.get(0);
        Arena sharedArena = (Arena) getField(nativeRunner, "arena");
        MemorySegment environment = bindings.createEnvironment(sharedArena);
        MemorySegment options = bindings.createOptions(sharedArena);
        bindings.setOptionsHardwareAccelerators(options, LiteRTNativeBindings.kLiteRtHwAcceleratorCpu);
        setField(nativeRunner, "environment", environment);
        setField(nativeRunner, "options", options);

        Method loadCompiledSegment = LiteRTGemmaNativeRunner.class.getDeclaredMethod(
                "loadCompiledSegment", String.class, long.class, long.class);
        loadCompiledSegment.setAccessible(true);
        Object embedderSegment = loadCompiledSegment.invoke(nativeRunner, "raw-embedder", embedderOffset, embedderSize);
        @SuppressWarnings("unchecked")
        Map<String, Object> signatures = (Map<String, Object>) getField(embedderSegment, "signatures");
        Object embedderSignature = signatures.get("embedder");
        Method runMethod = embedderSegment.getClass().getDeclaredMethod(
                "run",
                LiteRTNativeBindings.class,
                MemorySegment.class,
                byte[].class,
                embedderSignature.getClass());
        runMethod.setAccessible(true);
        return new NativeEmbedderProbe(bindings, environment, embedderSegment, embedderSignature, runMethod);
    }

    private static void dumpSignatureSpec(Object signature) throws Exception {
        @SuppressWarnings("unchecked")
        List<Object> inputs = (List<Object>) getField(signature, "inputs");
        @SuppressWarnings("unchecked")
        List<Object> outputs = (List<Object>) getField(signature, "outputs");
        for (Object input : inputs) {
            System.out.println("    input name=" + getField(input, "name")
                    + " typeId=" + getField(input, "typeId")
                    + " dims=" + Arrays.toString((int[]) getField(input, "dims"))
                    + " reqBytes=" + getField(input, "reqBytes"));
        }
        for (Object output : outputs) {
            System.out.println("    output name=" + getField(output, "name")
                    + " typeId=" + getField(output, "typeId")
                    + " dims=" + Arrays.toString((int[]) getField(output, "dims"))
                    + " reqBytes=" + getField(output, "reqBytes"));
        }
    }

    private static float[] inferEmbeddingCodebookFromNative(
            MemorySegment packed,
            MemorySegment scale,
            int vocabSize,
            int dim,
            LayoutCandidate layout,
            NativeEmbedderProbe probe,
            int[] trainTokens) throws Exception {
        float[] codebook = new float[4];
        int[] counts = new int[4];
        for (int tokenId : trainTokens) {
            float[] nativeEmbedding = probe.embeddingForToken(tokenId);
            float rowScale = scale.getAtIndex(ValueLayout.JAVA_FLOAT, tokenId);
            for (int i = 0; i < dim; i++) {
                int q = read2BitBlocked(packed, tokenId, i, dim, vocabSize, layout);
                codebook[q] += nativeEmbedding[i] / rowScale;
                counts[q]++;
            }
        }
        for (int q = 0; q < 4; q++) {
            codebook[q] /= Math.max(1, counts[q]);
        }
        return codebook;
    }

    private static float[] decodePacked2BitEmbeddingRowWithCodebook(
            MemorySegment packed,
            MemorySegment scale,
            int tokenId,
            int dim,
            int vocabSize,
            LayoutCandidate layout,
            float[] codebook) {
        float[] out = new float[dim];
        float s = scale.getAtIndex(ValueLayout.JAVA_FLOAT, tokenId);
        for (int i = 0; i < dim; i++) {
            int q = read2BitBlocked(packed, tokenId, i, dim, vocabSize, layout);
            out[i] = codebook[q] * s;
        }
        return out;
    }

    private static List<LayoutCandidate> buildLayoutCandidates() {
        List<LayoutCandidate> layouts = new ArrayList<>();
        for (boolean reverseBits : new boolean[] { false, true }) {
            String suffix = reverseBits ? "-revbits" : "";
            layouts.add(new LayoutCandidate("row-major" + suffix, 1, reverseBits));
            for (int blockSize : new int[] { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 262144 }) {
                layouts.add(new LayoutCandidate("blocked-" + blockSize + suffix, blockSize, reverseBits));
            }
        }
        return layouts;
    }

    private static byte[] encodeInt32(int value) {
        ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(value);
        return buffer.array();
    }

    private static byte[] encodeIntRange(int length, int activeCount, int start) {
        ByteBuffer buffer = ByteBuffer.allocate(length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < length; i++) {
            buffer.putInt(i < activeCount ? start + i : 0);
        }
        return buffer.array();
    }

    private static byte[] buildRawMask(int reqBytes) {
        byte[] mask = new byte[reqBytes];
        if (reqBytes > 0) {
            mask[reqBytes - 1] = 1;
        }
        return mask;
    }

    private static byte[] buildCausalRawMask(int reqBytes, int position) {
        byte[] mask = new byte[reqBytes];
        int currentSlot = reqBytes - 1;
        int endExclusive = Math.min(position, currentSlot);
        for (int i = 0; i < endExclusive; i++) {
            mask[i] = 1;
        }
        mask[currentSlot] = 1;
        return mask;
    }

    private static byte[] buildPrefillMask(
            String mode,
            int reqBytes,
            int rows,
            int cols,
            int tokenCount,
            int pastCount) {
        byte[] mask = new byte[reqBytes];
        int activeRows = Math.min(tokenCount, rows);
        int clampedPast = Math.max(0, Math.min(pastCount, cols));
        switch (mode) {
            case "absolute" -> {
                for (int row = 0; row < activeRows; row++) {
                    int visibleCols = Math.min(cols, clampedPast + row + 1);
                    for (int col = 0; col < visibleCols; col++) {
                        mask[row * cols + col] = 1;
                    }
                }
            }
            case "tail" -> {
                int rowOffset = Math.max(0, rows - activeRows);
                int colOffset = Math.max(0, cols - activeRows);
                for (int row = 0; row < activeRows; row++) {
                    for (int col = 0; col <= row && colOffset + col < cols; col++) {
                        mask[(rowOffset + row) * cols + colOffset + col] = 1;
                    }
                }
            }
            case "past_plus_tail" -> {
                int rowOffset = Math.max(0, rows - activeRows);
                int tailStart = Math.max(clampedPast, cols - activeRows);
                for (int row = 0; row < activeRows; row++) {
                    for (int col = 0; col < clampedPast; col++) {
                        mask[(rowOffset + row) * cols + col] = 1;
                    }
                    for (int col = 0; col <= row && tailStart + col < cols; col++) {
                        mask[(rowOffset + row) * cols + tailStart + col] = 1;
                    }
                }
            }
            default -> throw new IllegalArgumentException("Unknown prefill mask mode: " + mode);
        }
        return mask;
    }

    private static float[] littleEndianFloats(byte[] bytes) {
        if (bytes.length % Float.BYTES != 0) {
            throw new IllegalArgumentException("Byte length is not divisible by 4");
        }
        float[] out = new float[bytes.length / Float.BYTES];
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < out.length; i++) {
            out[i] = buffer.getFloat();
        }
        return out;
    }

    private static float[] littleEndianBf16(byte[] bytes) {
        if (bytes.length % Short.BYTES != 0) {
            throw new IllegalArgumentException("Byte length is not divisible by 2");
        }
        float[] out = new float[bytes.length / Short.BYTES];
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < out.length; i++) {
            int bf16 = buffer.getShort() & 0xFFFF;
            out[i] = Float.intBitsToFloat(bf16 << 16);
        }
        return out;
    }

    private static float[] decodeNativeFloatTensor(byte[] bytes) {
        if (bytes.length == 1536 * Float.BYTES) {
            return littleEndianFloats(bytes);
        }
        if (bytes.length == 1536 * Short.BYTES) {
            return littleEndianBf16(bytes);
        }
        throw new IllegalArgumentException("Unexpected embedding tensor byte length: " + bytes.length);
    }

    private static int[] topKIndices(float[] values, int k) {
        int[] ids = new int[k];
        float[] scores = new float[k];
        Arrays.fill(ids, -1);
        Arrays.fill(scores, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < values.length; i++) {
            int minIndex = 0;
            for (int j = 1; j < k; j++) {
                if (scores[j] < scores[minIndex]) {
                    minIndex = j;
                }
            }
            if (values[i] > scores[minIndex]) {
                scores[minIndex] = values[i];
                ids[minIndex] = i;
            }
        }
        for (int i = 0; i < k - 1; i++) {
            int best = i;
            for (int j = i + 1; j < k; j++) {
                if (scores[j] > scores[best]) {
                    best = j;
                }
            }
            if (best != i) {
                float tmpScore = scores[i];
                scores[i] = scores[best];
                scores[best] = tmpScore;
                int tmpId = ids[i];
                ids[i] = ids[best];
                ids[best] = tmpId;
            }
        }
        return ids;
    }

    private static Object findInputSpec(List<Object> inputs, String name) throws Exception {
        for (Object input : inputs) {
            if (name.equals(getField(input, "name"))) {
                return input;
            }
        }
        throw new IllegalArgumentException("Missing input spec: " + name);
    }

    private static int[] promptLoopTopKForPrompt(String prompt, int k) throws Exception {
        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm"))) {
            return promptLoopTopKForTokens(tokenizer.encodeChatPrompt(prompt), 0, k);
        }
    }

    private static int[] promptLoopTopKForTokens(int[] promptIds, int positionOffset, int k) throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            List<Long> offsets = LiteRTContainerParser.findTfl3SegmentsForInspection(litertlmPath);
            Arena sharedArena = (Arena) getField(nativeRunner, "arena");
            MemorySegment environment = bindings.createEnvironment(sharedArena);
            MemorySegment options = bindings.createOptions(sharedArena);
            bindings.setOptionsHardwareAccelerators(options, LiteRTNativeBindings.kLiteRtHwAcceleratorCpu);
            setField(nativeRunner, "environment", environment);
            setField(nativeRunner, "options", options);

            Method loadCompiledSegment = LiteRTGemmaNativeRunner.class.getDeclaredMethod(
                    "loadCompiledSegment", String.class, long.class, long.class);
            loadCompiledSegment.setAccessible(true);

            Object embedderSegment = loadCompiledSegment.invoke(nativeRunner, "raw-embedder",
                    offsets.get(0), offsets.get(1) - offsets.get(0));
            Object perLayerSegment = loadCompiledSegment.invoke(nativeRunner, "raw-per-layer",
                    offsets.get(1), offsets.get(2) - offsets.get(1));
            Object decodeSegment = loadCompiledSegment.invoke(nativeRunner, "raw-decode",
                    offsets.get(8), offsets.get(9) - offsets.get(8));

            @SuppressWarnings("unchecked")
            Object embedderSignature = ((Map<String, Object>) getField(embedderSegment, "signatures")).get("embedder");
            @SuppressWarnings("unchecked")
            Object perLayerSignature = ((Map<String, Object>) getField(perLayerSegment, "signatures")).get("per_layer_embedder");
            @SuppressWarnings("unchecked")
            Object decodeSignature = ((Map<String, Object>) getField(decodeSegment, "signatures")).get("decode");

            Method runEmbedder = embedderSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, byte[].class, embedderSignature.getClass());
            Method runPerLayer = perLayerSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, byte[].class, perLayerSignature.getClass());
            Method runDecode = decodeSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, Map.class, decodeSignature.getClass());
            runEmbedder.setAccessible(true);
            runPerLayer.setAccessible(true);
            runDecode.setAccessible(true);

            @SuppressWarnings("unchecked")
            List<Object> decodeInputs = (List<Object>) getField(decodeSignature, "inputs");
            Map<String, byte[]> state = new java.util.HashMap<>();
            for (Object inputSpec : decodeInputs) {
                String name = (String) getField(inputSpec, "name");
                int reqBytes = (Integer) getField(inputSpec, "reqBytes");
                if (name.startsWith("kv_cache_")) {
                    state.put(name, new byte[reqBytes]);
                }
            }

            byte[] lastLogits = null;
            for (int position = 0; position < promptIds.length; position++) {
                int tokenId = promptIds[position];
                @SuppressWarnings("unchecked")
                Map<String, byte[]> embedderOutputs = (Map<String, byte[]>) runEmbedder.invoke(
                        embedderSegment, bindings, environment, encodeInt32(tokenId), embedderSignature);
                @SuppressWarnings("unchecked")
                Map<String, byte[]> perLayerOutputs = (Map<String, byte[]>) runPerLayer.invoke(
                        perLayerSegment, bindings, environment, encodeInt32(tokenId), perLayerSignature);

                Map<String, byte[]> inputs = new java.util.HashMap<>(state);
                inputs.put("embeddings", embedderOutputs.get("embeddings"));
                inputs.put("per_layer_embeddings", perLayerOutputs.get("embeddings"));
                int adjustedPosition = Math.max(0, position + positionOffset);
                inputs.put("input_pos", encodeInt32(adjustedPosition));
                inputs.put("mask", buildCausalRawMask((Integer) getField(findInputSpec(decodeInputs, "mask"), "reqBytes"), adjustedPosition));
                inputs.put("param_tensor", new byte[(Integer) getField(findInputSpec(decodeInputs, "param_tensor"), "reqBytes")]);

                @SuppressWarnings("unchecked")
                Map<String, byte[]> outputs = (Map<String, byte[]>) runDecode.invoke(
                        decodeSegment, bindings, environment, inputs, decodeSignature);
                for (Map.Entry<String, byte[]> entry : outputs.entrySet()) {
                    if (entry.getKey().startsWith("kv_cache_")) {
                        state.put(entry.getKey(), entry.getValue());
                    }
                }
                lastLogits = outputs.get("logits");
            }

            return topKIndices(littleEndianFloats(lastLogits), k);
        }
    }

    private static String describeTokens(LiteRTTokenizer tokenizer, int[] tokenIds) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < tokenIds.length; i++) {
            if (i > 0) {
                out.append(", ");
            }
            int tokenId = tokenIds[i];
            out.append(tokenId).append("=").append(tokenizer.decodeToken(tokenId).replace("\n", "\\n"));
        }
        out.append(']');
        return out.toString();
    }

    private static int[] prefill128TopKForPrompt(String prompt, String maskMode, int k) throws Exception {
        Path litertlmPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs",
                "7c51c9e7-0272-4738-bb52-f3954da26957", "gemma-4-E2B-it.litertlm");
        Path litertLib = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libLiteRt.dylib");
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(litertLib);

        try (LiteRTTokenizer tokenizer = LiteRTTokenizer.create(litertlmPath);
             LiteRTGemmaNativeRunner nativeRunner = new LiteRTGemmaNativeRunner(bindings, litertlmPath, tokenizer, false)) {
            List<Long> offsets = LiteRTContainerParser.findTfl3SegmentsForInspection(litertlmPath);
            Arena sharedArena = (Arena) getField(nativeRunner, "arena");
            MemorySegment environment = bindings.createEnvironment(sharedArena);
            MemorySegment options = bindings.createOptions(sharedArena);
            bindings.setOptionsHardwareAccelerators(options, LiteRTNativeBindings.kLiteRtHwAcceleratorCpu);
            setField(nativeRunner, "environment", environment);
            setField(nativeRunner, "options", options);

            Method loadCompiledSegment = LiteRTGemmaNativeRunner.class.getDeclaredMethod(
                    "loadCompiledSegment", String.class, long.class, long.class);
            loadCompiledSegment.setAccessible(true);

            Object embedderSegment = loadCompiledSegment.invoke(nativeRunner, "raw-embedder",
                    offsets.get(0), offsets.get(1) - offsets.get(0));
            Object perLayerSegment = loadCompiledSegment.invoke(nativeRunner, "raw-per-layer",
                    offsets.get(1), offsets.get(2) - offsets.get(1));
            Object decodeSegment = loadCompiledSegment.invoke(nativeRunner, "raw-decode",
                    offsets.get(8), offsets.get(9) - offsets.get(8));

            @SuppressWarnings("unchecked")
            Object embedderSignature = ((Map<String, Object>) getField(embedderSegment, "signatures")).get("embedder");
            @SuppressWarnings("unchecked")
            Object perLayerSignature = ((Map<String, Object>) getField(perLayerSegment, "signatures")).get("per_layer_embedder");
            @SuppressWarnings("unchecked")
            Map<String, Object> decodeSignatures = (Map<String, Object>) getField(decodeSegment, "signatures");
            Object decodeSignature = decodeSignatures.get("decode");
            Object prefillSignature = decodeSignatures.get("prefill_128");

            Method runEmbedder = embedderSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, byte[].class, embedderSignature.getClass());
            Method runPerLayer = perLayerSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, byte[].class, perLayerSignature.getClass());
            Method runModel = decodeSegment.getClass().getDeclaredMethod(
                    "run", LiteRTNativeBindings.class, MemorySegment.class, Map.class, decodeSignature.getClass());
            runEmbedder.setAccessible(true);
            runPerLayer.setAccessible(true);
            runModel.setAccessible(true);

            @SuppressWarnings("unchecked")
            List<Object> decodeInputs = (List<Object>) getField(decodeSignature, "inputs");
            @SuppressWarnings("unchecked")
            List<Object> prefillInputs = (List<Object>) getField(prefillSignature, "inputs");

            int[] promptIds = tokenizer.encodeChatPrompt(prompt);
            int prefillCount = promptIds.length - 1;
            int lastToken = promptIds[promptIds.length - 1];
            int lastPosition = prefillCount;

            byte[] zeroParam = new byte[(Integer) getField(findInputSpec(prefillInputs, "param_tensor"), "reqBytes")];
            int prefillMaskBytes = (Integer) getField(findInputSpec(prefillInputs, "mask"), "reqBytes");
            int prefillRows = ((int[]) getField(findInputSpec(prefillInputs, "mask"), "dims"))[2];
            int prefillCols = ((int[]) getField(findInputSpec(prefillInputs, "mask"), "dims"))[3];

            Map<String, byte[]> prefillEmbeddings = packNativeEmbeddings(
                    promptIds, prefillCount, runEmbedder, runPerLayer,
                    embedderSegment, perLayerSegment, embedderSignature, perLayerSignature, bindings, environment,
                    findInputSpec(prefillInputs, "embeddings"),
                    findInputSpec(prefillInputs, "per_layer_embeddings"));

            Map<String, byte[]> state = new HashMap<>();
            for (Object inputSpec : prefillInputs) {
                String name = (String) getField(inputSpec, "name");
                int reqBytes = (Integer) getField(inputSpec, "reqBytes");
                if (name.startsWith("kv_cache_")) {
                    state.put(name, new byte[reqBytes]);
                }
            }

            Map<String, byte[]> prefillInputsMap = new HashMap<>(state);
            prefillInputsMap.put("embeddings", prefillEmbeddings.get("embeddings"));
            prefillInputsMap.put("per_layer_embeddings", prefillEmbeddings.get("per_layer_embeddings"));
            prefillInputsMap.put("input_pos", encodeIntRange(prefillRows, prefillCount, 0));
            prefillInputsMap.put("mask", buildPrefillMask(maskMode, prefillMaskBytes, prefillRows, prefillCols, prefillCount, 0));
            prefillInputsMap.put("param_tensor", zeroParam);

            @SuppressWarnings("unchecked")
            Map<String, byte[]> prefillOutputs = (Map<String, byte[]>) runModel.invoke(
                    decodeSegment, bindings, environment, prefillInputsMap, prefillSignature);
            for (Map.Entry<String, byte[]> entry : prefillOutputs.entrySet()) {
                if (entry.getKey().startsWith("kv_cache_")) {
                    state.put(entry.getKey(), entry.getValue());
                }
            }

            @SuppressWarnings("unchecked")
            Map<String, byte[]> embedderOutputs = (Map<String, byte[]>) runEmbedder.invoke(
                    embedderSegment, bindings, environment, encodeInt32(lastToken), embedderSignature);
            @SuppressWarnings("unchecked")
            Map<String, byte[]> perLayerOutputs = (Map<String, byte[]>) runPerLayer.invoke(
                    perLayerSegment, bindings, environment, encodeInt32(lastToken), perLayerSignature);

            Map<String, byte[]> decodeInputsMap = new HashMap<>(state);
            decodeInputsMap.put("embeddings", embedderOutputs.get("embeddings"));
            decodeInputsMap.put("per_layer_embeddings", perLayerOutputs.get("embeddings"));
            decodeInputsMap.put("input_pos", encodeInt32(lastPosition));
            decodeInputsMap.put("mask", buildCausalRawMask((Integer) getField(findInputSpec(decodeInputs, "mask"), "reqBytes"), lastPosition));
            decodeInputsMap.put("param_tensor", new byte[(Integer) getField(findInputSpec(decodeInputs, "param_tensor"), "reqBytes")]);

            @SuppressWarnings("unchecked")
            Map<String, byte[]> outputs = (Map<String, byte[]>) runModel.invoke(
                    decodeSegment, bindings, environment, decodeInputsMap, decodeSignature);
            return topKIndices(littleEndianFloats(outputs.get("logits")), k);
        }
    }

    private static Map<String, byte[]> packNativeEmbeddings(
            int[] prompt,
            int tokenCount,
            Method runEmbedder,
            Method runPerLayer,
            Object embedderSegment,
            Object perLayerSegment,
            Object embedderSignature,
            Object perLayerSignature,
            LiteRTNativeBindings bindings,
            MemorySegment environment,
            Object embeddingsInputSpec,
            Object perLayerEmbeddingsInputSpec) throws Exception {
        byte[] packedEmbeddings = new byte[(Integer) getField(embeddingsInputSpec, "reqBytes")];
        byte[] packedPerLayerEmbeddings = new byte[(Integer) getField(perLayerEmbeddingsInputSpec, "reqBytes")];

        int embeddingOffset = 0;
        int perLayerOffset = 0;
        for (int i = 0; i < tokenCount; i++) {
            @SuppressWarnings("unchecked")
            Map<String, byte[]> embedderOutputs = (Map<String, byte[]>) runEmbedder.invoke(
                    embedderSegment, bindings, environment, encodeInt32(prompt[i]), embedderSignature);
            @SuppressWarnings("unchecked")
            Map<String, byte[]> perLayerOutputs = (Map<String, byte[]>) runPerLayer.invoke(
                    perLayerSegment, bindings, environment, encodeInt32(prompt[i]), perLayerSignature);

            byte[] embedding = embedderOutputs.get("embeddings");
            byte[] perLayerEmbedding = perLayerOutputs.get("embeddings");
            if (embeddingOffset + embedding.length > packedEmbeddings.length) {
                throw new IllegalArgumentException("Packed embeddings overflow: offset="
                        + embeddingOffset + " len=" + embedding.length + " capacity=" + packedEmbeddings.length);
            }
            if (perLayerOffset + perLayerEmbedding.length > packedPerLayerEmbeddings.length) {
                throw new IllegalArgumentException("Packed per-layer embeddings overflow: offset="
                        + perLayerOffset + " len=" + perLayerEmbedding.length + " capacity=" + packedPerLayerEmbeddings.length);
            }
            System.arraycopy(embedding, 0, packedEmbeddings, embeddingOffset, embedding.length);
            System.arraycopy(perLayerEmbedding, 0, packedPerLayerEmbeddings, perLayerOffset, perLayerEmbedding.length);
            embeddingOffset += embedding.length;
            perLayerOffset += perLayerEmbedding.length;
        }

        Map<String, byte[]> packed = new HashMap<>();
        packed.put("embeddings", packedEmbeddings);
        packed.put("per_layer_embeddings", packedPerLayerEmbeddings);
        return packed;
    }

    private record Match(int index, double cosine) {}

    private record LayoutCandidate(String label, int blockSize, boolean reverseBitGroups) {}

    private static final class NativeEmbedderProbe {
        private final LiteRTNativeBindings bindings;
        private final MemorySegment environment;
        private final Object embedderSegment;
        private final Object embedderSignature;
        private final Method runMethod;
        private final Map<Integer, float[]> cache = new java.util.HashMap<>();

        private NativeEmbedderProbe(
                LiteRTNativeBindings bindings,
                MemorySegment environment,
                Object embedderSegment,
                Object embedderSignature,
                Method runMethod) {
            this.bindings = bindings;
            this.environment = environment;
            this.embedderSegment = embedderSegment;
            this.embedderSignature = embedderSignature;
            this.runMethod = runMethod;
        }

        private float[] embeddingForToken(int tokenId) throws Exception {
            float[] cached = cache.get(tokenId);
            if (cached != null) {
                return cached;
            }
            @SuppressWarnings("unchecked")
            Map<String, byte[]> outputs = (Map<String, byte[]>) runMethod.invoke(
                    embedderSegment, bindings, environment, encodeInt32(tokenId), embedderSignature);
            float[] embedding = decodeNativeFloatTensor(outputs.get("embeddings"));
            cache.put(tokenId, embedding);
            return embedding;
        }
    }

    private enum TwoBitMode {
        CENTERED_HALF {
            @Override
            float decode(int q) {
                return q - 1.5f;
            }
        },
        SIGNED_OFFSET_TWO {
            @Override
            float decode(int q) {
                return q - 2.0f;
            }
        },
        UNSIGNED {
            @Override
            float decode(int q) {
                return q;
            }
        },
        CENTERED_HALF_REVERSED {
            @Override
            float decode(int q) {
                return (3 - q) - 1.5f;
            }
        };

        abstract float decode(int q);
    }
}
