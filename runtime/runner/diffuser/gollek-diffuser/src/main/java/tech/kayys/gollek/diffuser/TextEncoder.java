package tech.kayys.gollek.diffuser;


import java.lang.foreign.*;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * CLIP text encoder: prompt (String) → embedding tensor (MemorySegment,
 * off-heap).
 *
 * Model: text_encoder.onnx
 * Input: input_ids [1, 77] (int64)
 * Output: last_hidden_state [1, 77, 768]
 *
 * Tokenization is simplified here (BPE subword in production would use a
 * vocab file + merge rules). Replace SimpleTokenizer with a full BPE impl
 * when you have the SD vocab.json / merges.txt files.
 */
public final class TextEncoder {

    private static final int SEQ_LEN = 77;
    private static final int EMBED_DIM = 768;
    private static final int BOS_TOKEN_ID = 49406;
    private static final int EOS_TOKEN_ID = 49407;
    private static final int PAD_TOKEN_ID = 0;

    private final OnnxRuntimeBridge ort;
    private final MemorySegment session;
    private final Arena arena;

    public TextEncoder(OnnxRuntimeBridge ort, Path modelPath, Arena arena) throws Throwable {
        this.ort = ort;
        this.arena = arena;
        this.session = ort.createSession(modelPath);
        System.out.println("[TextEncoder] Loaded: " + modelPath);
    }

    /**
     * Encode a text prompt into a CLIP embedding tensor.
     *
     * @param prompt     text to encode (may be empty for unconditional)
     * @param imageWidth used only for shape validation (ignored in CLIP)
     * @return off-heap MemorySegment of shape [1, 77, 768] floats
     */
    public MemorySegment encode(String prompt, int imageWidth) throws Throwable {
        // 1. Tokenize (simplified — replace with proper BPE for production)
        int[] tokens = tokenize(prompt);

        // 2. Build int64 input_ids tensor [1, 77]
        MemorySegment inputIds = buildInputIdsTensor(tokens);

        // 3. Wrap as OrtValue
        MemorySegment inputTensor = ort.createInt64Tensor(inputIds, new long[] { 1, SEQ_LEN });

        // 4. Run session
        MemorySegment[] outputs = ort.run(
                session,
                new String[] { "input_ids" },
                new MemorySegment[] { inputTensor },
                new String[] { "last_hidden_state" });

        // 5. Extract raw float pointer
        long embedFloats = (long) 1 * SEQ_LEN * EMBED_DIM;
        MemorySegment embedding = ort.getTensorFloatData(outputs[0], embedFloats);

        // Copy to arena-owned segment (ORT output lifetime is tied to the OrtValue)
        MemorySegment owned = arena.allocate(embedFloats * Float.BYTES, 8);
        owned.copyFrom(embedding);

        ort.releaseValue(inputTensor);
        ort.releaseValue(outputs[0]);

        return owned;
    }

    // ── Simplified tokeniser ──────────────────────────────────────────────────

    /**
     * Minimal whitespace tokenizer. Replace with BPE for proper SD use.
     * Each word maps to a stable hash-derived token id in [1, 49405] range.
     */
    private int[] tokenize(String text) {
        int[] ids = new int[SEQ_LEN];
        ids[0] = BOS_TOKEN_ID;

        String[] words = text == null || text.isBlank()
                ? new String[0]
                : text.toLowerCase().trim().split("\\s+");

        int pos = 1;
        for (String word : words) {
            if (pos >= SEQ_LEN - 1)
                break;
            ids[pos++] = Math.abs(word.hashCode() % 49405) + 1;
        }
        ids[pos] = EOS_TOKEN_ID;
        // remaining stay 0 (PAD)
        return ids;
    }

    private MemorySegment buildInputIdsTensor(int[] tokens) {
        MemorySegment seg = arena.allocate((long) SEQ_LEN * Long.BYTES, 8);
        for (int i = 0; i < SEQ_LEN; i++) {
            seg.setAtIndex(ValueLayout.JAVA_LONG, i, tokens[i]);
        }
        return seg;
    }
}
