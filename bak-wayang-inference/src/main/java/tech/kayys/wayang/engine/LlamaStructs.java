package tech.kayys.golek.engine;

import java.lang.foreign.*;
import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tech.kayys.golek.utils.TokenUtils;

public class LlamaStructs {

    private static final Logger log = LoggerFactory.getLogger(LlamaStructs.class);

    // Conservative offsets - these should work with most llama.cpp versions
    public static final long MODEL_PARAMS_GPU_LAYERS_OFFSET = 0;
    public static final long MODEL_PARAMS_USE_MMAP_OFFSET = 25;
    public static final long MODEL_PARAMS_USE_MLOCK_OFFSET = 26;
    // optionally:
    public static final long MODEL_PARAMS_CHECK_TENSORS_OFFSET = 27;
    // Context params offsets
    public static final long CONTEXT_PARAMS_SEED_OFFSET = 0;
    public static final long CONTEXT_PARAMS_N_CTX_OFFSET = 4;
    public static final long CONTEXT_PARAMS_N_BATCH_OFFSET = 8;
    public static final long CONTEXT_PARAMS_N_THREADS_OFFSET = 20;
    public static final long CONTEXT_PARAMS_N_THREADS_BATCH_OFFSET = 24;
    public static final long CONTEXT_PARAMS_ROPE_FREQ_BASE_OFFSET = 36;
    public static final long CONTEXT_PARAMS_ROPE_FREQ_SCALE_OFFSET = 40;
    public static final long CONTEXT_PARAMS_EMBEDDINGS_OFFSET = 96;
    public static final long CONTEXT_PARAMS_FLASH_ATTN_OFFSET = 97;

    public static final long MODEL_PARAMS_SPLIT_MODE_OFFSET = 4;
    public static final long MODEL_PARAMS_MAIN_GPU_OFFSET = 8;
    public static final long MODEL_PARAMS_TENSOR_SPLIT_OFFSET = 16;
    public static final long MODEL_PARAMS_VOCAB_ONLY_OFFSET = 24;

    // Offsets for context params (based on llama.h)
    public static final long CONTEXT_PARAMS_N_UBATCH_OFFSET = 12;
    public static final long CONTEXT_PARAMS_N_SEQ_MAX_OFFSET = 16;

    public static final long CONTEXT_PARAMS_ROPE_SCALING_TYPE_OFFSET = 28;
    public static final long CONTEXT_PARAMS_POOLING_TYPE_OFFSET = 32;

    public static final long CONTEXT_PARAMS_YARN_EXT_FACTOR_OFFSET = 44;
    public static final long CONTEXT_PARAMS_YARN_ATTN_FACTOR_OFFSET = 48;
    public static final long CONTEXT_PARAMS_YARN_BETA_FAST_OFFSET = 52;
    public static final long CONTEXT_PARAMS_YARN_BETA_SLOW_OFFSET = 56;
    public static final long CONTEXT_PARAMS_YARN_ORIG_CTX_OFFSET = 60;
    public static final long CONTEXT_PARAMS_DEFRAG_THOLD_OFFSET = 64;
    public static final long CONTEXT_PARAMS_CB_EVAL_OFFSET = 72;
    public static final long CONTEXT_PARAMS_CB_EVAL_USER_DATA_OFFSET = 80;
    public static final long CONTEXT_PARAMS_TYPE_K_OFFSET = 88;
    public static final long CONTEXT_PARAMS_TYPE_V_OFFSET = 92;
    public static final long CONTEXT_PARAMS_LOGITS_ALL_OFFSET = 96;

    public static final long CONTEXT_PARAMS_OFFLOAD_KQV_OFFSET = 98;

    public static final long CONTEXT_PARAMS_ABORT_CALLBACK_OFFSET = 104;
    public static final long CONTEXT_PARAMS_ABORT_CALLBACK_DATA_OFFSET = 112;

    // llama_model_params struct layout - FIXED padding
    public static final GroupLayout MODEL_PARAMS_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT, // n_gpu_layers
            ValueLayout.JAVA_INT, // split_mode
            ValueLayout.JAVA_INT, // main_gpu
            MemoryLayout.paddingLayout(4), // align pointer
            ValueLayout.ADDRESS, // tensor_split
            ValueLayout.JAVA_BOOLEAN, // vocab_only
            ValueLayout.JAVA_BOOLEAN, // use_mmap
            ValueLayout.JAVA_BOOLEAN, // use_mlock
            ValueLayout.JAVA_BOOLEAN, // check_tensors ← ADD THIS
            MemoryLayout.paddingLayout(4) // t
    ).withName("llama_model_params");

    // llama_context_params struct layout - FIXED padding and order
    public static final GroupLayout CONTEXT_PARAMS_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("seed"), // 0
            ValueLayout.JAVA_INT.withName("n_ctx"), // 4
            ValueLayout.JAVA_INT.withName("n_batch"), // 8
            ValueLayout.JAVA_INT.withName("n_ubatch"), // 12
            ValueLayout.JAVA_INT.withName("n_seq_max"), // 16
            ValueLayout.JAVA_INT.withName("n_threads"), // 20
            ValueLayout.JAVA_INT.withName("n_threads_batch"), // 24
            ValueLayout.JAVA_INT.withName("rope_scaling_type"), // 28
            ValueLayout.JAVA_INT.withName("pooling_type"), // 32
            ValueLayout.JAVA_FLOAT.withName("rope_freq_base"), // 36
            ValueLayout.JAVA_FLOAT.withName("rope_freq_scale"), // 40
            ValueLayout.JAVA_FLOAT.withName("yarn_ext_factor"), // 44
            ValueLayout.JAVA_FLOAT.withName("yarn_attn_factor"), // 48
            ValueLayout.JAVA_FLOAT.withName("yarn_beta_fast"), // 52
            ValueLayout.JAVA_FLOAT.withName("yarn_beta_slow"), // 56
            ValueLayout.JAVA_INT.withName("yarn_orig_ctx"), // 60
            ValueLayout.JAVA_FLOAT.withName("defrag_thold"), // 64
            MemoryLayout.paddingLayout(4), // 68-71
            ValueLayout.ADDRESS.withName("cb_eval"), // 72
            ValueLayout.ADDRESS.withName("cb_eval_user_data"), // 80
            ValueLayout.JAVA_INT.withName("type_k"), // 88
            ValueLayout.JAVA_INT.withName("type_v"), // 92
            ValueLayout.JAVA_BOOLEAN.withName("logits_all"), // 96
            ValueLayout.JAVA_BOOLEAN.withName("embeddings"), // 97
            ValueLayout.JAVA_BOOLEAN.withName("offload_kqv"), // 98
            ValueLayout.JAVA_BOOLEAN.withName("flash_attn"), // 99
            MemoryLayout.paddingLayout(4), // 100-103
            ValueLayout.ADDRESS.withName("abort_callback"), // 104
            ValueLayout.ADDRESS.withName("abort_callback_data") // 112
    ).withName("llama_context_params");

    // llama_batch struct - FIXED layout
    public static final GroupLayout BATCH_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("n_tokens"), // 0
            MemoryLayout.paddingLayout(4), // 4-7
            ValueLayout.ADDRESS.withName("token"), // 8
            ValueLayout.ADDRESS.withName("embd"), // 16
            ValueLayout.ADDRESS.withName("pos"), // 24
            ValueLayout.ADDRESS.withName("n_seq_id"), // 32
            ValueLayout.ADDRESS.withName("seq_id"), // 40
            ValueLayout.ADDRESS.withName("logits"), // 48
            ValueLayout.ADDRESS.withName("all_pos_0"), // 56
            ValueLayout.ADDRESS.withName("all_pos_1"), // 64
            ValueLayout.ADDRESS.withName("all_seq_id") // 72
    ).withName("llama_batch");

    // Batch creation with proper initialization
    public static MemorySegment createBatch(Arena arena, int[] tokens, int[] positions, boolean[] logits) {
        log.info("Create Batch");
        if (tokens.length != positions.length || tokens.length != logits.length) {
            throw new IllegalArgumentException("Tokens, positions, and logits arrays must have same length");
        }

        int nTokens = tokens.length;
        MemorySegment batch = arena.allocate(BATCH_LAYOUT);

        // Initialize batch struct
        batch.set(ValueLayout.JAVA_INT, 0, nTokens); // n_tokens

        // Allocate and fill token array
        MemorySegment tokenSegment = arena.allocate(ValueLayout.JAVA_INT, nTokens);
        for (int i = 0; i < nTokens; i++) {
            tokenSegment.setAtIndex(ValueLayout.JAVA_INT, i, tokens[i]);
        }
        batch.set(ValueLayout.ADDRESS, 8, tokenSegment); // token

        // embd is NULL for token-based batches
        batch.set(ValueLayout.ADDRESS, 16, MemorySegment.NULL); // embd

        // Allocate and fill position array
        MemorySegment posSegment = arena.allocate(ValueLayout.JAVA_INT, nTokens);
        for (int i = 0; i < nTokens; i++) {
            posSegment.setAtIndex(ValueLayout.JAVA_INT, i, positions[i]);
        }
        batch.set(ValueLayout.ADDRESS, 24, posSegment); // pos

        // n_seq_id array (number of sequence IDs per token)
        MemorySegment nSeqIdSegment = arena.allocate(ValueLayout.JAVA_INT, nTokens);
        for (int i = 0; i < nTokens; i++) {
            nSeqIdSegment.setAtIndex(ValueLayout.JAVA_INT, i, 1); // Each token has 1 sequence ID
        }
        batch.set(ValueLayout.ADDRESS, 32, nSeqIdSegment); // n_seq_id

        // seq_id array (sequence IDs for each token)
        MemorySegment seqIdSegment = arena.allocate(ValueLayout.ADDRESS, nTokens);
        for (int i = 0; i < nTokens; i++) {
            MemorySegment seqIdArray = arena.allocate(ValueLayout.JAVA_INT, 1);
            seqIdArray.setAtIndex(ValueLayout.JAVA_INT, 0, 0); // Default sequence ID 0
            seqIdSegment.setAtIndex(ValueLayout.ADDRESS, i, seqIdArray);
        }
        batch.set(ValueLayout.ADDRESS, 40, seqIdSegment); // seq_id

        // logits array (which tokens should return logits)
        MemorySegment logitsSegment = arena.allocate(ValueLayout.JAVA_BYTE, nTokens);
        for (int i = 0; i < nTokens; i++) {
            logitsSegment.setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (logits[i] ? 1 : 0));
        }
        batch.set(ValueLayout.ADDRESS, 48, logitsSegment); // logits

        // New fields in recent llama.cpp versions - set to NULL if not used
        batch.set(ValueLayout.ADDRESS, 56, MemorySegment.NULL); // all_pos_0
        batch.set(ValueLayout.ADDRESS, 64, MemorySegment.NULL); // all_pos_1
        batch.set(ValueLayout.ADDRESS, 72, MemorySegment.NULL); // all_seq_id

        return batch;
    }

    // Simplified batch creation for single-token generation
    public static MemorySegment createSingleTokenBatch(Arena arena, int token, int position, boolean returnLogits) {
        return createBatch(arena,
                new int[] { token },
                new int[] { position },
                new boolean[] { returnLogits });
    }

    // Batch creation for prompt processing
    public static MemorySegment createPromptBatch(Arena arena, int[] tokens, int startPos) {
        int nTokens = tokens.length;
        boolean[] logits = new boolean[nTokens];
        Arrays.fill(logits, false);
        // Only the last token should return logits for generation
        if (nTokens > 0) {
            logits[nTokens - 1] = true;
        }

        int[] positions = new int[nTokens];
        for (int i = 0; i < nTokens; i++) {
            positions[i] = startPos + i;
        }

        return createBatch(arena, tokens, positions, logits);
    }

    // Utility method to print batch info for debugging
    public static void printBatchInfo(MemorySegment batch, String name) {
        System.out.println("Batch " + name + ":");
        System.out.println("  n_tokens: " + batch.get(ValueLayout.JAVA_INT, 0));
        System.out.println("  token: " + batch.get(ValueLayout.ADDRESS, 8));
        System.out.println("  pos: " + batch.get(ValueLayout.ADDRESS, 24));
        System.out.println("  logits: " + batch.get(ValueLayout.ADDRESS, 48));
    }

    // Method to validate batch structure
    public static boolean validateBatch(MemorySegment batch) {
        if (batch == null || batch.address() == 0) {
            return false;
        }

        int nTokens = batch.get(ValueLayout.JAVA_INT, 0);
        if (nTokens <= 0) {
            return false;
        }

        MemorySegment tokens = batch.get(ValueLayout.ADDRESS, 8);
        MemorySegment positions = batch.get(ValueLayout.ADDRESS, 24);

        return tokens.address() != 0 && positions.address() != 0;
    }

    // Helper to get batch field values
    public static int getBatchTokenCount(MemorySegment batch) {
        return batch.get(ValueLayout.JAVA_INT, 0);
    }

    public static MemorySegment getBatchTokens(MemorySegment batch) {
        return batch.get(ValueLayout.ADDRESS, 8);
    }

    public static MemorySegment getBatchPositions(MemorySegment batch) {
        return batch.get(ValueLayout.ADDRESS, 24);
    }

    public static MemorySegment getBatchLogits(MemorySegment batch) {
        return batch.get(ValueLayout.ADDRESS, 48);
    }

    // Method to copy batch contents for debugging
    public static String batchToString(MemorySegment batch, Arena arena, MemorySegment model, LlamaCppBinding binding) {
        try {
            StringBuilder sb = new StringBuilder();
            int nTokens = getBatchTokenCount(batch);
            sb.append("Batch[n_tokens=").append(nTokens).append("]\n");

            MemorySegment tokens = getBatchTokens(batch);
            MemorySegment positions = getBatchPositions(batch);
            MemorySegment logits = getBatchLogits(batch);

            for (int i = 0; i < nTokens; i++) {
                int token = tokens.getAtIndex(ValueLayout.JAVA_INT, i);
                int pos = positions.getAtIndex(ValueLayout.JAVA_INT, i);
                boolean hasLogits = logits.getAtIndex(ValueLayout.JAVA_BYTE, i) != 0;

                String tokenStr = TokenUtils.safeTokenToString(binding, arena, model, token);

                sb.append(String.format("  [%d] pos=%d, token=%d (%s), logits=%b\n",
                        i, pos, token, tokenStr, hasLogits));
            }

            return sb.toString();
        } catch (Exception e) {
            return "Batch[error: " + e.getMessage() + "]";
        }
    }

}