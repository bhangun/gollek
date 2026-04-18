package tech.kayys.gollek.inference.llamacpp;

import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages {@code llama_batch} allocation, population, and deallocation.
 *
 * <p>Batches are allocated manually (off-heap via a per-batch {@link Arena}) rather
 * than via {@code llama_batch_init}, so that individual buffer pointers can be
 * updated in-place without copying. The arena is tracked by batch address and
 * closed on {@link #free}.
 */
final class LlamaBatchOps {

    private static final Logger log = Logger.getLogger(LlamaBatchOps.class);

    private final LlamaHandles h;
    private final ConcurrentHashMap<Long, Arena> batchArenas = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Long, BatchBuffers> batchBuffers = new ConcurrentHashMap<>();

    LlamaBatchOps(LlamaHandles handles) {
        this.h = handles;
    }

    /**
     * Allocates a new batch for up to {@code nTokens} token slots.
     *
     * @param nTokens  maximum number of tokens in this batch
     * @param embd     embedding dimension (0 for token-id batches)
     * @param nSeqMax  maximum number of sequences
     * @return a {@link MemorySegment} containing the populated {@code llama_batch} struct
     */
    MemorySegment init(int nTokens, int embd, int nSeqMax) {
        try {
            Arena batchArena = Arena.ofConfined();
            MemorySegment batch  = batchArena.allocate(LlamaStructLayouts.BATCH);
            MemorySegment token  = batchArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment pos    = batchArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment nSeqId = batchArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment logits = batchArena.allocate(ValueLayout.JAVA_BYTE, nTokens);
            MemorySegment seqIdFlat = batchArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment seqIdPtr  = batchArena.allocate(ValueLayout.ADDRESS, nTokens);

            for (int i = 0; i < nTokens; i++) {
                seqIdPtr.setAtIndex(ValueLayout.ADDRESS, i,
                        seqIdFlat.asSlice((long) i * ValueLayout.JAVA_INT.byteSize(),
                                ValueLayout.JAVA_INT.byteSize()));
            }

            set(batch, "n_tokens", nTokens);
            set(batch, "token",    token);
            set(batch, "embd",     MemorySegment.NULL);
            set(batch, "pos",      pos);
            set(batch, "n_seq_id", nSeqId);
            set(batch, "seq_id",   seqIdPtr);
            set(batch, "logits",   logits);

            batchArenas.put(batch.address(), batchArena);
            batchBuffers.put(batch.address(), new BatchBuffers(token, pos, nSeqId, seqIdPtr, logits));
            return batch;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialise batch", e);
        }
    }

    /**
     * Frees the batch and its associated off-heap memory.
     *
     * @param batch the batch segment returned by {@link #init}
     */
    void free(MemorySegment batch) {
        try {
            batchBuffers.remove(batch.address());
            Arena arena = batchArenas.remove(batch.address());
            if (arena != null) {
                arena.close();
                return;
            }
            if (h.batchFree != null) h.batchFree.invoke(batch);
        } catch (Throwable e) {
            log.error("Failed to free batch", e);
        }
    }

    // ── Batch field setters ───────────────────────────────────────────────────

    /** Sets {@code n_tokens} in the batch struct. */
    void setNTokens(MemorySegment batch, int n) {
        batch.set(ValueLayout.JAVA_INT,
                LlamaStructLayouts.BATCH.byteOffset(MemoryLayout.PathElement.groupElement("n_tokens")), n);
    }

    /** Returns the current {@code n_tokens} value from the batch struct. */
    int getNTokens(MemorySegment batch) {
        return batch.get(ValueLayout.JAVA_INT,
                LlamaStructLayouts.BATCH.byteOffset(MemoryLayout.PathElement.groupElement("n_tokens")));
    }

    /**
     * Sets all fields for a single token slot in the batch.
     *
     * @param batch        the batch segment
     * @param index        zero-based slot index
     * @param token        token ID
     * @param pos          position in the sequence
     * @param seqId        sequence ID
     * @param outputLogits {@code true} to request logits for this slot
     */
    void setToken(MemorySegment batch, int index, int token, int pos, int seqId, boolean outputLogits) {
        BatchBuffers manual = batchBuffers.get(batch.address());
        if (manual != null) {
            manual.token.setAtIndex(ValueLayout.JAVA_INT, index, token);
            manual.pos.setAtIndex(ValueLayout.JAVA_INT, index, pos);
            manual.nSeqId.setAtIndex(ValueLayout.JAVA_INT, index, 1);
            manual.seqIdPtr.getAtIndex(ValueLayout.ADDRESS, index)
                    .reinterpret(ValueLayout.JAVA_INT.byteSize())
                    .set(ValueLayout.JAVA_INT, 0, seqId);
            manual.logits.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) (outputLogits ? 1 : 0));
            return;
        }
        // Fallback for native-allocated batches
        ptr(batch, "token",    index, ValueLayout.JAVA_INT.byteSize()).setAtIndex(ValueLayout.JAVA_INT, index, token);
        ptr(batch, "pos",      index, ValueLayout.JAVA_INT.byteSize()).setAtIndex(ValueLayout.JAVA_INT, index, pos);
        ptr(batch, "n_seq_id", index, ValueLayout.JAVA_INT.byteSize()).setAtIndex(ValueLayout.JAVA_INT, index, 1);
        ptr(batch, "seq_id",   index, ValueLayout.ADDRESS.byteSize())
                .getAtIndex(ValueLayout.ADDRESS, index)
                .reinterpret(ValueLayout.JAVA_INT.byteSize())
                .set(ValueLayout.JAVA_INT, 0, seqId);
        ptr(batch, "logits",   index, ValueLayout.JAVA_BYTE.byteSize())
                .setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) (outputLogits ? 1 : 0));
    }

    /** Sets only the token ID for a slot (leaves pos/seqId/logits unchanged). */
    void setTokenId(MemorySegment batch, int index, int token) {
        BatchBuffers manual = batchBuffers.get(batch.address());
        if (manual != null) { manual.token.setAtIndex(ValueLayout.JAVA_INT, index, token); return; }
        ptr(batch, "token", index, ValueLayout.JAVA_INT.byteSize()).setAtIndex(ValueLayout.JAVA_INT, index, token);
    }

    /** Sets the position for a slot. */
    void setPos(MemorySegment batch, int index, int pos) {
        BatchBuffers manual = batchBuffers.get(batch.address());
        if (manual != null) { manual.pos.setAtIndex(ValueLayout.JAVA_INT, index, pos); return; }
        ptr(batch, "pos", index, ValueLayout.JAVA_INT.byteSize()).setAtIndex(ValueLayout.JAVA_INT, index, pos);
    }

    /** Sets the sequence ID for a slot. */
    void setSeqId(MemorySegment batch, int index, int seqId) {
        BatchBuffers manual = batchBuffers.get(batch.address());
        if (manual != null) {
            manual.nSeqId.setAtIndex(ValueLayout.JAVA_INT, index, 1);
            manual.seqIdPtr.getAtIndex(ValueLayout.ADDRESS, index)
                    .reinterpret(ValueLayout.JAVA_INT.byteSize())
                    .set(ValueLayout.JAVA_INT, 0, seqId);
            return;
        }
        ptr(batch, "n_seq_id", index, ValueLayout.JAVA_INT.byteSize()).setAtIndex(ValueLayout.JAVA_INT, index, 1);
        ptr(batch, "seq_id", index, ValueLayout.ADDRESS.byteSize())
                .getAtIndex(ValueLayout.ADDRESS, index)
                .reinterpret(ValueLayout.JAVA_INT.byteSize())
                .set(ValueLayout.JAVA_INT, 0, seqId);
    }

    /** Enables or disables logit output for a slot. */
    void setLogits(MemorySegment batch, int index, boolean enable) {
        BatchBuffers manual = batchBuffers.get(batch.address());
        if (manual != null) { manual.logits.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte)(enable?1:0)); return; }
        ptr(batch, "logits", index, ValueLayout.JAVA_BYTE.byteSize())
                .setAtIndex(ValueLayout.JAVA_BYTE, index, (byte)(enable?1:0));
    }

    /**
     * Sets a multimodal embedding slot (token ID = -1, embd pointer set).
     *
     * @param batch        the batch segment
     * @param index        slot index
     * @param embd         pointer to the embedding data
     * @param pos          position in the sequence
     * @param seqId        sequence ID
     * @param outputLogits {@code true} to request logits for this slot
     */
    void setMultimodalEmbd(MemorySegment batch, int index, MemorySegment embd,
            int pos, int seqId, boolean outputLogits) {
        BatchBuffers manual = batchBuffers.get(batch.address());
        if (manual != null) {
            manual.token.setAtIndex(ValueLayout.JAVA_INT, index, -1);
            batch.get(ValueLayout.ADDRESS,
                    LlamaStructLayouts.BATCH.byteOffset(MemoryLayout.PathElement.groupElement("embd")))
                    .reinterpret((long)(index+1) * ValueLayout.ADDRESS.byteSize())
                    .setAtIndex(ValueLayout.ADDRESS, index, embd);
            manual.pos.setAtIndex(ValueLayout.JAVA_INT, index, pos);
            manual.nSeqId.setAtIndex(ValueLayout.JAVA_INT, index, 1);
            manual.seqIdPtr.getAtIndex(ValueLayout.ADDRESS, index)
                    .reinterpret(ValueLayout.JAVA_INT.byteSize()).set(ValueLayout.JAVA_INT, 0, seqId);
            manual.logits.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte)(outputLogits?1:0));
            return;
        }
        ptr(batch, "token", index, ValueLayout.JAVA_INT.byteSize()).setAtIndex(ValueLayout.JAVA_INT, index, -1);
        ptr(batch, "embd",  index, ValueLayout.ADDRESS.byteSize()).setAtIndex(ValueLayout.ADDRESS, index, embd);
        ptr(batch, "pos",   index, ValueLayout.JAVA_INT.byteSize()).setAtIndex(ValueLayout.JAVA_INT, index, pos);
        ptr(batch, "n_seq_id", index, ValueLayout.JAVA_INT.byteSize()).setAtIndex(ValueLayout.JAVA_INT, index, 1);
        ptr(batch, "seq_id", index, ValueLayout.ADDRESS.byteSize())
                .getAtIndex(ValueLayout.ADDRESS, index)
                .reinterpret(ValueLayout.JAVA_INT.byteSize()).set(ValueLayout.JAVA_INT, 0, seqId);
        ptr(batch, "logits", index, ValueLayout.JAVA_BYTE.byteSize())
                .setAtIndex(ValueLayout.JAVA_BYTE, index, (byte)(outputLogits?1:0));
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void set(MemorySegment batch, String field, Object value) {
        long offset = LlamaStructLayouts.BATCH.byteOffset(MemoryLayout.PathElement.groupElement(field));
        if (value instanceof Integer i)       batch.set(ValueLayout.JAVA_INT, offset, i);
        else if (value instanceof MemorySegment ms) batch.set(ValueLayout.ADDRESS, offset, ms);
    }

    private static MemorySegment ptr(MemorySegment batch, String field, int index, long elemSize) {
        long offset = LlamaStructLayouts.BATCH.byteOffset(MemoryLayout.PathElement.groupElement(field));
        return batch.get(ValueLayout.ADDRESS, offset).reinterpret((long)(index+1) * elemSize);
    }

    // ── Inner type ────────────────────────────────────────────────────────────

    /** Off-heap buffer pointers for a manually-allocated batch. */
    record BatchBuffers(
            MemorySegment token,
            MemorySegment pos,
            MemorySegment nSeqId,
            MemorySegment seqIdPtr,
            MemorySegment logits) {}
}
