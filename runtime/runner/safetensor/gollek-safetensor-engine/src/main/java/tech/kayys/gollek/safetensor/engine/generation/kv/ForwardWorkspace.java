/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.kv;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Reusable scratch memory for direct forward execution.
 *
 * <p>The workspace is owned by a generation session, but kept separate from KV
 * cache block management so attention/cache code does not also own forward
 * projection, logits, and metadata scratch allocation policy.
 */
public class ForwardWorkspace implements AutoCloseable {
    private MemorySegment normedAttnSeg;
    private MemorySegment normedFfnSeg;
    private MemorySegment gateSeg;
    private MemorySegment upSeg;
    private MemorySegment combinedSeg;
    private MemorySegment hiddenASeg;
    private MemorySegment hiddenBSeg;
    private MemorySegment logitsSeg;
    private Arena hiddenArena;
    private Arena combinedArena;
    private Arena projectionArena;
    private Arena logitsArena;
    private Arena attentionMetadataArena;
    private MemorySegment attentionBlockTableSeg;
    private MemorySegment attentionContextLensSeg;
    private long hiddenCap = 0;
    private long combinedCap = 0;
    private long projectionCap = 0;
    private long logitsCap = 0;
    private long attentionBlockTableCap = 0;
    private long attentionContextLensCap = 0;

    public void ensureCapacity(long totalElements, long hiddenSize, long intermediateSize) {
        long needed = Math.max(hiddenSize, totalElements) * Float.BYTES;
        long paddedNeeded = align4096(needed);
        long tokenSlots = hiddenSize <= 0 ? 0L : totalElements / hiddenSize;
        long combinedElements = Math.max(intermediateSize, tokenSlots * intermediateSize);
        long paddedCombined = align4096(combinedElements * Float.BYTES);
        if (normedAttnSeg == null || hiddenCap < paddedNeeded) {
            closeArena(hiddenArena);
            hiddenArena = Arena.ofShared();
            normedAttnSeg = hiddenArena.allocate(paddedNeeded, 4096);
            normedFfnSeg = hiddenArena.allocate(paddedNeeded, 4096);
            hiddenASeg = hiddenArena.allocate(paddedNeeded, 4096);
            hiddenBSeg = hiddenArena.allocate(paddedNeeded, 4096);
            hiddenCap = paddedNeeded;
        }
        if (combinedSeg == null || combinedCap < paddedCombined) {
            closeArena(combinedArena);
            combinedArena = Arena.ofShared();
            combinedSeg = combinedArena.allocate(paddedCombined, 4096);
            combinedCap = paddedCombined;
        }
    }

    public void ensureLogitsCapacity(long requiredElements) {
        long needed = Math.max(1L, requiredElements) * Float.BYTES;
        long padded = align4096(needed);
        if (logitsSeg == null || logitsCap < padded) {
            closeArena(logitsArena);
            logitsArena = Arena.ofShared();
            logitsSeg = logitsArena.allocate(padded, 4096);
            logitsCap = padded;
        }
    }

    public void ensureProjectionScratchCapacity(long requiredBytes) {
        long padded = align4096(requiredBytes);
        if (gateSeg == null || projectionCap < padded) {
            closeArena(projectionArena);
            projectionArena = Arena.ofShared();
            gateSeg = projectionArena.allocate(padded, 4096);
            upSeg = projectionArena.allocate(padded, 4096);
            projectionCap = padded;
        }
    }

    public void ensureAttentionMetadataCapacity(int blockTableEntries, int batch) {
        long blockTableBytes = (long) Math.max(1, blockTableEntries) * Integer.BYTES;
        long contextLensBytes = (long) Math.max(1, batch) * Integer.BYTES;
        if (attentionMetadataArena == null
                || attentionBlockTableCap < blockTableBytes
                || attentionContextLensCap < contextLensBytes) {
            closeArena(attentionMetadataArena);
            attentionMetadataArena = Arena.ofShared();
            attentionBlockTableSeg = attentionMetadataArena.allocate(blockTableBytes, Integer.BYTES);
            attentionContextLensSeg = attentionMetadataArena.allocate(contextLensBytes, Integer.BYTES);
            attentionBlockTableCap = blockTableBytes;
            attentionContextLensCap = contextLensBytes;
        }
    }

    public MemorySegment getNormedAttnSeg() {
        return normedAttnSeg;
    }

    public MemorySegment getNormedFfnSeg() {
        return normedFfnSeg;
    }

    public MemorySegment getGateSeg() {
        return gateSeg;
    }

    public MemorySegment getUpSeg() {
        return upSeg;
    }

    public MemorySegment getCombinedSeg() {
        return combinedSeg;
    }

    public MemorySegment getHiddenASeg() {
        return hiddenASeg;
    }

    public MemorySegment getHiddenBSeg() {
        return hiddenBSeg;
    }

    public MemorySegment getLogitsSeg() {
        return logitsSeg;
    }

    public MemorySegment getAttentionBlockTableSeg() {
        return attentionBlockTableSeg;
    }

    public MemorySegment getAttentionContextLensSeg() {
        return attentionContextLensSeg;
    }

    @Override
    public void close() {
        closeArena(hiddenArena);
        closeArena(combinedArena);
        closeArena(projectionArena);
        closeArena(logitsArena);
        closeArena(attentionMetadataArena);
        hiddenArena = null;
        combinedArena = null;
        projectionArena = null;
        logitsArena = null;
        attentionMetadataArena = null;
        normedAttnSeg = null;
        normedFfnSeg = null;
        gateSeg = null;
        upSeg = null;
        combinedSeg = null;
        hiddenASeg = null;
        hiddenBSeg = null;
        logitsSeg = null;
        attentionBlockTableSeg = null;
        attentionContextLensSeg = null;
        hiddenCap = 0;
        combinedCap = 0;
        projectionCap = 0;
        logitsCap = 0;
        attentionBlockTableCap = 0;
        attentionContextLensCap = 0;
    }

    private static long align4096(long bytes) {
        return (bytes + 4095) & ~4095;
    }

    private static void closeArena(Arena arena) {
        if (arena != null) {
            arena.close();
        }
    }
}
