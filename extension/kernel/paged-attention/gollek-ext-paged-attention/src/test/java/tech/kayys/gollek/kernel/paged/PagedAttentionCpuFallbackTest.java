package tech.kayys.gollek.kernel.paged;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for the PagedAttention CPU fallback implementation.
 * <p>
 * Verifies correctness of the pure-Java PagedAttention against known
 * mathematical results. These tests run on CPU without requiring
 * a GPU or CUDA toolkit.
 */
class PagedAttentionCpuFallbackTest {

    private Arena arena;

    @BeforeEach
    void setUp() {
        arena = Arena.ofConfined();
    }

    @AfterEach
    void tearDown() {
        arena.close();
    }

    @Test
    @DisplayName("Single sequence, single head, trivial attention")
    void singleSequenceSingleHead() {
        int numSeqs = 1;
        int numHeads = 1;
        int headDim = 2;
        int blockSize = 4;
        int maxBlocksPerSeq = 1;
        int contextLen = 2;

        // Allocate tensors
        MemorySegment output = arena.allocate(ValueLayout.JAVA_FLOAT, numSeqs * numHeads * headDim);
        MemorySegment query = arena.allocate(ValueLayout.JAVA_FLOAT, numSeqs * numHeads * headDim);
        MemorySegment keyCache = arena.allocate(ValueLayout.JAVA_FLOAT, 1 * numHeads * blockSize * headDim);
        MemorySegment valueCache = arena.allocate(ValueLayout.JAVA_FLOAT, 1 * numHeads * blockSize * headDim);
        MemorySegment blockTables = arena.allocate(ValueLayout.JAVA_INT, numSeqs * maxBlocksPerSeq);
        MemorySegment contextLens = arena.allocate(ValueLayout.JAVA_INT, numSeqs);

        // Query = [1.0, 0.0]
        query.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1.0f);
        query.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 0.0f);

        // Key cache: token 0 = [1.0, 0.0], token 1 = [0.0, 1.0]
        keyCache.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1.0f); // block0, head0, tok0, d0
        keyCache.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 0.0f); // block0, head0, tok0, d1
        keyCache.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 0.0f); // block0, head0, tok1, d0
        keyCache.setAtIndex(ValueLayout.JAVA_FLOAT, 3, 1.0f); // block0, head0, tok1, d1

        // Value cache: same values for simplicity
        valueCache.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1.0f);
        valueCache.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 0.0f);
        valueCache.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 0.0f);
        valueCache.setAtIndex(ValueLayout.JAVA_FLOAT, 3, 1.0f);

        // Block table: seq 0 uses physical block 0
        blockTables.setAtIndex(ValueLayout.JAVA_INT, 0, 0);

        // Context length: 2 tokens
        contextLens.setAtIndex(ValueLayout.JAVA_INT, 0, contextLen);

        float scale = 1.0f / (float) Math.sqrt(headDim);

        // Execute
        int result = PagedAttentionCpuFallback.execute(
                output, query, keyCache, valueCache,
                blockTables, contextLens,
                numSeqs, numHeads, headDim, blockSize, maxBlocksPerSeq, scale
        );

        assertThat(result).isEqualTo(0);

        // Q·K with scale: [1*1+0*0, 1*0+0*1] * scale = [0.707, 0.0]
        // After softmax: [e^0.707 / (e^0.707 + e^0), e^0 / (e^0.707 + e^0)]
        //              ≈ [0.669, 0.331]
        // V weighted sum: [0.669*1+0.331*0, 0.669*0+0.331*1] = [0.669, 0.331]
        float out0 = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        float out1 = output.getAtIndex(ValueLayout.JAVA_FLOAT, 1);

        // Verify output is a valid weighted sum
        assertThat(out0).isGreaterThan(0.5f); // Attends more to token 0 (matching key)
        assertThat(out1).isLessThan(0.5f);    // Less attention to token 1
        assertThat(out0 + out1).isCloseTo(1.0f, within(0.001f)); // Weights sum to 1
    }

    @Test
    @DisplayName("Multiple blocks span correctly")
    void multipleBlocks() {
        int numSeqs = 1;
        int numHeads = 1;
        int headDim = 1; // Minimal for simplicity
        int blockSize = 2;
        int maxBlocksPerSeq = 2;
        int contextLen = 3; // Spans 2 blocks

        MemorySegment output = arena.allocate(ValueLayout.JAVA_FLOAT, 1);
        MemorySegment query = arena.allocate(ValueLayout.JAVA_FLOAT, 1);
        MemorySegment keyCache = arena.allocate(ValueLayout.JAVA_FLOAT, 2 * 1 * blockSize * 1);
        MemorySegment valueCache = arena.allocate(ValueLayout.JAVA_FLOAT, 2 * 1 * blockSize * 1);
        MemorySegment blockTables = arena.allocate(ValueLayout.JAVA_INT, 1 * maxBlocksPerSeq);
        MemorySegment contextLens = arena.allocate(ValueLayout.JAVA_INT, 1);

        // Query = [1.0]
        query.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1.0f);

        // K cache block 0: [1.0, 1.0], block 1: [1.0, 0.0]
        keyCache.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1.0f); // block0 tok0
        keyCache.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 1.0f); // block0 tok1
        keyCache.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 1.0f); // block1 tok0
        keyCache.setAtIndex(ValueLayout.JAVA_FLOAT, 3, 0.0f); // block1 tok1 (unused, contextLen=3)

        // V cache: different values to test weighted sum
        valueCache.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 2.0f);
        valueCache.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 4.0f);
        valueCache.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 6.0f);

        // Block table: block 0, then block 1
        blockTables.setAtIndex(ValueLayout.JAVA_INT, 0, 0);
        blockTables.setAtIndex(ValueLayout.JAVA_INT, 1, 1);

        contextLens.setAtIndex(ValueLayout.JAVA_INT, 0, contextLen);

        int result = PagedAttentionCpuFallback.execute(
                output, query, keyCache, valueCache,
                blockTables, contextLens,
                numSeqs, numHeads, headDim, blockSize, maxBlocksPerSeq, 1.0f
        );

        assertThat(result).isEqualTo(0);

        // All K values are 1.0 → all scores equal → uniform softmax → mean of V
        // Mean of [2.0, 4.0, 6.0] = 4.0
        float out = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        assertThat(out).isCloseTo(4.0f, within(0.001f));
    }

    @Test
    @DisplayName("Binding initializes in fallback mode")
    void bindingFallback() {
        PagedAttentionBinding.reset();
        PagedAttentionBinding.initializeFallback();

        PagedAttentionBinding binding = PagedAttentionBinding.getInstance();
        assertThat(binding.isNativeAvailable()).isFalse();
        assertThat(binding.getCudaDeviceCount()).isEqualTo(0);

        PagedAttentionBinding.reset();
    }
}
