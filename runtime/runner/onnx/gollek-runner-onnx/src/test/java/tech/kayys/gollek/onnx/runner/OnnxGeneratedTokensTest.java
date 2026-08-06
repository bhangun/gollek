package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

class OnnxGeneratedTokensTest {

    @Test
    void appendsAndExportsLongArray() {
        OnnxGeneratedTokens tokens = OnnxGeneratedTokens.allocate(1);

        assertTrue(tokens.isEmpty());

        tokens.append(10);
        tokens.append(20);
        tokens.append(30);

        assertFalse(tokens.isEmpty());
        assertEquals(3, tokens.size());
        assertArrayEquals(new long[] { 10L, 20L, 30L }, tokens.toLongArray());
    }

    @Test
    void decodesEachTokenForFallbackTokenizerPath() {
        OnnxGeneratedTokens tokens = OnnxGeneratedTokens.allocate(2);

        tokens.append(3);
        tokens.append(4);

        assertEquals("<3><4>", tokens.decodeEach(tokenId -> "<" + tokenId + ">"));
    }

    @Test
    void decodesWithTokenizerUsingReusableLongScratch() {
        OnnxGeneratedTokens tokens = OnnxGeneratedTokens.allocate(4);
        RangeDecodeTokenizer tokenizer = new RangeDecodeTokenizer();

        tokens.append(10);
        tokens.append(20);

        assertEquals("10,20", tokens.decodeWith(tokenizer, DecodeOptions.defaultOptions()));
        assertEquals(1, tokenizer.rangeDecodeCalls);
        assertEquals(0, tokenizer.fullDecodeCalls);
        assertTrue(tokenizer.sawSpareCapacity);

        tokens.reset(2);
        tokens.append(30);

        assertEquals("30", tokens.decodeWith(tokenizer, DecodeOptions.defaultOptions()));
        assertEquals(2, tokenizer.rangeDecodeCalls);
        assertEquals(0, tokenizer.fullDecodeCalls);
    }

    @Test
    void resetsForWorkspaceReuse() {
        OnnxGeneratedTokens tokens = OnnxGeneratedTokens.allocate(1);
        tokens.append(10);
        tokens.append(20);

        OnnxGeneratedTokens reset = tokens.reset(4);

        assertSame(tokens, reset);
        assertTrue(tokens.isEmpty());
        tokens.append(30);
        assertArrayEquals(new long[] { 30L }, tokens.toLongArray());
    }

    @Test
    void rejectsNegativeInitialCapacity() {
        assertThrows(IllegalArgumentException.class, () -> OnnxGeneratedTokens.allocate(-1));
        assertThrows(IllegalArgumentException.class, () -> OnnxGeneratedTokens.allocate(1).reset(-1));
    }

    private static final class RangeDecodeTokenizer implements Tokenizer {
        private int fullDecodeCalls;
        private int rangeDecodeCalls;
        private boolean sawSpareCapacity;

        @Override
        public long[] encode(String text, EncodeOptions options) {
            return new long[0];
        }

        @Override
        public String decode(long[] tokens, DecodeOptions options) {
            fullDecodeCalls++;
            throw new AssertionError("Generated token decode should use ranged decode");
        }

        @Override
        public String decode(long[] tokens, int offset, int length, DecodeOptions options) {
            rangeDecodeCalls++;
            sawSpareCapacity |= tokens.length > length;
            StringBuilder text = new StringBuilder();
            int end = offset + length;
            for (int i = offset; i < end; i++) {
                if (text.length() > 0) {
                    text.append(',');
                }
                text.append(tokens[i]);
            }
            return text.toString();
        }

        @Override
        public int vocabSize() {
            return 100;
        }

        @Override
        public int bosTokenId() {
            return -1;
        }

        @Override
        public int eosTokenId() {
            return -1;
        }

        @Override
        public int padTokenId() {
            return -1;
        }

        @Override
        public int[] allStopTokenIds() {
            return new int[0];
        }
    }
}
