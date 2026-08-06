package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.lang.reflect.Field;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

class OnnxRuntimeRunnerDecodeTest {

    @Test
    void detokenizeUsesRangeDecodeWithThreadLocalScratch() throws Exception {
        ExposedRunner runner = new ExposedRunner();
        RangeTrackingTokenizer tokenizer = new RangeTrackingTokenizer();
        setTokenizer(runner, tokenizer);

        assertEquals("42", runner.decodeToken(42));
        Object firstScratch = tokenizer.lastTokens;
        assertEquals("7", runner.decodeToken(7));

        assertEquals(0, tokenizer.fullDecodeCalls);
        assertEquals(2, tokenizer.rangeDecodeCalls);
        assertSame(firstScratch, tokenizer.lastTokens);
    }

    private static void setTokenizer(OnnxRuntimeRunner runner, Tokenizer tokenizer) throws Exception {
        Field field = OnnxRuntimeRunner.class.getDeclaredField("tokenizer");
        field.setAccessible(true);
        field.set(runner, tokenizer);
    }

    private static final class ExposedRunner extends OnnxRuntimeRunner {
        String decodeToken(int tokenId) {
            return detokenize(tokenId);
        }
    }

    private static final class RangeTrackingTokenizer implements Tokenizer {
        private int fullDecodeCalls;
        private int rangeDecodeCalls;
        private long[] lastTokens;

        @Override
        public long[] encode(String text, EncodeOptions options) {
            return new long[0];
        }

        @Override
        public String decode(long[] tokens, DecodeOptions options) {
            fullDecodeCalls++;
            throw new AssertionError("Single-token decode should use ranged decode");
        }

        @Override
        public String decode(long[] tokens, int offset, int length, DecodeOptions options) {
            rangeDecodeCalls++;
            lastTokens = tokens;
            return Long.toString(tokens[offset]);
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
