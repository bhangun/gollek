package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

class OnnxStreamingTokenDecoderTest {

    @Test
    void usesTokenizerStreamingDecoderForIncrementalDeltas() {
        OnnxStreamingTokenDecoder decoder = OnnxStreamingTokenDecoder.create(new JoiningTokenizer(), ignored -> {
            throw new AssertionError("Fallback decoder should not be used when tokenizer is available");
        });

        assertEquals("H", decoder.decodeNext(1));
        assertEquals("ello", decoder.decodeNext(2));
        assertEquals("Hello", decoder.currentText());
    }

    @Test
    void usesFallbackDecoderWhenTokenizerIsUnavailable() {
        OnnxStreamingTokenDecoder decoder = OnnxStreamingTokenDecoder.create(null, tokenId -> "<" + tokenId + ">");

        assertEquals("<7>", decoder.decodeNext(7));
        assertEquals("<8>", decoder.decodeNext(8));
        assertEquals("<7><8>", decoder.currentText());
        assertEquals("<8>", decoder.currentSuffix(3));
    }

    @Test
    void normalizesNullDeltasToEmptyStrings() {
        OnnxStreamingTokenDecoder decoder = OnnxStreamingTokenDecoder.create(null, ignored -> null);

        assertEquals("", decoder.decodeNext(9));
        assertEquals("", decoder.currentText());
        assertEquals("", decoder.currentSuffix(3));
    }

    @Test
    void returnsBoundedSuffixForTokenizerBackedDecoder() {
        OnnxStreamingTokenDecoder decoder = OnnxStreamingTokenDecoder.create(new JoiningTokenizer(), ignored -> {
            throw new AssertionError("Fallback decoder should not be used when tokenizer is available");
        });

        decoder.decodeNext(1);
        decoder.decodeNext(2);

        assertEquals("llo", decoder.currentSuffix(3));
        assertEquals("", decoder.currentSuffix(0));
    }

    @Test
    void tokenizerStreamingDecoderUsesRangeDecodeWithoutCopyingTokenPrefix() {
        RangeTrackingTokenizer tokenizer = new RangeTrackingTokenizer();
        OnnxStreamingTokenDecoder decoder = OnnxStreamingTokenDecoder.create(tokenizer, ignored -> {
            throw new AssertionError("Fallback decoder should not be used when tokenizer is available");
        });

        assertEquals("H", decoder.decodeNext(1));
        assertEquals("ello", decoder.decodeNext(2));

        assertEquals(0, tokenizer.fullDecodeCalls);
        assertEquals(2, tokenizer.rangeDecodeCalls);
        assertTrue(tokenizer.sawReusableBufferWithSpareCapacity);
    }

    private static final class JoiningTokenizer implements Tokenizer {

        @Override
        public long[] encode(String text, EncodeOptions options) {
            return new long[0];
        }

        @Override
        public String decode(long[] tokens, DecodeOptions options) {
            StringBuilder text = new StringBuilder();
            for (long token : tokens) {
                if (token == 1L) {
                    text.append("H");
                } else if (token == 2L) {
                    text.append("ello");
                }
            }
            return text.toString();
        }

        @Override
        public int vocabSize() {
            return 3;
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

    private static final class RangeTrackingTokenizer implements Tokenizer {
        private int fullDecodeCalls;
        private int rangeDecodeCalls;
        private boolean sawReusableBufferWithSpareCapacity;

        @Override
        public long[] encode(String text, EncodeOptions options) {
            return new long[0];
        }

        @Override
        public String decode(long[] tokens, DecodeOptions options) {
            fullDecodeCalls++;
            throw new AssertionError("Streaming decode should use ranged decode");
        }

        @Override
        public String decode(long[] tokens, int offset, int length, DecodeOptions options) {
            rangeDecodeCalls++;
            sawReusableBufferWithSpareCapacity |= tokens.length > length;
            StringBuilder text = new StringBuilder();
            int end = offset + length;
            for (int i = offset; i < end; i++) {
                if (tokens[i] == 1L) {
                    text.append("H");
                } else if (tokens[i] == 2L) {
                    text.append("ello");
                }
            }
            return text.toString();
        }

        @Override
        public int vocabSize() {
            return 3;
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
