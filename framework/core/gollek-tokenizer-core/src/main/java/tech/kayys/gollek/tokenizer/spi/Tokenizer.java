package tech.kayys.gollek.tokenizer.spi;

public interface Tokenizer {

    long[] encode(String text, EncodeOptions options);

    String decode(long[] tokens, DecodeOptions options);

    default String decode(long[] tokens, int offset, int length, DecodeOptions options) {
        java.util.Objects.requireNonNull(tokens, "tokens");
        if (offset < 0 || length < 0 || offset > tokens.length || length > tokens.length - offset) {
            throw new IndexOutOfBoundsException("Token decode range offset=" + offset
                    + " length=" + length + " is outside token count " + tokens.length);
        }
        if (offset == 0 && length == tokens.length) {
            return decode(tokens, options);
        }
        return decode(java.util.Arrays.copyOfRange(tokens, offset, offset + length), options);
    }

    int vocabSize();

    int bosTokenId();

    int eosTokenId();

    int padTokenId();
    
    int[] allStopTokenIds();

    default java.util.Map<String, Integer> specialTokens() {
        return java.util.Collections.emptyMap();
    }
}