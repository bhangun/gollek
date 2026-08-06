package tech.kayys.gollek.onnx.runner;

import java.util.Objects;
import java.util.function.IntFunction;

import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.StreamingDecoder;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

final class OnnxStreamingTokenDecoder {

    private final StreamingDecoder streamingDecoder;
    private final IntFunction<String> fallbackDecoder;
    private final StringBuilder fallbackText;

    private OnnxStreamingTokenDecoder(
            StreamingDecoder streamingDecoder,
            IntFunction<String> fallbackDecoder,
            StringBuilder fallbackText) {
        this.streamingDecoder = streamingDecoder;
        this.fallbackDecoder = fallbackDecoder;
        this.fallbackText = fallbackText;
    }

    static OnnxStreamingTokenDecoder create(Tokenizer tokenizer, IntFunction<String> fallbackDecoder) {
        if (tokenizer != null) {
            return new OnnxStreamingTokenDecoder(
                    new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions()),
                    null,
                    null);
        }
        return new OnnxStreamingTokenDecoder(
                null,
                Objects.requireNonNull(fallbackDecoder, "fallbackDecoder"),
                new StringBuilder());
    }

    String decodeNext(int tokenId) {
        if (streamingDecoder != null) {
            String delta = streamingDecoder.decodeNext(tokenId);
            return delta == null ? "" : delta;
        }
        String delta = fallbackDecoder.apply(tokenId);
        if (delta == null) {
            return "";
        }
        fallbackText.append(delta);
        return delta;
    }

    String currentText() {
        return streamingDecoder == null ? fallbackText.toString() : streamingDecoder.currentText();
    }

    String currentSuffix(int maxChars) {
        if (maxChars <= 0) {
            return "";
        }
        if (streamingDecoder != null) {
            return suffix(streamingDecoder.currentText(), maxChars);
        }
        int length = fallbackText.length();
        if (length <= maxChars) {
            return fallbackText.toString();
        }
        return fallbackText.substring(length - maxChars);
    }

    private static String suffix(String text, int maxChars) {
        if (text == null || text.length() <= maxChars) {
            return text == null ? "" : text;
        }
        return text.substring(text.length() - maxChars);
    }
}
