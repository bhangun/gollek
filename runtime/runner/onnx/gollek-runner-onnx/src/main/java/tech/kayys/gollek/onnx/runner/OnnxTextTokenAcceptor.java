package tech.kayys.gollek.onnx.runner;

import java.util.Objects;
import java.util.function.IntPredicate;

final class OnnxTextTokenAcceptor {

    private final OnnxGeneratedTokens generatedTokens;
    private final OnnxTokenHistory tokenHistory;
    private final OnnxGeneratedTokenObserver observer;
    private final OnnxStreamingTokenDecoder stopStringDecoder;
    private final OnnxTextStopStrings stopStrings;
    private final IntPredicate eosMatcher;

    private OnnxTextTokenAcceptor(
            OnnxGeneratedTokens generatedTokens,
            OnnxTokenHistory tokenHistory,
            OnnxGeneratedTokenObserver observer,
            OnnxStreamingTokenDecoder stopStringDecoder,
            OnnxTextStopStrings stopStrings,
            IntPredicate eosMatcher) {
        this.generatedTokens = Objects.requireNonNull(generatedTokens, "generatedTokens");
        this.tokenHistory = Objects.requireNonNull(tokenHistory, "tokenHistory");
        this.observer = observer == null ? OnnxGeneratedTokenObserver.NOOP : observer;
        this.stopStringDecoder = stopStringDecoder;
        this.stopStrings = Objects.requireNonNull(stopStrings, "stopStrings");
        this.eosMatcher = Objects.requireNonNull(eosMatcher, "eosMatcher");
    }

    static OnnxTextTokenAcceptor create(
            OnnxGeneratedTokens generatedTokens,
            OnnxTokenHistory tokenHistory,
            OnnxGeneratedTokenObserver observer,
            OnnxStreamingTokenDecoder stopStringDecoder,
            OnnxTextStopStrings stopStrings,
            IntPredicate eosMatcher) {
        return new OnnxTextTokenAcceptor(
                generatedTokens,
                tokenHistory,
                observer,
                stopStringDecoder,
                stopStrings,
                eosMatcher);
    }

    OnnxTextTokenDecision accept(int tokenId) {
        if (eosMatcher.test(tokenId)) {
            return OnnxTextTokenDecision.stop(OnnxTextFinishReason.STOP);
        }

        generatedTokens.append(tokenId);
        String observedText = observer.onToken(tokenId, generatedTokens.size() - 1, !stopStrings.isEmpty());
        tokenHistory.append(tokenId);
        if (matchesStopString(tokenId, observedText)) {
            return OnnxTextTokenDecision.stop(OnnxTextFinishReason.STOP);
        }
        return OnnxTextTokenDecision.continueGeneration();
    }

    private boolean matchesStopString(int tokenId, String observedText) {
        if (stopStringDecoder == null || stopStrings.isEmpty()) {
            return false;
        }
        String currentText = observedText;
        if (currentText == null) {
            stopStringDecoder.decodeNext(tokenId);
            currentText = stopStringDecoder.currentSuffix(stopStrings.maxLength());
        }
        return stopStrings.matches(currentText);
    }
}
