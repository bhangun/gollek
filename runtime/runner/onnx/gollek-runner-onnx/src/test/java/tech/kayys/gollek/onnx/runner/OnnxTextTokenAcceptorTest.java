package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class OnnxTextTokenAcceptorTest {

    @Test
    void stopsOnEosWithoutAppendingOrObservingToken() {
        OnnxGeneratedTokens generated = OnnxGeneratedTokens.allocate(4);
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 1, 2 });
        List<Integer> observed = new ArrayList<>();
        OnnxTextTokenAcceptor acceptor = OnnxTextTokenAcceptor.create(
                generated,
                history,
                (tokenId, tokenIndex) -> {
                    observed.add(tokenId);
                    return null;
                },
                null,
                OnnxTextStopStrings.fromParameters(Map.of()),
                tokenId -> tokenId == 99);

        OnnxTextTokenDecision decision = acceptor.accept(99);

        assertTrue(decision.finished());
        assertEquals(OnnxTextFinishReason.STOP, decision.finishReason());
        assertEquals(0, generated.size());
        assertEquals(2, history.size());
        assertEquals(List.of(), observed);
    }

    @Test
    void appendsObservedTokenAndContinuesWhenNoStopMatches() {
        OnnxGeneratedTokens generated = OnnxGeneratedTokens.allocate(1);
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 10, 20 });
        List<String> observed = new ArrayList<>();
        List<Boolean> currentTextNeeded = new ArrayList<>();
        OnnxTextTokenAcceptor acceptor = OnnxTextTokenAcceptor.create(
                generated,
                history,
                new OnnxGeneratedTokenObserver() {
                    @Override
                    public String onToken(int tokenId, int tokenIndex) {
                        return onToken(tokenId, tokenIndex, true);
                    }

                    @Override
                    public String onToken(int tokenId, int tokenIndex, boolean textNeeded) {
                        observed.add(tokenId + "@" + tokenIndex);
                        currentTextNeeded.add(textNeeded);
                        return textNeeded ? "unused" : null;
                    }
                },
                null,
                OnnxTextStopStrings.fromParameters(Map.of()),
                tokenId -> false);

        OnnxTextTokenDecision decision = acceptor.accept(30);

        assertFalse(decision.finished());
        assertEquals(OnnxTextFinishReason.LENGTH, decision.finishReason());
        assertEquals(1, generated.size());
        assertEquals(3, history.size());
        assertEquals(30, history.last());
        assertEquals(List.of("30@0"), observed);
        assertEquals(List.of(false), currentTextNeeded);
    }

    @Test
    void matchesStopStringFromObserverTextWithoutFallbackDecode() {
        List<Boolean> currentTextNeeded = new ArrayList<>();
        OnnxTextTokenAcceptor acceptor = OnnxTextTokenAcceptor.create(
                OnnxGeneratedTokens.allocate(2),
                OnnxTokenHistory.from(new int[] { 1 }),
                new OnnxGeneratedTokenObserver() {
                    @Override
                    public String onToken(int tokenId, int tokenIndex) {
                        return onToken(tokenId, tokenIndex, true);
                    }

                    @Override
                    public String onToken(int tokenId, int tokenIndex, boolean textNeeded) {
                        currentTextNeeded.add(textNeeded);
                        return "hello";
                    }
                },
                OnnxStreamingTokenDecoder.create(null, tokenId -> {
                    throw new AssertionError("fallback decoder should not be used when observer provides text");
                }),
                OnnxTextStopStrings.fromParameters(Map.of("stop", "hello")),
                tokenId -> false);

        OnnxTextTokenDecision decision = acceptor.accept(7);

        assertTrue(decision.finished());
        assertEquals(OnnxTextFinishReason.STOP, decision.finishReason());
        assertEquals(List.of(true), currentTextNeeded);
    }

    @Test
    void matchesStopStringWithFallbackDecoderText() {
        OnnxGeneratedTokens generated = OnnxGeneratedTokens.allocate(2);
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 1 });
        OnnxTextTokenAcceptor acceptor = OnnxTextTokenAcceptor.create(
                generated,
                history,
                OnnxGeneratedTokenObserver.NOOP,
                OnnxStreamingTokenDecoder.create(null, tokenId -> tokenId == 2 ? "i" : "h"),
                OnnxTextStopStrings.fromParameters(Map.of("stop", "hi")),
                tokenId -> false);

        assertFalse(acceptor.accept(1).finished());
        OnnxTextTokenDecision decision = acceptor.accept(2);

        assertTrue(decision.finished());
        assertEquals(2, generated.size());
        assertEquals(2, history.last());
    }

    @Test
    void matchesStopStringUsingBoundedFallbackSuffix() {
        OnnxTextTokenAcceptor acceptor = OnnxTextTokenAcceptor.create(
                OnnxGeneratedTokens.allocate(4),
                OnnxTokenHistory.from(new int[] { 1 }),
                OnnxGeneratedTokenObserver.NOOP,
                OnnxStreamingTokenDecoder.create(null, tokenId -> switch (tokenId) {
                    case 1 -> "long generated prefix ";
                    case 2 -> "almost ";
                    default -> "END";
                }),
                OnnxTextStopStrings.fromParameters(Map.of("stop", "END")),
                tokenId -> false);

        assertFalse(acceptor.accept(1).finished());
        assertFalse(acceptor.accept(2).finished());
        OnnxTextTokenDecision decision = acceptor.accept(3);

        assertTrue(decision.finished());
        assertEquals(OnnxTextFinishReason.STOP, decision.finishReason());
    }

    @Test
    void validatesRequiredArguments() {
        OnnxGeneratedTokens generated = OnnxGeneratedTokens.allocate(1);
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 1 });
        OnnxTextStopStrings stops = OnnxTextStopStrings.fromParameters(Map.of());

        assertThrows(NullPointerException.class,
                () -> OnnxTextTokenAcceptor.create(null, history, null, null, stops, tokenId -> false));
        assertThrows(NullPointerException.class,
                () -> OnnxTextTokenAcceptor.create(generated, null, null, null, stops, tokenId -> false));
        assertThrows(NullPointerException.class,
                () -> OnnxTextTokenAcceptor.create(generated, history, null, null, null, tokenId -> false));
        assertThrows(NullPointerException.class,
                () -> OnnxTextTokenAcceptor.create(generated, history, null, null, stops, null));
    }
}
