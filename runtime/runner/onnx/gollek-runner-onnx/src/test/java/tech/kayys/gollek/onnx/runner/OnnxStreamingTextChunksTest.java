package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

class OnnxStreamingTextChunksTest {

    @Test
    void emitsNonEmptyDeltasInOrderAndFinalUsage() {
        List<StreamingInferenceChunk> emitted = new ArrayList<>();
        OnnxStreamingTextChunks chunks = OnnxStreamingTextChunks.create(
                "req-1",
                OnnxStreamingTokenDecoder.create(null, tokenId -> tokenId == 2 ? "" : "<" + tokenId + ">"),
                emitted::add);

        chunks.emitToken(1);
        chunks.emitToken(2);
        chunks.emitToken(3);
        chunks.emitFinal(new OnnxTextGenerationResult(
                "req-1", "<1><3>", 5, 3, 42L, null, false, OnnxTextFinishReason.STOP));

        assertEquals(3, emitted.size());
        assertEquals(0, emitted.get(0).index());
        assertEquals("<1>", emitted.get(0).delta());
        assertFalse(emitted.get(0).finished());
        assertEquals(1, emitted.get(1).index());
        assertEquals("<3>", emitted.get(1).delta());
        assertFalse(emitted.get(1).finished());
        assertEquals(2, emitted.get(2).index());
        assertEquals("", emitted.get(2).delta());
        assertTrue(emitted.get(2).finished());
        assertEquals("stop", emitted.get(2).finishReason());
        assertEquals(5L, emitted.get(2).usage().inputTokens());
        assertEquals(3L, emitted.get(2).usage().outputTokens());
        assertEquals(42L, emitted.get(2).usage().latencyMs());
        assertEquals(2, chunks.emittedDeltas());
    }

    @Test
    void finalChunkUsesZeroIndexWhenNoTextDeltaWasEmitted() {
        List<StreamingInferenceChunk> emitted = new ArrayList<>();
        OnnxStreamingTextChunks chunks = OnnxStreamingTextChunks.create(
                "req-empty",
                OnnxStreamingTokenDecoder.create(null, ignored -> ""),
                emitted::add);

        chunks.emitToken(9);
        chunks.emitFinal(new OnnxTextGenerationResult(
                "req-empty", "", 1, 0, 7L, null, false, OnnxTextFinishReason.LENGTH));

        assertEquals(1, emitted.size());
        assertEquals(0, emitted.get(0).index());
        assertTrue(emitted.get(0).finished());
        assertEquals("length", emitted.get(0).finishReason());
        assertEquals(0, chunks.emittedDeltas());
    }

    @Test
    void currentTextTracksDecodedTokensForStopMatching() {
        OnnxStreamingTextChunks chunks = OnnxStreamingTextChunks.create(
                "req-stop",
                OnnxStreamingTokenDecoder.create(null, tokenId -> tokenId == 2 ? "END" : "hello "),
                ignored -> {
                });

        assertTrue(chunks.emitToken(1));
        assertEquals("hello ", chunks.currentText());
        assertTrue(chunks.emitToken(2));
        assertEquals("hello END", chunks.currentText());
    }

    @Test
    void cancellationSuppressesFurtherDeltasAndFinalChunk() {
        List<StreamingInferenceChunk> emitted = new ArrayList<>();
        AtomicBoolean cancelled = new AtomicBoolean(false);
        AtomicInteger decodeCount = new AtomicInteger();
        OnnxStreamingTextChunks chunks = OnnxStreamingTextChunks.create(
                "req-cancel",
                OnnxStreamingTokenDecoder.create(null, tokenId -> {
                    decodeCount.incrementAndGet();
                    return "<" + tokenId + ">";
                }),
                emitted::add,
                cancelled::get);

        assertTrue(chunks.emitToken(1));
        cancelled.set(true);

        assertFalse(chunks.emitToken(2));
        assertFalse(chunks.emitFinal(new OnnxTextGenerationResult(
                "req-cancel", "<1>", 1, 1, 9L, null, false, OnnxTextFinishReason.CANCELLED)));

        assertEquals(1, emitted.size());
        assertEquals("<1>", emitted.get(0).delta());
        assertEquals(1, decodeCount.get());
        assertEquals(1, chunks.emittedDeltas());
    }
}
