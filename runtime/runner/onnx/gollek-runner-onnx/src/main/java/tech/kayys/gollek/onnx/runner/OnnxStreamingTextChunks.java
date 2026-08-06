package tech.kayys.gollek.onnx.runner;

import java.util.Objects;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

final class OnnxStreamingTextChunks {

    private final String requestId;
    private final OnnxStreamingTokenDecoder decoder;
    private final Consumer<StreamingInferenceChunk> sink;
    private final BooleanSupplier cancelled;
    private int nextIndex;

    private OnnxStreamingTextChunks(
            String requestId,
            OnnxStreamingTokenDecoder decoder,
            Consumer<StreamingInferenceChunk> sink,
            BooleanSupplier cancelled) {
        this.requestId = requestId == null ? "" : requestId;
        this.decoder = Objects.requireNonNull(decoder, "decoder");
        this.sink = Objects.requireNonNull(sink, "sink");
        this.cancelled = Objects.requireNonNull(cancelled, "cancelled");
    }

    static OnnxStreamingTextChunks create(
            String requestId,
            OnnxStreamingTokenDecoder decoder,
            Consumer<StreamingInferenceChunk> sink) {
        return create(requestId, decoder, sink, () -> false);
    }

    static OnnxStreamingTextChunks create(
            String requestId,
            OnnxStreamingTokenDecoder decoder,
            Consumer<StreamingInferenceChunk> sink,
            BooleanSupplier cancelled) {
        return new OnnxStreamingTextChunks(requestId, decoder, sink, cancelled);
    }

    boolean emitToken(int tokenId) {
        if (cancelled.getAsBoolean()) {
            return false;
        }
        String delta = decoder.decodeNext(tokenId);
        if (delta.isEmpty()) {
            return false;
        }
        sink.accept(StreamingInferenceChunk.textDelta(requestId, nextIndex++, delta));
        return true;
    }

    boolean emitFinal(OnnxTextGenerationResult generation) {
        Objects.requireNonNull(generation, "generation");
        if (cancelled.getAsBoolean()) {
            return false;
        }
        sink.accept(new StreamingInferenceChunk(
                requestId,
                nextIndex,
                tech.kayys.alkhawarizm.spi.model.ModalityType.TEXT,
                "",
                null,
                true,
                generation.finishReason().wireValue(),
                new StreamingInferenceChunk.ChunkUsage(
                        generation.inputTokens(),
                        generation.outputTokens(),
                        generation.durationMs()),
                java.time.Instant.now(),
                null,
                                    null,
                                    null,
                                    null));
        return true;
    }

    int emittedDeltas() {
        return nextIndex;
    }

    String currentText() {
        return decoder.currentText();
    }
}
