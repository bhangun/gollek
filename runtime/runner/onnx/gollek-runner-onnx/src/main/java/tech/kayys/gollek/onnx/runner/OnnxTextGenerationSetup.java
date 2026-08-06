package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.spi.inference.InferenceRequest;

import java.util.Objects;
import java.util.function.BooleanSupplier;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.ToIntFunction;

record OnnxTextGenerationSetup(
        String requestId,
        OnnxGeneratedTokenObserver observer,
        BooleanSupplier cancellation,
        OnnxInferenceProfile profile,
        int maxTokens,
        int[] prompt,
        OnnxTextStopStrings stopStrings,
        OnnxStreamingTokenDecoder stopStringDecoder,
        Supplier<String> finalContentSupplier) {

    static OnnxTextGenerationSetup prepare(
            InferenceRequest request,
            OnnxGeneratedTokenObserver observer,
            BooleanSupplier cancellation,
            OnnxInferenceProfile profile,
            ToIntFunction<InferenceRequest> maxTokensResolver,
            Function<InferenceRequest, int[]> promptTokenizer,
            Supplier<OnnxStreamingTokenDecoder> stopDecoderFactory) {
        return prepare(
                request,
                observer,
                cancellation,
                profile,
                maxTokensResolver,
                promptTokenizer,
                stopDecoderFactory,
                null);
    }

    static OnnxTextGenerationSetup prepare(
            InferenceRequest request,
            OnnxGeneratedTokenObserver observer,
            BooleanSupplier cancellation,
            OnnxInferenceProfile profile,
            ToIntFunction<InferenceRequest> maxTokensResolver,
            Function<InferenceRequest, int[]> promptTokenizer,
            Supplier<OnnxStreamingTokenDecoder> stopDecoderFactory,
            Supplier<String> finalContentSupplier) {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(profile, "profile");
        Objects.requireNonNull(maxTokensResolver, "maxTokensResolver");
        Objects.requireNonNull(promptTokenizer, "promptTokenizer");
        Objects.requireNonNull(stopDecoderFactory, "stopDecoderFactory");

        long tokenizeStart = profile.mark();
        int maxTokens = maxTokensResolver.applyAsInt(request);
        int[] prompt = promptTokenizer.apply(request);
        OnnxTextStopStrings stopStrings = OnnxTextStopStrings.from(request);
        OnnxStreamingTokenDecoder stopStringDecoder = stopStrings.isEmpty() ? null : stopDecoderFactory.get();
        profile.recordTokenize(tokenizeStart);

        return new OnnxTextGenerationSetup(
                request.getRequestId(),
                observer == null ? OnnxGeneratedTokenObserver.NOOP : observer,
                cancellation == null ? () -> false : cancellation,
                profile,
                maxTokens,
                prompt,
                stopStrings,
                stopStringDecoder,
                finalContentSupplier);
    }

    OnnxTextGenerationSetup {
        observer = observer == null ? OnnxGeneratedTokenObserver.NOOP : observer;
        cancellation = cancellation == null ? () -> false : cancellation;
        profile = Objects.requireNonNull(profile, "profile");
        prompt = Objects.requireNonNull(prompt, "prompt");
        stopStrings = Objects.requireNonNull(stopStrings, "stopStrings");
        if (maxTokens < 0) {
            throw new IllegalArgumentException("maxTokens must be >= 0");
        }
        if (prompt.length == 0) {
            throw new IllegalArgumentException("prompt must contain at least one token");
        }
    }

    int promptLength() {
        return prompt.length;
    }

    String finalContentOrNull() {
        if (finalContentSupplier == null) {
            return null;
        }
        return finalContentSupplier.get();
    }
}
