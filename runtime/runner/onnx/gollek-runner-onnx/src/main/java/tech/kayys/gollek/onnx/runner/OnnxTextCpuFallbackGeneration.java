package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeCpuFallback;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.IntFunction;
import java.util.function.ToIntFunction;

final class OnnxTextCpuFallbackGeneration {

    private final Ops ops;
    private final int vocabSize;
    private final ToIntFunction<float[]> sampler;
    private final IntFunction<String> decoder;

    private OnnxTextCpuFallbackGeneration(
            Ops ops,
            int vocabSize,
            ToIntFunction<float[]> sampler,
            IntFunction<String> decoder) {
        this.ops = Objects.requireNonNull(ops, "ops");
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("vocabSize must be > 0");
        }
        this.vocabSize = vocabSize;
        this.sampler = Objects.requireNonNull(sampler, "sampler");
        this.decoder = Objects.requireNonNull(decoder, "decoder");
    }

    static OnnxTextCpuFallbackGeneration create(
            int vocabSize,
            ToIntFunction<float[]> sampler,
            IntFunction<String> decoder) {
        return new OnnxTextCpuFallbackGeneration(new BindingOps(), vocabSize, sampler, decoder);
    }

    static OnnxTextCpuFallbackGeneration createForTest(
            Ops ops,
            int vocabSize,
            ToIntFunction<float[]> sampler,
            IntFunction<String> decoder) {
        return new OnnxTextCpuFallbackGeneration(ops, vocabSize, sampler, decoder);
    }

    OnnxTextGenerationResult generate(OnnxTextGenerationSetup setup, long requestStartMillis) {
        Objects.requireNonNull(setup, "setup");
        float[] logits = ops.run(vocabSize);
        int next = sampler.applyAsInt(logits);
        setup.observer().onToken(next, 0);
        String content = setup.finalContentOrNull();
        if (content == null) {
            content = decoder.apply(next);
        }
        return new OnnxTextGenerationResult(
                setup.requestId(),
                content,
                setup.promptLength(),
                1,
                Math.max(0L, System.currentTimeMillis() - requestStartMillis),
                setup.profile(),
                true,
                OnnxTextFinishReason.STOP);
    }

    interface Ops {
        float[] run(int outputFloats);
    }

    private static final class BindingOps implements Ops {
        @Override
        public float[] run(int outputFloats) {
            return OnnxRuntimeCpuFallback.run(MemorySegment.NULL, outputFloats);
        }
    }
}
