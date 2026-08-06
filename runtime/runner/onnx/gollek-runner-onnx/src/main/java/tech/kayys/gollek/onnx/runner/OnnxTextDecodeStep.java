package tech.kayys.gollek.onnx.runner;

final class OnnxTextDecodeStep {

    private boolean prefill;
    private long sequenceLength;
    private long attentionLength;
    private long positionStart;

    OnnxTextDecodeStep(
            boolean prefill,
            long sequenceLength,
            long attentionLength,
            long positionStart) {
        this.prefill = prefill;
        this.sequenceLength = sequenceLength;
        this.attentionLength = attentionLength;
        this.positionStart = positionStart;
    }

    static OnnxTextDecodeStep reusable() {
        return new OnnxTextDecodeStep(false, 1L, 1L, 0L);
    }

    static OnnxTextDecodeStep plan(
            boolean hasKvInputs,
            boolean hasCurrentKv,
            int promptLength,
            int tokenHistorySize,
            long consumedTokens) {
        return reusable().update(hasKvInputs, hasCurrentKv, promptLength, tokenHistorySize, consumedTokens);
    }

    OnnxTextDecodeStep update(
            boolean hasKvInputs,
            boolean hasCurrentKv,
            int promptLength,
            int tokenHistorySize,
            long consumedTokens) {
        if (promptLength <= 0) {
            throw new IllegalArgumentException("promptLength must be > 0");
        }
        if (tokenHistorySize <= 0) {
            throw new IllegalArgumentException("tokenHistorySize must be > 0");
        }
        if (consumedTokens < 0) {
            throw new IllegalArgumentException("consumedTokens must be >= 0");
        }

        prefill = hasKvInputs && !hasCurrentKv;
        if (!hasKvInputs) {
            sequenceLength = tokenHistorySize;
            attentionLength = tokenHistorySize;
            positionStart = 0L;
            return this;
        }

        sequenceLength = prefill ? promptLength : 1L;
        attentionLength = consumedTokens + sequenceLength;
        positionStart = consumedTokens;
        return this;
    }

    boolean prefill() {
        return prefill;
    }

    long sequenceLength() {
        return sequenceLength;
    }

    long attentionLength() {
        return attentionLength;
    }

    long positionStart() {
        return positionStart;
    }
}
