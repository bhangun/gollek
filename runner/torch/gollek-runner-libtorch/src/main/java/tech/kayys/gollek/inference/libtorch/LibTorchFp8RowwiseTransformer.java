package tech.kayys.gollek.inference.libtorch;

/**
 * Applies FP8 rowwise calibration scales to logits.
 */
public final class LibTorchFp8RowwiseTransformer {

    public boolean canApply(int vocabSize, float[] rowScales) {
        return vocabSize > 0
                && rowScales != null
                && rowScales.length == vocabSize;
    }

    public void applyInPlace(float[] logits, float[] rowScales) {
        if (logits == null || rowScales == null) {
            return;
        }
        int n = Math.min(logits.length, rowScales.length);
        for (int i = 0; i < n; i++) {
            logits[i] = logits[i] * rowScales[i];
        }
    }
}
