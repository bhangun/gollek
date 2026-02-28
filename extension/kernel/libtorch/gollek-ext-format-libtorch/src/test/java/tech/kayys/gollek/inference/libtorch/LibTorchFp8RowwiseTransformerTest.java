package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class LibTorchFp8RowwiseTransformerTest {

    @Test
    void appliesScalesInPlace() {
        LibTorchFp8RowwiseTransformer transformer = new LibTorchFp8RowwiseTransformer();
        float[] logits = new float[] { 2.0f, 4.0f, 8.0f };
        float[] scales = new float[] { 0.5f, 0.25f, 0.125f };
        transformer.applyInPlace(logits, scales);
        assertThat(logits).containsExactly(1.0f, 1.0f, 1.0f);
    }

    @Test
    void checksScaleCompatibility() {
        LibTorchFp8RowwiseTransformer transformer = new LibTorchFp8RowwiseTransformer();
        assertThat(transformer.canApply(3, new float[] { 1f, 1f, 1f })).isTrue();
        assertThat(transformer.canApply(3, new float[] { 1f })).isFalse();
        assertThat(transformer.canApply(0, new float[] { 1f })).isFalse();
    }
}
