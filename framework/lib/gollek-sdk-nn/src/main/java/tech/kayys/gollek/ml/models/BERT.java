package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;

/**
 * BERT (Bidirectional Encoder Representations from Transformers) model family.
 *
 * <p>Implements the encoder-only Transformer architecture from
 * <em>"BERT: Pre-training of Deep Bidirectional Transformers"</em> (Devlin et al., 2018).
 *
 * <p>Available variants:
 * <ul>
 *   <li>{@link #bertBase(int)} — 12 layers, 768 hidden, 12 heads, ~110M params</li>
 *   <li>{@link #bertLarge(int)} — 24 layers, 1024 hidden, 16 heads, ~340M params</li>
 *   <li>{@link #forSequenceClassification(int, int)} — BERT-base + classification head</li>
 * </ul>
 *
 * <h3>Example — fine-tuning for classification</h3>
 * <pre>{@code
 * NNModule model = BERT.forSequenceClassification(numClasses = 2, vocabSize = 30522);
 * GradTensor logits = model.forward(embeddings); // [B, T, 768] → [B, numClasses]
 * }</pre>
 */
public final class BERT {

    /** BERT-base vocabulary size (WordPiece). */
    public static final int VOCAB_SIZE_BASE  = 30522;

    /** BERT-large vocabulary size. */
    public static final int VOCAB_SIZE_LARGE = 30522;

    private BERT() {}

    /**
     * Constructs a BERT-base encoder (12 layers, dModel=768, 12 heads).
     *
     * @param vocabSize vocabulary size (use {@link #VOCAB_SIZE_BASE} for standard BERT)
     * @return BERT-base {@link TransformerEncoder}
     */
    public static TransformerEncoder bertBase(int vocabSize) {
        return TransformerEncoder.builder()
            .vocabSize(vocabSize)
            .dModel(768)
            .nHeads(12)
            .dFF(3072)
            .nLayers(12)
            .maxSeqLen(512)
            .dropout(0.1f)
            .build();
    }

    /**
     * Constructs a BERT-large encoder (24 layers, dModel=1024, 16 heads).
     *
     * @param vocabSize vocabulary size
     * @return BERT-large {@link TransformerEncoder}
     */
    public static TransformerEncoder bertLarge(int vocabSize) {
        return TransformerEncoder.builder()
            .vocabSize(vocabSize)
            .dModel(1024)
            .nHeads(16)
            .dFF(4096)
            .nLayers(24)
            .maxSeqLen(512)
            .dropout(0.1f)
            .build();
    }

    /**
     * Constructs a BERT-base model with a sequence classification head.
     *
     * <p>Architecture: BERT-base encoder → [CLS] pooling → Dropout → Linear(numClasses)
     *
     * @param numClasses number of output classes
     * @param vocabSize  vocabulary size
     * @return classification model
     */
    public static NNModule forSequenceClassification(int numClasses, int vocabSize) {
        return new BertForClassification(bertBase(vocabSize), numClasses);
    }

    // ── Classification head ───────────────────────────────────────────────

    /**
     * BERT encoder with a linear classification head on the [CLS] token.
     */
    static final class BertForClassification extends NNModule {

        private final TransformerEncoder encoder;
        private final Linear classifier;

        /**
         * @param encoder   pre-built BERT encoder
         * @param numClasses number of output classes
         */
        BertForClassification(TransformerEncoder encoder, int numClasses) {
            this.encoder    = register("encoder",    encoder);
            this.classifier = register("classifier", new Linear(768, numClasses));
        }

        /**
         * Forward pass: encode → extract [CLS] token → classify.
         *
         * @param x input embeddings {@code [B, T, dModel]}
         * @return logits {@code [B, numClasses]}
         */
        @Override
        public GradTensor forward(GradTensor x) {
            GradTensor encoded = encoder.forward(x); // [B, T, dModel]
            // Extract [CLS] token (position 0): [B, dModel]
            long[] s = encoded.shape();
            int B = (int) s[0], dModel = (int) s[2];
            float[] cls = new float[B * dModel];
            for (int b = 0; b < B; b++)
                System.arraycopy(encoded.data(), b * (int) s[1] * dModel, cls, b * dModel, dModel);
            return classifier.forward(GradTensor.of(cls, B, dModel));
        }
    }
}
