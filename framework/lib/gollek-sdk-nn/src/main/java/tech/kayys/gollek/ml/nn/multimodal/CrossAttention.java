package tech.kayys.gollek.ml.nn.multimodal;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.MultiHeadAttention;

/**
 * Cross-Attention — conditions one modality on another via attention.
 *
 * <p>Used in vision-language models (Flamingo, CLIP), audio-text models,
 * and encoder-decoder Transformers (T5, BART).
 *
 * <p>Query comes from the primary modality (e.g. text);
 * Key and Value come from the conditioning modality (e.g. image patches).
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var crossAttn = new CrossAttention(embedDim=768, numHeads=12);
 * GradTensor out = crossAttn.forward(textQuery, imageFeatures); // [B, L_q, 768]
 * }</pre>
 */
public class CrossAttention extends NNModule {

    private final MultiHeadAttention attention;

    /**
     * @param embedDim model dimension (must be divisible by numHeads)
     * @param numHeads number of attention heads
     */
    public CrossAttention(int embedDim, int numHeads) {
        this.attention = register("attn", new MultiHeadAttention(embedDim, numHeads));
    }

    /**
     * Cross-attention: query attends to key/value from a different modality.
     *
     * @param query    primary modality representations {@code [B, L_q, D]}
     * @param keyValue conditioning modality features {@code [B, L_kv, D]}
     * @return cross-attended output {@code [B, L_q, D]}
     */
    public GradTensor forward(GradTensor query, GradTensor keyValue) {
        return attention.forward(query, keyValue, keyValue, false);
    }

    /** Self-attention fallback (query = key = value). */
    @Override
    public GradTensor forward(GradTensor input) {
        return attention.forward(input, input, input, false);
    }
}
