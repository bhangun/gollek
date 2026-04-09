package tech.kayys.gollek.ml.nn.multimodal;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.TensorOps;
import tech.kayys.gollek.ml.nn.Linear;
import tech.kayys.gollek.ml.nn.NNModule;

import java.util.List;

/**
 * Modality Fusion — concatenates embeddings from two modalities and
 * projects them into a shared representation space.
 *
 * <p>Used in vision-language models (CLIP-style), audio-visual models,
 * and any multi-modal architecture requiring late fusion.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var fusion = new ModalityFusion(768, 512, 256);
 * GradTensor joint = fusion.forward(imageEmb, textEmb); // [N, 256]
 * }</pre>
 */
public class ModalityFusion extends NNModule {

    private final Linear projection;

    /**
     * @param modalityADim dimension of the first modality embedding
     * @param modalityBDim dimension of the second modality embedding
     * @param outDim       output joint embedding dimension
     */
    public ModalityFusion(int modalityADim, int modalityBDim, int outDim) {
        this.projection = register("projection",
            new Linear(modalityADim + modalityBDim, outDim));
    }

    /**
     * Fuses two modality embeddings via concatenation and linear projection.
     *
     * @param modA first modality embedding {@code [N, modalityADim]}
     * @param modB second modality embedding {@code [N, modalityBDim]}
     * @return joint embedding {@code [N, outDim]}
     */
    public GradTensor forward(GradTensor modA, GradTensor modB) {
        GradTensor joint = TensorOps.cat(List.of(modA, modB), 1);
        return projection.forward(joint);
    }

    @Override
    public GradTensor forward(GradTensor input) {
        throw new UnsupportedOperationException("Use forward(modA, modB)");
    }
}
