package tech.kayys.gollek.ml.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.nn.optim.Optimizer;

import java.util.List;

/**
 * Quantization-Aware Training (QAT) — simulates INT8 quantization during
 * the forward pass so the model learns to be robust to quantization error.
 *
 * <p>
 * Unlike post-training quantization, QAT typically recovers 1-2% accuracy
 * that PTQ loses, at the cost of additional training time.
 *
 * <p>
 * Strategy: insert fake-quantize operations in the forward pass that
 * round weights/activations to INT8 precision but keep float32 gradients.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var qat = new QuantizationAwareTraining(model, optimizer);
 * qat.enableQAT();
 *
 * for (var batch : loader) {
 *     model.zeroGrad();
 *     GradTensor loss = qat.forward(batch.inputs(), batch.labels(), lossFn);
 *     loss.backward();
 *     qat.step();
 * }
 *
 * // Convert to actual INT8 model
 * var quantized = qat.convertToInt8();
 * }</pre>
 */
public final class QuantizationAwareTraining {

    private final NNModule model;
    private final Optimizer optimizer;
    private boolean qatEnabled = false;

    /**
     * Creates a QAT wrapper around a model and optimizer.
     *
     * @param model     model to quantize-aware train
     * @param optimizer optimizer for parameter updates
     */
    public QuantizationAwareTraining(NNModule model, Optimizer optimizer) {
        this.model = model;
        this.optimizer = optimizer;
    }

    /**
     * Enables fake-quantization in the forward pass.
     * Call before starting QAT training.
     */
    public void enableQAT() {
        this.qatEnabled = true;
    }

    /**
     * Disables fake-quantization (returns to normal float32 training).
     */
    public void disableQAT() {
        this.qatEnabled = false;
    }

    /**
     * Forward pass with optional fake-quantization of weights.
     *
     * <p>
     * When QAT is enabled, weights are fake-quantized to INT8 precision
     * before the forward pass, then restored to float32 for gradient computation.
     *
     * @param input input tensor
     * @return model output with quantization noise injected
     */
    public GradTensor forward(GradTensor input) {
        if (qatEnabled)
            fakeQuantizeWeights();
        return model.forward(input);
    }

    /**
     * Performs an optimizer step (updates float32 master weights).
     */
    public void step() {
        optimizer.step();
    }

    /**
     * Converts the trained model to actual INT8 using post-training quantization.
     *
     * @return INT8 quantized state dict
     */
    public java.util.Map<String, PostTrainingQuantizer.QuantizedTensor> convertToInt8() {
        return new PostTrainingQuantizer().quantizeModel(model.stateDict());
    }

    // ── Fake quantization ─────────────────────────────────────────────────

    /**
     * Applies fake-quantization to all model weights in-place.
     * Rounds to INT8 precision then dequantizes back to float32.
     * Gradients flow through as if the operation were identity (straight-through
     * estimator).
     */
    private void fakeQuantizeWeights() {
        PostTrainingQuantizer q = new PostTrainingQuantizer();
        for (Parameter p : model.parameters()) {
            GradTensor t = p.data();
            PostTrainingQuantizer.QuantizedTensor qt = q.quantize(t);
            GradTensor dequant = q.dequantize(qt);
            // Copy dequantized values back (straight-through: grad passes unchanged)
            System.arraycopy(dequant.data(), 0, t.data(), 0, (int) t.numel());
        }
    }
}
