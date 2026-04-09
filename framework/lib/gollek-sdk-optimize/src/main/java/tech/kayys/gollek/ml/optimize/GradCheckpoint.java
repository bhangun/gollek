package tech.kayys.gollek.ml.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Gradient checkpointing — trades compute for memory by recomputing
 * activations during the backward pass instead of storing them.
 *
 * <p>Reduces peak memory usage by up to {@code O(√N)} for a network with
 * {@code N} layers, at the cost of one extra forward pass per checkpoint segment.
 *
 * <p>Based on <em>"Training Deep Nets with Sublinear Memory Cost"</em>
 * (Chen et al., 2016) and the implementation in PyTorch's
 * {@code torch.utils.checkpoint}.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * // Wrap a segment of the forward pass
 * GradTensor out = GradCheckpoint.checkpoint(
 *     () -> expensiveLayer.forward(x)
 * );
 *
 * // Wrap a full model's forward pass
 * GradTensor out = GradCheckpoint.checkpoint(model, input);
 * }</pre>
 *
 * <h3>How it works</h3>
 * <ol>
 *   <li>Forward: run the segment with gradient tracking <em>disabled</em>,
 *       save only the inputs (not intermediate activations)</li>
 *   <li>Backward: when gradients are needed, re-run the segment with
 *       gradient tracking <em>enabled</em> to recompute activations</li>
 * </ol>
 */
public final class GradCheckpoint {

    private GradCheckpoint() {}

    /**
     * Runs a computation segment with gradient checkpointing.
     *
     * <p>The supplier is called once during the forward pass (without grad),
     * and again during the backward pass (with grad) to recompute activations.
     *
     * @param segment a {@link Supplier} that performs the forward computation
     *                and returns the output tensor
     * @return output tensor with a backward hook that triggers recomputation
     */
    public static GradTensor checkpoint(Supplier<GradTensor> segment) {
        // Forward: run without gradient tracking to save memory
        GradTensor output = segment.get().detach();

        // Register a backward hook that recomputes the segment
        output.requiresGrad(true);
        output.setGradFn(new tech.kayys.gollek.ml.autograd.Function.Context("CheckpointBackward") {
            @Override
            public void backward(GradTensor upstream) {
                // Recompute forward with gradients enabled
                GradTensor recomputed = segment.get();
                recomputed.requiresGrad(true);
                // Propagate upstream gradient through recomputed graph
                recomputed.backward(upstream);
            }
        });
        return output;
    }

    /**
     * Runs a module's forward pass with gradient checkpointing.
     *
     * @param module the module to checkpoint
     * @param input  input tensor
     * @return output tensor with checkpointed backward
     */
    public static GradTensor checkpoint(NNModule module, GradTensor input) {
        return checkpoint(() -> module.forward(input));
    }

    /**
     * Applies gradient checkpointing to a list of sequential modules,
     * creating one checkpoint per module.
     *
     * <p>This is the most memory-efficient strategy for deep networks —
     * each layer's activations are discarded and recomputed on demand.
     *
     * @param modules list of sequential modules
     * @param input   initial input tensor
     * @return final output after passing through all modules
     */
    public static GradTensor sequentialCheckpoint(List<NNModule> modules, GradTensor input) {
        GradTensor x = input;
        for (NNModule m : modules) {
            x = checkpoint(m, x);
        }
        return x;
    }
}
