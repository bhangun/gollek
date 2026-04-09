package tech.kayys.gollek.ml.autograd;

/**
 * Base class for differentiable functions in the autograd engine.
 * <p>
 * Each function implements a forward computation and a backward
 * gradient computation. The {@link Context} links the function
 * to the computation graph for reverse-mode AD.
 *
 * <h3>Implementing a custom function</h3>
 * <pre>{@code
 * public class MyOp {
 *     public static GradTensor apply(GradTensor input) {
 *         // Forward
 *         float[] result = computeForward(input.data());
 *         GradTensor output = GradTensor.of(result, input.shape());
 *         
 *         // Save context for backward
 *         if (input.requiresGrad()) {
 *             output.requiresGrad(true);
 *             output.setGradFn(new Function.Context("MyOp") {
 *                 public void backward(GradTensor upstream) {
 *                     float[] grad = computeGrad(upstream.data(), input.data());
 *                     input.backward(GradTensor.of(grad, input.shape()));
 *                 }
 *             });
 *         }
 *         return output;
 *     }
 * }
 * }</pre>
 */
public abstract class Function {

    /**
     * Links a function invocation to the autograd graph.
     * <p>
     * During forward, each differentiable operation creates a {@code Context}
     * that captures the inputs and any saved tensors needed for the backward
     * pass. When {@link GradTensor#backward()} is called, the context's
     * {@link #backward(GradTensor)} method is invoked with the upstream gradient.
     */
    public abstract static class Context {

        private final String name;

        protected Context(String name) {
            this.name = name;
        }

        /** Human-readable name of this function (e.g. "Add", "Matmul"). */
        public String name() {
            return name;
        }

        /**
         * Compute and propagate gradients to inputs.
         *
         * @param upstream the gradient flowing from downstream operations
         */
        public abstract void backward(GradTensor upstream);
    }
}
