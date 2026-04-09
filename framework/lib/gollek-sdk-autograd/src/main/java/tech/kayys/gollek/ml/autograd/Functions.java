package tech.kayys.gollek.ml.autograd;

import tech.kayys.gollek.spi.tensor.ComputeBackend;
import tech.kayys.gollek.spi.tensor.ComputeBackendRegistry;

import java.util.Arrays;

/**
 * Built-in differentiable functions for the autograd engine.
 * <p>
 * Each inner class implements a specific mathematical operation with
 * both forward and backward passes. These are used by {@link GradTensor}'s
 * operator methods (add, matmul, relu, etc.).
 */
public final class Functions {

    private Functions() {}

    // ── Element-wise Add ─────────────────────────────────────────────────

    public static final class Add {
        public static GradTensor apply(GradTensor a, GradTensor b) {
            ComputeBackend backend = ComputeBackendRegistry.get();
            float[] ad = a.data(), bd = b.data();
            float[] result;
            if (bd.length == ad.length) {
                result = backend.add(ad, bd, a.shape());
            } else {
                // broadcast fallback
                result = new float[ad.length];
                for (int i = 0; i < ad.length; i++) {
                    result[i] = ad[i] + bd[i % bd.length];
                }
            }
            GradTensor out = GradTensor.of(result, a.shape());
            if (a.requiresGrad() || b.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Add") {
                    @Override
                    public void backward(GradTensor upstream) {
                        if (a.requiresGrad()) a.backward(upstream);
                        if (b.requiresGrad()) {
                            if (b.numel() == a.numel()) {
                                b.backward(upstream);
                            } else {
                                // Sum over broadcast dimensions
                                float[] bg = reduceBroadcast(upstream.data(), a.shape(), b.shape());
                                b.backward(GradTensor.of(bg, b.shape()));
                            }
                        }
                    }
                });
            }
            return out;
        }
    }

    // ── Element-wise Sub ─────────────────────────────────────────────────

    public static final class Sub {
        public static GradTensor apply(GradTensor a, GradTensor b) {
            ComputeBackend backend = ComputeBackendRegistry.get();
            float[] ad = a.data(), bd = b.data();
            float[] result;
            if (bd.length == ad.length) {
                result = backend.sub(ad, bd, a.shape());
            } else {
                result = new float[ad.length];
                for (int i = 0; i < ad.length; i++) {
                    result[i] = ad[i] - bd[i % bd.length];
                }
            }
            GradTensor out = GradTensor.of(result, a.shape());
            if (a.requiresGrad() || b.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Sub") {
                    @Override
                    public void backward(GradTensor upstream) {
                        if (a.requiresGrad()) a.backward(upstream);
                        if (b.requiresGrad()) {
                            float[] negGrad = new float[upstream.data().length];
                            for (int i = 0; i < negGrad.length; i++) negGrad[i] = -upstream.data()[i];
                            if (b.numel() == a.numel()) {
                                b.backward(GradTensor.of(negGrad, b.shape()));
                            } else {
                                float[] bg = reduceBroadcast(negGrad, a.shape(), b.shape());
                                b.backward(GradTensor.of(bg, b.shape()));
                            }
                        }
                    }
                });
            }
            return out;
        }
    }

    // ── Element-wise Mul ─────────────────────────────────────────────────

    public static final class Mul {
        public static GradTensor apply(GradTensor a, GradTensor b) {
            ComputeBackend backend = ComputeBackendRegistry.get();
            float[] ad = a.data(), bd = b.data();
            float[] result;
            if (bd.length == ad.length) {
                result = backend.mul(ad, bd, a.shape());
            } else {
                result = new float[ad.length];
                for (int i = 0; i < ad.length; i++) {
                    result[i] = ad[i] * bd[i % bd.length];
                }
            }
            GradTensor out = GradTensor.of(result, a.shape());
            if (a.requiresGrad() || b.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Mul") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        if (a.requiresGrad()) {
                            float[] ga = new float[ad.length];
                            for (int i = 0; i < ga.length; i++) ga[i] = ug[i] * bd[i % bd.length];
                            a.backward(GradTensor.of(ga, a.shape()));
                        }
                        if (b.requiresGrad()) {
                            float[] gb = new float[ad.length];
                            for (int i = 0; i < gb.length; i++) gb[i] = ug[i] * ad[i];
                            if (b.numel() == a.numel()) {
                                b.backward(GradTensor.of(gb, b.shape()));
                            } else {
                                float[] reduced = reduceBroadcast(gb, a.shape(), b.shape());
                                b.backward(GradTensor.of(reduced, b.shape()));
                            }
                        }
                    }
                });
            }
            return out;
        }
    }

    // ── Element-wise Div ─────────────────────────────────────────────────

    public static final class Div {
        public static GradTensor apply(GradTensor a, GradTensor b) {
            ComputeBackend backend = ComputeBackendRegistry.get();
            float[] ad = a.data(), bd = b.data();
            float[] result;
            if (bd.length == ad.length) {
                result = backend.div(ad, bd, a.shape());
            } else {
                result = new float[ad.length];
                for (int i = 0; i < ad.length; i++) {
                    result[i] = ad[i] / bd[i % bd.length];
                }
            }
            GradTensor out = GradTensor.of(result, a.shape());
            if (a.requiresGrad() || b.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Div") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        if (a.requiresGrad()) {
                            float[] ga = new float[ad.length];
                            for (int i = 0; i < ga.length; i++) ga[i] = ug[i] / bd[i % bd.length];
                            a.backward(GradTensor.of(ga, a.shape()));
                        }
                        if (b.requiresGrad()) {
                            float[] gb = new float[ad.length];
                            for (int i = 0; i < gb.length; i++) {
                                float bv = bd[i % bd.length];
                                gb[i] = -ug[i] * ad[i] / (bv * bv);
                            }
                            if (b.numel() == a.numel()) {
                                b.backward(GradTensor.of(gb, b.shape()));
                            } else {
                                float[] reduced = reduceBroadcast(gb, a.shape(), b.shape());
                                b.backward(GradTensor.of(reduced, b.shape()));
                            }
                        }
                    }
                });
            }
            return out;
        }
    }

    // ── Matmul ───────────────────────────────────────────────────────────

    public static final class Matmul {
        /**
         * Matrix multiplication supporting 2D tensors: [M,K] @ [K,N] = [M,N].
         * Also supports batched: [...,M,K] @ [...,K,N] = [...,M,N].
         */
        public static GradTensor apply(GradTensor a, GradTensor b) {
            long[] as = a.shape(), bs = b.shape();
            if (as.length < 2 || bs.length < 2) {
                throw new IllegalArgumentException("matmul requires at least 2D tensors");
            }
            int M = (int) as[as.length - 2];
            int K = (int) as[as.length - 1];
            int N = (int) bs[bs.length - 1];
            if (bs[bs.length - 2] != K) {
                throw new IllegalArgumentException(
                    "matmul shape mismatch: " + Arrays.toString(as) + " @ " + Arrays.toString(bs));
            }

            // Compute batch size
            int batchA = 1, batchB = 1;
            for (int i = 0; i < as.length - 2; i++) batchA *= (int) as[i];
            for (int i = 0; i < bs.length - 2; i++) batchB *= (int) bs[i];
            int batch = Math.max(batchA, batchB);

            float[] ad = a.data(), bd = b.data();
            float[] result = new float[batch * M * N];

            for (int bIdx = 0; bIdx < batch; bIdx++) {
                int aOff = (bIdx % batchA) * M * K;
                int bOff = (bIdx % batchB) * K * N;
                int rOff = bIdx * M * N;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        float sum = 0;
                        for (int k = 0; k < K; k++) {
                            sum += ad[aOff + i * K + k] * bd[bOff + k * N + j];
                        }
                        result[rOff + i * N + j] = sum;
                    }
                }
            }

            long[] outShape = new long[Math.max(as.length, bs.length)];
            System.arraycopy(as, 0, outShape, 0, as.length - 2);
            outShape[outShape.length - 2] = M;
            outShape[outShape.length - 1] = N;

            GradTensor out = GradTensor.of(result, outShape);
            if (a.requiresGrad() || b.requiresGrad()) {
                out.requiresGrad(true);
                final int fM = M, fK = K, fN = N, fBatch = batch, fBatchA = batchA, fBatchB = batchB;
                out.setGradFn(new Function.Context("Matmul") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        // ∂L/∂A = upstream @ B^T  → [M,N] @ [N,K] = [M,K]
                        if (a.requiresGrad()) {
                            float[] ga = new float[fBatchA * fM * fK];
                            for (int bIdx = 0; bIdx < fBatch; bIdx++) {
                                int ugOff = bIdx * fM * fN;
                                int bOff2 = (bIdx % fBatchB) * fK * fN;
                                int gaOff = (bIdx % fBatchA) * fM * fK;
                                for (int i = 0; i < fM; i++) {
                                    for (int k = 0; k < fK; k++) {
                                        float s = 0;
                                        for (int j = 0; j < fN; j++) {
                                            s += ug[ugOff + i * fN + j] * bd[bOff2 + k * fN + j];
                                        }
                                        ga[gaOff + i * fK + k] += s;
                                    }
                                }
                            }
                            a.backward(GradTensor.of(ga, a.shape()));
                        }
                        // ∂L/∂B = A^T @ upstream  → [K,M] @ [M,N] = [K,N]
                        if (b.requiresGrad()) {
                            float[] gb = new float[fBatchB * fK * fN];
                            for (int bIdx = 0; bIdx < fBatch; bIdx++) {
                                int aOff2 = (bIdx % fBatchA) * fM * fK;
                                int ugOff = bIdx * fM * fN;
                                int gbOff = (bIdx % fBatchB) * fK * fN;
                                for (int k = 0; k < fK; k++) {
                                    for (int j = 0; j < fN; j++) {
                                        float s = 0;
                                        for (int i = 0; i < fM; i++) {
                                            s += ad[aOff2 + i * fK + k] * ug[ugOff + i * fN + j];
                                        }
                                        gb[gbOff + k * fN + j] += s;
                                    }
                                }
                            }
                            b.backward(GradTensor.of(gb, b.shape()));
                        }
                    }
                });
            }
            return out;
        }
    }

    // ── Transpose ────────────────────────────────────────────────────────

    public static final class Transpose {
        /** Transpose the last two dimensions. */
        public static GradTensor apply(GradTensor input) {
            long[] s = input.shape();
            if (s.length < 2) throw new IllegalArgumentException("transpose requires at least 2D");
            int rows = (int) s[s.length - 2];
            int cols = (int) s[s.length - 1];
            int batch = (int) (input.numel() / (rows * cols));

            float[] data = input.data();
            float[] result = new float[data.length];
            for (int b = 0; b < batch; b++) {
                int off = b * rows * cols;
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        result[off + j * rows + i] = data[off + i * cols + j];
                    }
                }
            }

            long[] newShape = s.clone();
            newShape[s.length - 2] = cols;
            newShape[s.length - 1] = rows;

            GradTensor out = GradTensor.of(result, newShape);
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Transpose") {
                    @Override
                    public void backward(GradTensor upstream) {
                        input.backward(Transpose.apply(upstream));
                    }
                });
            }
            return out;
        }
    }

    // ── ReLU ─────────────────────────────────────────────────────────────

    public static final class Relu {
        public static GradTensor apply(GradTensor input) {
            ComputeBackend backend = ComputeBackendRegistry.get();
            float[] data = input.data();
            float[] result = backend.relu(data, input.shape());
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Relu") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        float[] grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) {
                            grad[i] = data[i] > 0 ? ug[i] : 0;
                        }
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    // ── Sigmoid ──────────────────────────────────────────────────────────

    public static final class Sigmoid {
        public static GradTensor apply(GradTensor input) {
            ComputeBackend backend = ComputeBackendRegistry.get();
            float[] data = input.data();
            float[] result = backend.sigmoid(data, input.shape());
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Sigmoid") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        float[] grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) {
                            grad[i] = ug[i] * result[i] * (1 - result[i]);
                        }
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    // ── Tanh ─────────────────────────────────────────────────────────────

    public static final class Tanh {
        public static GradTensor apply(GradTensor input) {
            ComputeBackend backend = ComputeBackendRegistry.get();
            float[] data = input.data();
            float[] result = backend.tanh(data, input.shape());
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Tanh") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        float[] grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) {
                            grad[i] = ug[i] * (1 - result[i] * result[i]);
                        }
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    // ── Log ──────────────────────────────────────────────────────────────

    public static final class Log {
        public static GradTensor apply(GradTensor input) {
            ComputeBackend backend = ComputeBackendRegistry.get();
            float[] data = input.data();
            float[] result = backend.log(data, input.shape());
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Log") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        float[] grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) {
                            grad[i] = ug[i] / (data[i] + 1e-8f);
                        }
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    // ── Exp ──────────────────────────────────────────────────────────────

    public static final class Exp {
        public static GradTensor apply(GradTensor input) {
            ComputeBackend backend = ComputeBackendRegistry.get();
            float[] data = input.data();
            float[] result = backend.exp(data, input.shape());
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Exp") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        float[] grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) {
                            grad[i] = ug[i] * result[i];
                        }
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    // ── Softmax ──────────────────────────────────────────────────────────

    public static final class Softmax {
        /** Softmax along the last dimension. */
        public static GradTensor apply(GradTensor input) {
            long[] s = input.shape();
            int lastDim = (int) s[s.length - 1];
            int batches = (int) (input.numel() / lastDim);

            float[] data = input.data();
            float[] result = new float[data.length];

            for (int b = 0; b < batches; b++) {
                int off = b * lastDim;
                float max = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < lastDim; i++) max = Math.max(max, data[off + i]);
                float sumExp = 0;
                for (int i = 0; i < lastDim; i++) {
                    result[off + i] = (float) Math.exp(data[off + i] - max);
                    sumExp += result[off + i];
                }
                for (int i = 0; i < lastDim; i++) {
                    result[off + i] /= sumExp;
                }
            }
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Softmax") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        float[] grad = new float[data.length];
                        for (int b2 = 0; b2 < batches; b2++) {
                            int off2 = b2 * lastDim;
                            float dot = 0;
                            for (int i = 0; i < lastDim; i++) {
                                dot += ug[off2 + i] * result[off2 + i];
                            }
                            for (int i = 0; i < lastDim; i++) {
                                grad[off2 + i] = result[off2 + i] * (ug[off2 + i] - dot);
                            }
                        }
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    // ── Sum ──────────────────────────────────────────────────────────────

    public static final class Sum {
        public static GradTensor apply(GradTensor input) {
            float[] data = input.data();
            float total = 0;
            for (float v : data) total += v;
            GradTensor out = GradTensor.scalar(total);
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Sum") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float s = upstream.item();
                        float[] grad = new float[data.length];
                        Arrays.fill(grad, s);
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    // ── Mean ─────────────────────────────────────────────────────────────

    public static final class Mean {
        public static GradTensor apply(GradTensor input) {
            float[] data = input.data();
            float total = 0;
            for (float v : data) total += v;
            float mean = total / data.length;
            GradTensor out = GradTensor.scalar(mean);
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Mean") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float s = upstream.item() / data.length;
                        float[] grad = new float[data.length];
                        Arrays.fill(grad, s);
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    // ── Pow ──────────────────────────────────────────────────────────────

    public static final class Pow {
        public static GradTensor apply(GradTensor input, float p) {
            ComputeBackend backend = ComputeBackendRegistry.get();
            float[] data = input.data();
            float[] result = backend.pow(data, input.shape(), p);
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Pow") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        float[] grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) {
                            grad[i] = ug[i] * p * (float) Math.pow(data[i], p - 1);
                        }
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    // ── Broadcast utility ────────────────────────────────────────────────

    /**
     * Reduce a gradient array from a larger shape back to a smaller (broadcast) shape.
     * Sums over the dimensions that were broadcast.
     */
    static float[] reduceBroadcast(float[] grad, long[] fromShape, long[] toShape) {
        // Simple case: toShape is a scalar
        if (toShape.length == 0) {
            float sum = 0;
            for (float v : grad) sum += v;
            return new float[]{sum};
        }

        int toNumel = (int) GradTensor.numelFor(toShape);
        float[] result = new float[toNumel];
        for (int i = 0; i < grad.length; i++) {
            result[i % toNumel] += grad[i];
        }
        return result;
    }
}
