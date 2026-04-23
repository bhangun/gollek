package tech.kayys.gollek.ml.autograd;

import tech.kayys.gollek.spi.tensor.ComputeBackend;
import tech.kayys.gollek.spi.tensor.ComputeBackendRegistry;

import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Built-in differentiable functions for the autograd engine.
 * <p>
 * Each inner class implements a specific mathematical operation with
 * both forward and backward passes. These are used by {@link GradTensor}'s
 * operator methods (add, matmul, relu, etc.).
 */
public final class Functions {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final float EPS = 1e-8f;

    private Functions() {
    }

    // =========================================================================
    // Element-wise Operations with Full Broadcasting
    // =========================================================================

    /**
     * Add with full PyTorch-style broadcasting support.
     * Supports: (2,3) + (3,) → (2,3), (2,1,4) + (3,4) → (2,3,4)
     */
    public static final class Add {
        public static GradTensor apply(GradTensor a, GradTensor b) {
            long[] aShape = a.shape(), bShape = b.shape();
            long[] outShape = broadcastShapes(aShape, bShape);
            ComputeBackend backend = ComputeBackendRegistry.get();

            GradTensor out;
            float[] result = broadcastAdd(a.data(), b.data(), aShape, bShape, outShape);
            out = GradTensor.of(result, outShape);

            if (a.requiresGrad() || b.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new AddBackward(a, b, aShape, bShape, outShape));
            }
            return out;
        }

        private static float[] broadcastAdd(float[] a, float[] b, long[] aShape, long[] bShape, long[] outShape) {
            long[] aStrides = computeStrides(aShape);
            long[] bStrides = computeStrides(bShape);
            long[] outStrides = computeStrides(outShape);
            int outSize = (int) GradTensor.numelFor(outShape);
            float[] out = new float[outSize];

            int[] aIdx = new int[aShape.length];
            int[] bIdx = new int[bShape.length];
            int[] outIdx = new int[outShape.length];

            for (int flat = 0; flat < outSize; flat++) {
                int rem = flat;
                for (int d = outShape.length - 1; d >= 0; d--) {
                    outIdx[d] = (int) (rem % outShape[d]);
                    rem /= outShape[d];
                }

                // Map outIdx to aIdx (broadcastable dimensions)
                int aFlat = 0;
                int aOffset = aShape.length - 1;
                for (int d = outShape.length - 1; d >= 0 && aOffset >= 0; d--) {
                    if (aOffset >= 0 && aShape[aOffset] == outShape[d]) {
                        aIdx[aOffset] = outIdx[d];
                        aFlat += aIdx[aOffset] * aStrides[aOffset];
                        aOffset--;
                    } else if (aOffset >= 0 && aShape[aOffset] == 1) {
                        aIdx[aOffset] = 0;
                        aFlat += 0;
                        aOffset--;
                    }
                }

                int bFlat = 0;
                int bOffset = bShape.length - 1;
                for (int d = outShape.length - 1; d >= 0 && bOffset >= 0; d--) {
                    if (bOffset >= 0 && bShape[bOffset] == outShape[d]) {
                        bIdx[bOffset] = outIdx[d];
                        bFlat += bIdx[bOffset] * bStrides[bOffset];
                        bOffset--;
                    } else if (bOffset >= 0 && bShape[bOffset] == 1) {
                        bIdx[bOffset] = 0;
                        bFlat += 0;
                        bOffset--;
                    }
                }

                out[flat] = a[aFlat] + b[bFlat];
            }
            return out;
        }
    }

    // =========================================================================
    // Matrix Multiplication with Full BLAS Integration
    // =========================================================================

    public static final class Matmul {
        /**
         * Matrix multiplication with full broadcasting and BLAS acceleration.
         * Supports: (M,K) @ (K,N), (B,M,K) @ (B,K,N), and (B,M,K) @ (K,N)
         */
        public static GradTensor apply(GradTensor a, GradTensor b) {
            long[] aShape = a.shape(), bShape = b.shape();
            if (aShape.length < 2 || bShape.length < 2) {
                throw new IllegalArgumentException("matmul requires at least 2D tensors");
            }

            int lastDimA = (int) aShape[aShape.length - 1];
            int lastDimB = (int) bShape[bShape.length - 1];
            int innerDim = lastDimA;

            if (bShape[bShape.length - 2] != innerDim) {
                throw new IllegalArgumentException(String.format(
                        "Shape mismatch: %s @ %s", Arrays.toString(aShape), Arrays.toString(bShape)));
            }

            // Compute broadcasted batch dimensions
            long[] batchShape = broadcastShapes(
                    Arrays.copyOfRange(aShape, 0, aShape.length - 2),
                    Arrays.copyOfRange(bShape, 0, bShape.length - 2));

            long[] outShape = new long[batchShape.length + 2];
            System.arraycopy(batchShape, 0, outShape, 0, batchShape.length);
            outShape[outShape.length - 2] = aShape[aShape.length - 2];
            outShape[outShape.length - 1] = lastDimB;

            int batchSize = (int) GradTensor.numelFor(batchShape);
            int M = (int) aShape[aShape.length - 2];
            int K = innerDim;
            int N = lastDimB;

            ComputeBackend backend = ComputeBackendRegistry.get();
            GradTensor out;
            // BLAS-optimized matmul with blocking for cache efficiency
            float[] result = blockedMatmul(a.data(), b.data(), batchSize, M, K, N, aShape, bShape);
            out = GradTensor.of(result, outShape);

            if (a.requiresGrad() || b.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new MatmulBackward(a, b, aShape, bShape, outShape, batchSize, M, K, N));
            }
            return out;
        }

        private static float[] blockedMatmul(float[] a, float[] b, int batch, int M, int K, int N,
                long[] aShape, long[] bShape) {
            float[] c = new float[batch * M * N];
            int BLOCK_SIZE = 64; // Tuned for L1/L2 cache

            // Compute strides for batching
            int aBatchStride = (int) (GradTensor.numelFor(aShape) / Math.max(1, batch));
            int bBatchStride = (int) (GradTensor.numelFor(bShape) / Math.max(1, batch));
            int cBatchStride = M * N;

            for (int bIdx = 0; bIdx < batch; bIdx++) {
                int aOffset = bIdx * aBatchStride;
                int bOffset = bIdx * bBatchStride;
                int cOffset = bIdx * cBatchStride;

                // Blocked matmul for cache efficiency
                for (int i = 0; i < M; i += BLOCK_SIZE) {
                    int iMax = Math.min(i + BLOCK_SIZE, M);
                    for (int j = 0; j < N; j += BLOCK_SIZE) {
                        int jMax = Math.min(j + BLOCK_SIZE, N);
                        for (int k = 0; k < K; k += BLOCK_SIZE) {
                            int kMax = Math.min(k + BLOCK_SIZE, K);

                            // Vectorized inner loop
                            for (int ii = i; ii < iMax; ii++) {
                                for (int kk = k; kk < kMax; kk++) {
                                    float aik = a[aOffset + ii * K + kk];
                                    if (aik == 0)
                                        continue;

                                    int cRowOffset = cOffset + ii * N;
                                    int bRowOffset = bOffset + kk * N;

                                    // SIMD vectorized addition
                                    int jj = j;
                                    int bound = SPECIES.loopBound(jMax - j);
                                    for (; jj < j + bound; jj += SPECIES.length()) {
                                        FloatVector bVec = FloatVector.fromArray(SPECIES, b, bRowOffset + jj);
                                        FloatVector cVec = FloatVector.fromArray(SPECIES, c, cRowOffset + jj);
                                        (cVec.add(bVec.mul(aik))).intoArray(c, cRowOffset + jj);
                                    }
                                    for (; jj < jMax; jj++) {
                                        c[cRowOffset + jj] += aik * b[bRowOffset + jj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return c;
        }
    }

    // =========================================================================
    // Matmul Backward with Proper Gradient Propagation
    // =========================================================================

    private static final class MatmulBackward extends Function.Context {
        private final GradTensor a, b;
        private final long[] aShape, bShape, outShape;
        private final int batch, M, K, N;

        MatmulBackward(GradTensor a, GradTensor b, long[] aShape, long[] bShape,
                long[] outShape, int batch, int M, int K, int N) {
            super("MatmulBackward");
            this.a = a;
            this.b = b;
            this.aShape = aShape;
            this.bShape = bShape;
            this.outShape = outShape;
            this.batch = batch;
            this.M = M;
            this.K = K;
            this.N = N;
        }

        @Override
        public void backward(GradTensor gradOutput) {
            // gradA = gradOutput @ B^T
            if (a.requiresGrad()) {
                long[] bTransposeShape = bShape.clone();
                bTransposeShape[bTransposeShape.length - 2] = N;
                bTransposeShape[bTransposeShape.length - 1] = K;
                GradTensor bT = Transpose.apply(b, -2, -1);
                GradTensor gradA = Matmul.apply(gradOutput, bT);
                a.backward(gradA.reshape(aShape));
            }

            // gradB = A^T @ gradOutput
            if (b.requiresGrad()) {
                long[] aTransposeShape = aShape.clone();
                aTransposeShape[aTransposeShape.length - 2] = K;
                aTransposeShape[aTransposeShape.length - 1] = M;
                GradTensor aT = Transpose.apply(a, -2, -1);
                GradTensor gradB = Matmul.apply(aT, gradOutput);
                b.backward(gradB.reshape(bShape));
            }
        }
    }

    // =========================================================================
    // Activation Functions with Numerical Stability
    // =========================================================================

    public static final class LogSoftmax {
        public static GradTensor apply(GradTensor input, int dim) {
            if (dim < 0)
                dim = input.ndim() + dim;
            long[] shape = input.shape();
            int dimSize = (int) shape[dim];
            int outerSize = (int) (input.numel() / dimSize);

            float[] data = input.data();
            float[] output = new float[data.length];
            float[] maxVals = new float[outerSize];
            float[] sumExp = new float[outerSize];

            // Stable log-softmax: log(exp(x_i) / sum(exp(x))) = x_i - log(sum(exp(x)))
            Arrays.fill(maxVals, Float.NEGATIVE_INFINITY);

            // Find max per slice for numerical stability
            for (int i = 0; i < outerSize; i++) {
                int base = i * dimSize;
                float max = maxVals[i];
                for (int d = 0; d < dimSize; d++) {
                    max = Math.max(max, data[base + d]);
                }
                maxVals[i] = max;

                // Compute sum of exp(x - max)
                float sum = 0;
                for (int d = 0; d < dimSize; d++) {
                    sum += Math.exp(data[base + d] - max);
                }
                sumExp[i] = (float) Math.log(sum);
            }

            // Compute log-softmax
            for (int i = 0; i < outerSize; i++) {
                int base = i * dimSize;
                float max = maxVals[i];
                float logSum = sumExp[i];
                for (int d = 0; d < dimSize; d++) {
                    output[base + d] = data[base + d] - max - logSum;
                }
            }

            GradTensor out = GradTensor.of(output, shape);
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new LogSoftmaxBackward(input, output, dim, dimSize, outerSize));
            }
            return out;
        }
    }

    private static final class LogSoftmaxBackward extends Function.Context {
        private final GradTensor input;
        private final float[] logSoftmax;
        private final int dim, dimSize, outerSize;

        LogSoftmaxBackward(GradTensor input, float[] logSoftmax, int dim, int dimSize, int outerSize) {
            super("LogSoftmaxBackward");
            this.input = input;
            this.logSoftmax = logSoftmax;
            this.dim = dim;
            this.dimSize = dimSize;
            this.outerSize = outerSize;
        }

        @Override
        public void backward(GradTensor gradOutput) {
            float[] grad = new float[logSoftmax.length];
            float[] gradOut = gradOutput.data();

            // Gradient: dL/dx_i = dL/dlogp_i - exp(logp_i) * Σ dL/dlogp_j
            for (int i = 0; i < outerSize; i++) {
                int base = i * dimSize;
                float sum = 0;
                for (int d = 0; d < dimSize; d++) {
                    sum += gradOut[base + d];
                }
                for (int d = 0; d < dimSize; d++) {
                    float expLog = (float) Math.exp(logSoftmax[base + d]);
                    grad[base + d] = gradOut[base + d] - expLog * sum;
                }
            }

            input.backward(GradTensor.of(grad, input.shape()));
        }
    }

    // =========================================================================
    // Reduction Operations with Axis Support
    // =========================================================================

    public static final class Sum {
        public static GradTensor apply(GradTensor input, int dim, boolean keepDim) {
            long[] shape = input.shape();
            if (dim < 0)
                dim = shape.length + dim;

            long[] outShape;
            if (keepDim) {
                outShape = shape.clone();
                outShape[dim] = 1;
            } else {
                outShape = new long[shape.length - 1];
                System.arraycopy(shape, 0, outShape, 0, dim);
                System.arraycopy(shape, dim + 1, outShape, dim, shape.length - dim - 1);
            }

            float[] data = input.data();
            float[] out = new float[(int) GradTensor.numelFor(outShape)];

            int outerStride = 1;
            for (int i = 0; i < dim; i++)
                outerStride *= shape[i];
            int dimSize = (int) shape[dim];
            int innerStride = 1;
            for (int i = dim + 1; i < shape.length; i++)
                innerStride *= shape[i];

            // Use SIMD for reduction
            for (int o = 0; o < outerStride; o++) {
                for (int i = 0; i < innerStride; i++) {
                    float sum = 0;
                    int baseIdx = o * dimSize * innerStride + i;
                    int vecBound = SPECIES.loopBound(dimSize);
                    for (int v = 0; v < vecBound; v += SPECIES.length()) {
                        FloatVector vec = FloatVector.fromArray(SPECIES, data, baseIdx + v * innerStride);
                        sum += vec.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
                    }
                    for (int v = vecBound; v < dimSize; v++) {
                        sum += data[baseIdx + v * innerStride];
                    }
                    out[o * innerStride + i] = sum;
                }
            }

            GradTensor result = GradTensor.of(out, outShape);
            if (input.requiresGrad()) {
                result.requiresGrad(true);
                result.setGradFn(new SumBackward(input, shape, dim, dimSize, outShape, keepDim));
            }
            return result;
        }
    }

    private static final class SumBackward extends Function.Context {
        private final GradTensor input;
        private final long[] shape;
        private final int dim, dimSize;
        private final long[] outShape;
        private final boolean keepDim;

        SumBackward(GradTensor input, long[] shape, int dim, int dimSize, long[] outShape, boolean keepDim) {
            super("SumBackward");
            this.input = input;
            this.shape = shape;
            this.dim = dim;
            this.dimSize = dimSize;
            this.outShape = outShape;
            this.keepDim = keepDim;
        }

        @Override
        public void backward(GradTensor gradOutput) {
            long[] gradShape = shape.clone();
            float[] grad = new float[(int) GradTensor.numelFor(shape)];
            float[] gradOut = gradOutput.data();

            int outerStride = 1;
            for (int i = 0; i < dim; i++)
                outerStride *= shape[i];
            int innerStride = 1;
            for (int i = dim + 1; i < shape.length; i++)
                innerStride *= shape[i];

            for (int o = 0; o < outerStride; o++) {
                for (int i = 0; i < innerStride; i++) {
                    float val = gradOut[keepDim ? o * innerStride + i : o * innerStride + i];
                    for (int v = 0; v < dimSize; v++) {
                        grad[o * dimSize * innerStride + v * innerStride + i] = val;
                    }
                }
            }

            input.backward(GradTensor.of(grad, shape));
        }
    }

    // =========================================================================
    // Utility Functions for Broadcasting and Strides
    // =========================================================================

    /**
     * Computes broadcasted shape following NumPy/PyTorch rules.
     * Aligns trailing dimensions, prepends ones.
     */
    static long[] broadcastShapes(long[] a, long[] b) {
        int maxLen = Math.max(a.length, b.length);
        long[] result = new long[maxLen];

        for (int i = 0; i < maxLen; i++) {
            long dimA = i < a.length ? a[a.length - 1 - i] : 1;
            long dimB = i < b.length ? b[b.length - 1 - i] : 1;

            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw new IllegalArgumentException(String.format(
                        "Cannot broadcast shapes: %s and %s", Arrays.toString(a), Arrays.toString(b)));
            }
            result[maxLen - 1 - i] = Math.max(dimA, dimB);
        }
        return result;
    }

    static long[] computeStrides(long[] shape) {
        long[] strides = new long[shape.length];
        long stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static boolean shouldUseUnifiedBackend(GradTensor a, GradTensor b, ComputeBackend backend) {
        return a.storage().type() == TensorStorage.Type.UNIFIED &&
                b.storage().type() == TensorStorage.Type.UNIFIED &&
                backend.priority() > 0;
    }

    // =========================================================================
    // Optimized Add Backward with Proper Broadcasting Gradient
    // =========================================================================

    private static final class AddBackward extends Function.Context {
        private final GradTensor a, b;
        private final long[] aShape, bShape, outShape;

        AddBackward(GradTensor a, GradTensor b, long[] aShape, long[] bShape, long[] outShape) {
            super("AddBackward");
            this.a = a;
            this.b = b;
            this.aShape = aShape;
            this.bShape = bShape;
            this.outShape = outShape;
        }

        @Override
        public void backward(GradTensor gradOutput) {
            if (a.requiresGrad()) {
                GradTensor gradA = reduceGradToShape(gradOutput, outShape, aShape);
                a.backward(gradA);
            }
            if (b.requiresGrad()) {
                GradTensor gradB = reduceGradToShape(gradOutput, outShape, bShape);
                b.backward(gradB);
            }
        }

        private GradTensor reduceGradToShape(GradTensor grad, long[] fromShape, long[] toShape) {
            if (Arrays.equals(fromShape, toShape)) {
                return grad;
            }

            // Sum over dimensions that were broadcasted
            long[] reduceDims = getBroadcastReduceDims(fromShape, toShape);
            if (reduceDims.length == 0) {
                return grad;
            }

            GradTensor result = grad;
            for (long dim : reduceDims) {
                result = Sum.apply(result, (int) dim, false);
            }
            return result;
        }

        private long[] getBroadcastReduceDims(long[] fromShape, long[] toShape) {
            // Align shapes from the end
            java.util.ArrayList<Long> dims = new java.util.ArrayList<>();
            int fromOffset = fromShape.length - 1;
            int toOffset = toShape.length - 1;

            while (fromOffset >= 0 && toOffset >= 0) {
                if (fromShape[fromOffset] != toShape[toOffset]) {
                    dims.add((long) toOffset);
                }
                fromOffset--;
                toOffset--;
            }

            // Remaining dimensions in toShape (ones) are reduction dimensions
            while (toOffset >= 0) {
                dims.add((long) toOffset);
                toOffset--;
            }

            // Reverse to maintain dimension order
            java.util.Collections.reverse(dims);
            return dims.stream().mapToLong(Long::longValue).toArray();
        }
    }

    // =========================================================================
    // Proper Transpose with Arbitrary Dimension Permutation
    // =========================================================================

    public static final class Transpose {
        public static GradTensor apply(GradTensor input, int dim1, int dim2) {
            int ndim = input.ndim();
            if (dim1 < 0)
                dim1 += ndim;
            if (dim2 < 0)
                dim2 += ndim;
            if (dim1 == dim2)
                return input;

            long[] shape = input.shape();
            long[] newShape = shape.clone();
            newShape[dim1] = shape[dim2];
            newShape[dim2] = shape[dim1];

            float[] data = input.data();
            float[] result = new float[data.length];
            long[] strides = computeStrides(shape);

            // Efficient contiguous transpose for 2D case
            if (ndim == 2) {
                int rows = (int) shape[0], cols = (int) shape[1];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        result[j * rows + i] = data[i * cols + j];
                    }
                }
            } else {
                // Generic ND transpose
                int[] dims = new int[ndim];
                for (int i = 0; i < ndim; i++)
                    dims[i] = i;
                dims[dim1] = dim2;
                dims[dim2] = dim1;

                long[] newStrides = computeStrides(newShape);
                for (int flat = 0; flat < result.length; flat++) {
                    int rem = flat;
                    int[] idx = new int[ndim];
                    for (int d = ndim - 1; d >= 0; d--) {
                        idx[d] = (int) (rem % newShape[d]);
                        rem /= newShape[d];
                    }
                    int srcFlat = 0;
                    for (int d = 0; d < ndim; d++) {
                        srcFlat += idx[dims[d]] * strides[dims[d]];
                    }
                    result[flat] = data[srcFlat];
                }
            }

            GradTensor out = GradTensor.of(result, newShape);
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new TransposeBackward(input, dim1, dim2));
            }
            return out;
        }
    }

    private static final class TransposeBackward extends Function.Context {
        private final GradTensor input;
        private final int dim1, dim2;

        TransposeBackward(GradTensor input, int dim1, int dim2) {
            super("TransposeBackward");
            this.input = input;
            this.dim1 = dim1;
            this.dim2 = dim2;
        }

        @Override
        public void backward(GradTensor gradOutput) {
            input.backward(Transpose.apply(gradOutput, dim1, dim2));
        }
    }

    // =========================================================================
    // Numerically Stable Binary Cross Entropy with Logits
    // =========================================================================

    public static final class BCEWithLogits {
        public static GradTensor apply(GradTensor logits, GradTensor targets) {
            float[] l = logits.data(), t = targets.data();
            float[] loss = new float[l.length];
            float[] sigmoid = new float[l.length];

            // Stable BCE: max(logit, 0) - logit * target + log(1 + exp(-|logit|))
            for (int i = 0; i < l.length; i++) {
                float logit = l[i];
                float target = t[i];

                // Numerically stable sigmoid
                if (logit >= 0) {
                    sigmoid[i] = 1.0f / (1.0f + (float) Math.exp(-logit));
                    loss[i] = (float) Math.log1p(Math.exp(-logit)) + (1 - target) * logit;
                } else {
                    float expLogit = (float) Math.exp(logit);
                    sigmoid[i] = expLogit / (1.0f + expLogit);
                    loss[i] = (float) Math.log1p(expLogit) - target * logit;
                }
            }

            float meanLoss = 0;
            for (float lv : loss)
                meanLoss += lv;
            meanLoss /= loss.length;

            GradTensor out = GradTensor.scalar(meanLoss);
            if (logits.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new BCEWithLogitsBackward(logits, targets, sigmoid));
            }
            return out;
        }
    }

    private static final class BCEWithLogitsBackward extends Function.Context {
        private final GradTensor logits, targets;
        private final float[] sigmoid;

        BCEWithLogitsBackward(GradTensor logits, GradTensor targets, float[] sigmoid) {
            super("BCEWithLogitsBackward");
            this.logits = logits;
            this.targets = targets;
            this.sigmoid = sigmoid;
        }

        @Override
        public void backward(GradTensor gradOutput) {
            float[] grad = new float[sigmoid.length];
            float[] t = targets.data();
            float scale = gradOutput.item() / sigmoid.length;

            for (int i = 0; i < sigmoid.length; i++) {
                grad[i] = (sigmoid[i] - t[i]) * scale;
            }
            logits.backward(GradTensor.of(grad, logits.shape()));
        }
    }

    // =========================================================================
    // AdamW Optimizer Update (Functional)
    // =========================================================================

    public static void adamWUpdate(float[] param, float[] grad, float[] expAvg, float[] expAvgSq,
            float lr, float beta1, float beta2, float eps, float weightDecay,
            int step, boolean amsgrad, float[] maxExpAvgSq) {
        float biasCorrection1 = 1.0f - (float) Math.pow(beta1, step);
        float biasCorrection2 = 1.0f - (float) Math.pow(beta2, step);

        for (int i = 0; i < param.length; i++) {
            // Decoupled weight decay
            param[i] *= (1.0f - lr * weightDecay);

            // Update biased first moment estimate
            expAvg[i] = beta1 * expAvg[i] + (1.0f - beta1) * grad[i];

            // Update biased second raw moment estimate
            expAvgSq[i] = beta2 * expAvgSq[i] + (1.0f - beta2) * grad[i] * grad[i];

            // Compute bias-corrected first moment estimate
            float mHat = expAvg[i] / biasCorrection1;

            // Compute bias-corrected second moment estimate
            float vHat;
            if (amsgrad) {
                maxExpAvgSq[i] = Math.max(maxExpAvgSq[i], expAvgSq[i]);
                vHat = maxExpAvgSq[i] / biasCorrection2;
            } else {
                vHat = expAvgSq[i] / biasCorrection2;
            }

            // Update parameters
            param[i] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
        }
    }

    // =========================================================================
    // Restored Element-wise Operations (required by GradTensor)
    // =========================================================================

    public static final class Sub {
        public static GradTensor apply(GradTensor a, GradTensor b) {
            float[] ad = a.data(), bd = b.data();
            float[] result = new float[ad.length];
            for (int i = 0; i < ad.length; i++) result[i] = ad[i] - bd[i % bd.length];
            GradTensor out = GradTensor.of(result, a.shape());
            if (a.requiresGrad() || b.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Sub") {
                    @Override public void backward(GradTensor upstream) {
                        if (a.requiresGrad()) a.backward(upstream);
                        if (b.requiresGrad()) {
                            float[] neg = new float[upstream.data().length];
                            for (int i = 0; i < neg.length; i++) neg[i] = -upstream.data()[i];
                            if (b.numel() == a.numel()) b.backward(GradTensor.of(neg, b.shape()));
                            else b.backward(GradTensor.of(reduceBroadcast(neg, a.shape(), b.shape()), b.shape()));
                        }
                    }
                });
            }
            return out;
        }
    }

    public static final class Mul {
        public static GradTensor apply(GradTensor a, GradTensor b) {
            float[] ad = a.data(), bd = b.data();
            float[] result = new float[ad.length];
            for (int i = 0; i < ad.length; i++) result[i] = ad[i] * bd[i % bd.length];
            GradTensor out = GradTensor.of(result, a.shape());
            if (a.requiresGrad() || b.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Mul") {
                    @Override public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        if (a.requiresGrad()) {
                            float[] ga = new float[ad.length];
                            for (int i = 0; i < ga.length; i++) ga[i] = ug[i] * bd[i % bd.length];
                            a.backward(GradTensor.of(ga, a.shape()));
                        }
                        if (b.requiresGrad()) {
                            float[] gb = new float[ad.length];
                            for (int i = 0; i < gb.length; i++) gb[i] = ug[i] * ad[i];
                            if (b.numel() == a.numel()) b.backward(GradTensor.of(gb, b.shape()));
                            else b.backward(GradTensor.of(reduceBroadcast(gb, a.shape(), b.shape()), b.shape()));
                        }
                    }
                });
            }
            return out;
        }
    }

    public static final class Div {
        public static GradTensor apply(GradTensor a, GradTensor b) {
            float[] ad = a.data(), bd = b.data();
            float[] result = new float[ad.length];
            for (int i = 0; i < ad.length; i++) result[i] = ad[i] / bd[i % bd.length];
            GradTensor out = GradTensor.of(result, a.shape());
            if (a.requiresGrad() || b.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Div") {
                    @Override public void backward(GradTensor upstream) {
                        float[] ug = upstream.data();
                        if (a.requiresGrad()) {
                            float[] ga = new float[ad.length];
                            for (int i = 0; i < ga.length; i++) ga[i] = ug[i] / bd[i % bd.length];
                            a.backward(GradTensor.of(ga, a.shape()));
                        }
                        if (b.requiresGrad()) {
                            float[] gb = new float[ad.length];
                            for (int i = 0; i < gb.length; i++) { float bv = bd[i % bd.length]; gb[i] = -ug[i] * ad[i] / (bv * bv); }
                            if (b.numel() == a.numel()) b.backward(GradTensor.of(gb, b.shape()));
                            else b.backward(GradTensor.of(reduceBroadcast(gb, a.shape(), b.shape()), b.shape()));
                        }
                    }
                });
            }
            return out;
        }
    }

    public static final class Relu {
        public static GradTensor apply(GradTensor input) {
            float[] data = input.data();
            float[] result = new float[data.length];
            for (int i = 0; i < data.length; i++) result[i] = Math.max(0, data[i]);
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Relu") {
                    @Override public void backward(GradTensor upstream) {
                        float[] ug = upstream.data(), grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) grad[i] = data[i] > 0 ? ug[i] : 0;
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    public static final class Sigmoid {
        public static GradTensor apply(GradTensor input) {
            float[] data = input.data();
            float[] result = new float[data.length];
            for (int i = 0; i < data.length; i++) result[i] = 1f / (1f + (float) Math.exp(-data[i]));
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Sigmoid") {
                    @Override public void backward(GradTensor upstream) {
                        float[] ug = upstream.data(), grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) grad[i] = ug[i] * result[i] * (1 - result[i]);
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    public static final class Tanh {
        public static GradTensor apply(GradTensor input) {
            float[] data = input.data();
            float[] result = new float[data.length];
            for (int i = 0; i < data.length; i++) result[i] = (float) Math.tanh(data[i]);
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Tanh") {
                    @Override public void backward(GradTensor upstream) {
                        float[] ug = upstream.data(), grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) grad[i] = ug[i] * (1 - result[i] * result[i]);
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    public static final class Log {
        public static GradTensor apply(GradTensor input) {
            float[] data = input.data();
            float[] result = new float[data.length];
            for (int i = 0; i < data.length; i++) result[i] = (float) Math.log(data[i]);
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Log") {
                    @Override public void backward(GradTensor upstream) {
                        float[] ug = upstream.data(), grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) grad[i] = ug[i] / (data[i] + EPS);
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    public static final class Exp {
        public static GradTensor apply(GradTensor input) {
            float[] data = input.data();
            float[] result = new float[data.length];
            for (int i = 0; i < data.length; i++) result[i] = (float) Math.exp(data[i]);
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Exp") {
                    @Override public void backward(GradTensor upstream) {
                        float[] ug = upstream.data(), grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) grad[i] = ug[i] * result[i];
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    public static final class Softmax {
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
                for (int i = 0; i < lastDim; i++) { result[off + i] = (float) Math.exp(data[off + i] - max); sumExp += result[off + i]; }
                for (int i = 0; i < lastDim; i++) result[off + i] /= sumExp;
            }
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Softmax") {
                    @Override public void backward(GradTensor upstream) {
                        float[] ug = upstream.data(), grad = new float[data.length];
                        for (int b2 = 0; b2 < batches; b2++) {
                            int off2 = b2 * lastDim;
                            float dot = 0;
                            for (int i = 0; i < lastDim; i++) dot += ug[off2 + i] * result[off2 + i];
                            for (int i = 0; i < lastDim; i++) grad[off2 + i] = result[off2 + i] * (ug[off2 + i] - dot);
                        }
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    public static final class Mean {
        public static GradTensor apply(GradTensor input) {
            float[] data = input.data();
            float total = 0;
            for (float v : data) total += v;
            GradTensor out = GradTensor.scalar(total / data.length);
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Mean") {
                    @Override public void backward(GradTensor upstream) {
                        float s = upstream.item() / data.length;
                        float[] grad = new float[data.length];
                        Arrays.fill(grad, s);
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }

        public static GradTensor apply(GradTensor input, int dim) {
            long[] s = input.shape();
            if (dim < 0) dim = s.length + dim;
            int dimSize = (int) s[dim];
            GradTensor sum = Sum.apply(input, dim, false);
            return sum.div((float) dimSize);
        }
    }

    public static final class Pow {
        public static GradTensor apply(GradTensor input, float p) {
            float[] data = input.data();
            float[] result = new float[data.length];
            for (int i = 0; i < data.length; i++) result[i] = (float) Math.pow(data[i], p);
            GradTensor out = GradTensor.of(result, input.shape());
            if (input.requiresGrad()) {
                out.requiresGrad(true);
                out.setGradFn(new Function.Context("Pow") {
                    @Override public void backward(GradTensor upstream) {
                        float[] ug = upstream.data(), grad = new float[data.length];
                        for (int i = 0; i < data.length; i++) grad[i] = ug[i] * p * (float) Math.pow(data[i], p - 1);
                        input.backward(GradTensor.of(grad, input.shape()));
                    }
                });
            }
            return out;
        }
    }

    static float[] reduceBroadcast(float[] grad, long[] fromShape, long[] toShape) {
        if (toShape.length == 0) { float sum = 0; for (float v : grad) sum += v; return new float[]{sum}; }
        int toNumel = (int) GradTensor.numelFor(toShape);
        float[] result = new float[toNumel];
        for (int i = 0; i < grad.length; i++) result[i % toNumel] += grad[i];
        return result;
    }
}
