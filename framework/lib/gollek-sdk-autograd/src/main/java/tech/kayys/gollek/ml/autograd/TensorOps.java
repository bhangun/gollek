package tech.kayys.gollek.ml.autograd;

import java.util.ArrayList;
import java.util.List;

/**
 * Advanced tensor operations on {@link GradTensor} — slice, gather, cat, stack, einsum.
 * All ops that need gradients register a backward context.
 *
 * <p>Uses JDK 25 Vector API via {@link VectorOps} for bulk copies where applicable.
 */
public final class TensorOps {

    private TensorOps() {}

    // ── Slice ────────────────────────────────────────────────────────────

    /**
     * Slice along {@code dim}: returns elements [start, end).
     * Equivalent to {@code tensor[..., start:end, ...]} in PyTorch.
     */
    public static GradTensor slice(GradTensor t, int dim, int start, int end) {
        long[] shape = t.shape();
        int ndim = shape.length;
        if (dim < 0) dim += ndim;
        int dimSize = (int) shape[dim];
        if (start < 0) start += dimSize;
        if (end   < 0) end   += dimSize;

        long[] outShape = shape.clone();
        outShape[dim] = end - start;
        int outLen = (int) GradTensor.numelFor(outShape);
        float[] out = new float[outLen];

        // stride before dim, stride at dim, stride after dim
        int strideBefore = 1;
        for (int d = 0; d < dim; d++) strideBefore *= (int) shape[d];
        int strideAt    = (int) shape[dim];
        int strideAfter = outLen / (strideBefore * (end - start));

        for (int b = 0; b < strideBefore; b++) {
            for (int s = start; s < end; s++) {
                int srcBase = (b * strideAt + s) * strideAfter;
                int dstBase = (b * (end - start) + (s - start)) * strideAfter;
                System.arraycopy(t.data(), srcBase, out, dstBase, strideAfter);
            }
        }

        GradTensor result = GradTensor.of(out, outShape);
        if (t.requiresGrad()) {
            result.requiresGrad(true);
            final int fDim = dim, fStart = start, fEnd = end;
            result.setGradFn(new Function.Context("SliceBackward") {
                @Override public void backward(GradTensor g) {
                    if (!t.requiresGrad()) return;
                    float[] fullGrad = new float[(int) t.numel()];
                    // scatter gradient back into the sliced region
                    int sb = 1;
                    for (int d = 0; d < fDim; d++) sb *= (int) shape[d];
                    int sa = (int) (g.numel() / (sb * (fEnd - fStart)));
                    for (int b = 0; b < sb; b++) {
                        for (int s = fStart; s < fEnd; s++) {
                            int src = (b * (fEnd - fStart) + (s - fStart)) * sa;
                            int dst = (b * strideAt + s) * sa;
                            for (int i = 0; i < sa; i++) fullGrad[dst + i] += g.data()[src + i];
                        }
                    }
                    t.backward(GradTensor.of(fullGrad, shape));
                }
            });
        }
        return result;
    }

    // ── Cat ──────────────────────────────────────────────────────────────

    /**
     * Concatenate tensors along {@code dim}.
     * All tensors must have the same shape except at {@code dim}.
     */
    public static GradTensor cat(List<GradTensor> tensors, int dim) {
        if (tensors.isEmpty()) throw new IllegalArgumentException("cat: empty list");
        long[] refShape = tensors.get(0).shape();
        int ndim = refShape.length;
        if (dim < 0) dim += ndim;

        long totalDim = 0;
        for (GradTensor t : tensors) totalDim += t.shape()[dim];

        long[] outShape = refShape.clone();
        outShape[dim] = totalDim;
        float[] out = new float[(int) GradTensor.numelFor(outShape)];

        int strideBefore = 1;
        for (int d = 0; d < dim; d++) strideBefore *= (int) refShape[d];
        int strideAfter = (int) (GradTensor.numelFor(refShape) / (strideBefore * refShape[dim]));

        int dstDimOffset = 0;
        for (GradTensor t : tensors) {
            int tDimSize = (int) t.shape()[dim];
            for (int b = 0; b < strideBefore; b++) {
                int src = b * tDimSize * strideAfter;
                int dst = (b * (int) totalDim + dstDimOffset) * strideAfter;
                System.arraycopy(t.data(), src, out, dst, tDimSize * strideAfter);
            }
            dstDimOffset += tDimSize;
        }

        GradTensor result = GradTensor.of(out, outShape);
        boolean anyGrad = tensors.stream().anyMatch(GradTensor::requiresGrad);
        if (anyGrad) {
            result.requiresGrad(true);
            final int fDim = dim;
            final long fTotal = totalDim;
            result.setGradFn(new Function.Context("CatBackward") {
                @Override public void backward(GradTensor g) {
                    int offset = 0;
                    for (GradTensor t : tensors) {
                        if (t.requiresGrad()) {
                            t.backward(slice(g, fDim, offset, offset + (int) t.shape()[fDim]));
                        }
                        offset += (int) t.shape()[fDim];
                    }
                }
            });
        }
        return result;
    }

    // ── Stack ────────────────────────────────────────────────────────────

    /**
     * Stack tensors along a new dimension at {@code dim}.
     * All tensors must have identical shapes.
     */
    public static GradTensor stack(List<GradTensor> tensors, int dim) {
        List<GradTensor> unsqueezed = new ArrayList<>();
        for (GradTensor t : tensors) unsqueezed.add(t.unsqueeze(dim));
        return cat(unsqueezed, dim);
    }

    // ── Gather ───────────────────────────────────────────────────────────

    /**
     * Gather values along {@code dim} using {@code index}.
     * Equivalent to {@code torch.gather(input, dim, index)}.
     *
     * <p>out[i][j][k] = input[i][index[i][j][k]][k]  (for dim=1, 3D)
     */
    public static GradTensor gather(GradTensor input, int dim, GradTensor index) {
        long[] shape = input.shape();
        if (dim < 0) dim += shape.length;
        long[] idxShape = index.shape();
        float[] inp = input.data();
        float[] idx = index.data();
        float[] out = new float[(int) GradTensor.numelFor(idxShape)];

        int outerSize = 1;
        for (int d = 0; d < dim; d++) outerSize *= (int) shape[d];
        int dimSize  = (int) shape[dim];
        int innerSize = (int) (GradTensor.numelFor(shape) / (outerSize * dimSize));
        int idxDimSize = (int) idxShape[dim];

        for (int outer = 0; outer < outerSize; outer++) {
            for (int i = 0; i < idxDimSize; i++) {
                for (int inner = 0; inner < innerSize; inner++) {
                    int idxPos = outer * idxDimSize * innerSize + i * innerSize + inner;
                    int srcIdx = (int) idx[idxPos];
                    int srcPos = outer * dimSize * innerSize + srcIdx * innerSize + inner;
                    out[idxPos] = inp[srcPos];
                }
            }
        }

        GradTensor result = GradTensor.of(out, idxShape);
        if (input.requiresGrad()) {
            result.requiresGrad(true);
            final int fDim = dim, fOuter = outerSize, fDimSize = dimSize, fInner = innerSize;
            result.setGradFn(new Function.Context("GatherBackward") {
                @Override public void backward(GradTensor g) {
                    float[] grad = new float[(int) input.numel()];
                    float[] gd = g.data();
                    for (int outer = 0; outer < fOuter; outer++) {
                        for (int i = 0; i < idxDimSize; i++) {
                            for (int inner = 0; inner < fInner; inner++) {
                                int pos = outer * (int) idxShape[fDim] * fInner + i * fInner + inner;
                                int srcIdx = (int) idx[pos];
                                grad[outer * fDimSize * fInner + srcIdx * fInner + inner] += gd[pos];
                            }
                        }
                    }
                    input.backward(GradTensor.of(grad, shape));
                }
            });
        }
        return result;
    }

    // ── Einsum (common patterns) ─────────────────────────────────────────

    /**
     * Einstein summation for common patterns used in attention and matmul.
     * Supported: "ij,jk->ik", "bij,bjk->bik", "bhid,bhjd->bhij"
     */
    public static GradTensor einsum(String equation, GradTensor a, GradTensor b) {
        return switch (equation.trim()) {
            // 2D matmul
            case "ij,jk->ik" -> a.matmul(b);
            // Batched matmul
            case "bij,bjk->bik" -> batchedMatmul(a, b);
            // Attention scores: Q[b,h,i,d] x K[b,h,j,d] -> [b,h,i,j]
            case "bhid,bhjd->bhij" -> batchedMatmulTranspose(a, b);
            // Attention output: scores[b,h,i,j] x V[b,h,j,d] -> [b,h,i,d]
            case "bhij,bhjd->bhid" -> batchedMatmul(a, b);
            default -> throw new UnsupportedOperationException("einsum: unsupported equation: " + equation);
        };
    }

    // ── Permute / Repeat / Chunk ──────────────────────────────────────────

    /**
     * Permutes tensor dimensions — equivalent to {@code torch.permute}.
     *
     * @param t    input tensor
     * @param dims new dimension order (e.g. {@code {0,2,1}} transposes dims 1 and 2)
     * @return permuted tensor (data reordered, not a view)
     */
    public static GradTensor permute(GradTensor t, int... dims) {
        long[] shape = t.shape();
        int ndim = shape.length;
        long[] newShape = new long[ndim];
        for (int i = 0; i < ndim; i++) newShape[i] = shape[dims[i]];

        long numel = t.numel();
        float[] src = t.data(), dst = new float[(int) numel];

        // Compute strides for source and destination
        long[] srcStride = strides(shape);
        long[] dstStride = strides(newShape);

        for (int flat = 0; flat < (int) numel; flat++) {
            // Decompose flat index into multi-dim index in dst layout
            int[] idx = new int[ndim];
            int rem = flat;
            for (int d = 0; d < ndim; d++) {
                idx[d] = (int)(rem / dstStride[d]);
                rem    = (int)(rem % dstStride[d]);
            }
            // Map to source index via inverse permutation
            int[] srcIdx = new int[ndim];
            for (int d = 0; d < ndim; d++) srcIdx[dims[d]] = idx[d];
            int srcFlat = 0;
            for (int d = 0; d < ndim; d++) srcFlat += srcIdx[d] * srcStride[d];
            dst[flat] = src[srcFlat];
        }
        return GradTensor.of(dst, newShape);
    }

    /**
     * Repeats tensor along each dimension — equivalent to {@code torch.Tensor.repeat}.
     *
     * @param t       input tensor
     * @param repeats number of repetitions per dimension
     * @return repeated tensor
     */
    public static GradTensor repeat(GradTensor t, int... repeats) {
        long[] shape = t.shape();
        long[] newShape = new long[shape.length];
        for (int d = 0; d < shape.length; d++) newShape[d] = shape[d] * repeats[d];

        float[] src = t.data(), dst = new float[(int) GradTensor.numelFor(newShape)];
        long[] srcStride = strides(shape);
        long[] dstStride = strides(newShape);

        for (int flat = 0; flat < dst.length; flat++) {
            int[] idx = new int[shape.length];
            int rem = flat;
            for (int d = 0; d < shape.length; d++) {
                idx[d] = (int)(rem / dstStride[d]) % (int)shape[d];
                rem    = (int)(rem % dstStride[d]);
            }
            int srcFlat = 0;
            for (int d = 0; d < shape.length; d++) srcFlat += idx[d] * srcStride[d];
            dst[flat] = src[srcFlat];
        }
        return GradTensor.of(dst, newShape);
    }

    /**
     * Splits a tensor into chunks of size {@code chunkSize} along {@code dim}.
     *
     * @param t         input tensor
     * @param chunkSize size of each chunk
     * @param dim       dimension to split along
     * @return list of chunk tensors (last chunk may be smaller)
     */
    public static java.util.List<GradTensor> chunk(GradTensor t, int chunkSize, int dim) {
        long[] shape = t.shape();
        int dimSize = (int) shape[dim];
        java.util.List<GradTensor> chunks = new java.util.ArrayList<>();
        for (int start = 0; start < dimSize; start += chunkSize) {
            int end = Math.min(start + chunkSize, dimSize);
            chunks.add(slice(t, dim, start, end));
        }
        return chunks;
    }

    // ── Stride helpers ────────────────────────────────────────────────────

    private static long[] strides(long[] shape) {
        long[] s = new long[shape.length];
        long stride = 1;
        for (int d = shape.length - 1; d >= 0; d--) { s[d] = stride; stride *= shape[d]; }
        return s;
    }

    // ── Batched matmul helpers ───────────────────────────────────────────

    /** [B, M, K] @ [B, K, N] → [B, M, N] */
    private static GradTensor batchedMatmul(GradTensor a, GradTensor b) {
        long[] sa = a.shape(), sb = b.shape();
        int B = (int) sa[0], M = (int) sa[1], K = (int) sa[2], N = (int) sb[2];
        float[] ad = a.data(), bd = b.data();
        float[] out = new float[B * M * N];
        for (int batch = 0; batch < B; batch++) {
            float[] aSlice = slice1d(ad, batch * M * K, M * K);
            float[] bSlice = slice1d(bd, batch * K * N, K * N);
            float[] c = VectorOps.matmul(aSlice, bSlice, M, K, N);
            System.arraycopy(c, 0, out, batch * M * N, M * N);
        }
        return GradTensor.of(out, B, M, N);
    }

    /** [B, H, I, D] @ [B, H, J, D]^T → [B, H, I, J] */
    private static GradTensor batchedMatmulTranspose(GradTensor a, GradTensor b) {
        long[] sa = a.shape();
        int B = (int) sa[0], H = (int) sa[1], I = (int) sa[2], D = (int) sa[3];
        int J = (int) b.shape()[2];
        float[] ad = a.data(), bd = b.data();
        float[] out = new float[B * H * I * J];
        for (int batch = 0; batch < B; batch++) {
            for (int head = 0; head < H; head++) {
                float[] q = slice1d(ad, (batch * H + head) * I * D, I * D);
                float[] k = slice1d(bd, (batch * H + head) * J * D, J * D);
                // q: [I, D], k: [J, D] → q @ k^T = [I, J]
                // transpose k to [D, J] then matmul
                float[] kt = transpose2d(k, J, D);
                float[] scores = VectorOps.matmul(q, kt, I, D, J);
                System.arraycopy(scores, 0, out, (batch * H + head) * I * J, I * J);
            }
        }
        return GradTensor.of(out, B, H, I, J);
    }

    private static float[] slice1d(float[] src, int offset, int len) {
        float[] out = new float[len];
        System.arraycopy(src, offset, out, 0, len);
        return out;
    }

    private static float[] transpose2d(float[] m, int rows, int cols) {
        float[] t = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                t[c * rows + r] = m[r * cols + c];
        return t;
    }
}
