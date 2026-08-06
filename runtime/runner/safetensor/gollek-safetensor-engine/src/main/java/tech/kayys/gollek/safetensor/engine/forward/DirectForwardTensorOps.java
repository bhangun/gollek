/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

final class DirectForwardTensorOps {
    private static final float GELU_INNER_SCALE = 0.79788456f;
    private static final float GELU_CUBIC_COEFF = 0.044715f;

    private DirectForwardTensorOps() {
    }

    static long elementCount(long[] shape) {
        long count = 1L;
        for (long dim : shape) {
            count = Math.multiplyExact(count, dim);
        }
        return count;
    }

    static AccelTensor reusableWorkspaceView(MemorySegment segment, long[] shape) {
        long requiredBytes = elementCount(shape) * Float.BYTES;
        if (segment == null || segment.byteSize() < requiredBytes) {
            return null;
        }
        return AccelTensor.view(segment, shape);
    }

    static AccelTensor reusableOutputTensor(AccelTensor outputBuffer, long[] outputShape) {
        if (outputBuffer != null
                && !outputBuffer.isClosed()
                && outputBuffer.hasShape(outputShape)) {
            return outputBuffer;
        }
        return AccelTensor.zeros(outputShape);
    }

    static String tensorSummary(AccelTensor tensor) {
        if (tensor == null) {
            return "null";
        }
        return tensor.quantType() + Arrays.toString(tensor.shape());
    }

    static boolean allFinite(AccelTensor tensor) {
        if (tensor == null || tensor.quantType() != AccelTensor.QuantType.F32) {
            return false;
        }
        MemorySegment segment = tensor.dataPtr();
        long n = tensor.numel();
        for (long i = 0; i < n; i++) {
            if (!Float.isFinite(segment.getAtIndex(ValueLayout.JAVA_FLOAT, i))) {
                return false;
            }
        }
        return true;
    }

    static boolean isMultiRowLinearInput(AccelTensor input) {
        if (input == null || input.rank() == 0) {
            return false;
        }
        long hidden = input.size(-1);
        if (hidden <= 0L) {
            return false;
        }
        return input.numel() / hidden > 1L;
    }

    static boolean isSingleRowLinearInput(AccelTensor input) {
        if (input == null || input.rank() == 0) {
            return false;
        }
        long hidden = input.size(-1);
        if (hidden <= 0L) {
            return false;
        }
        return input.numel() / hidden == 1L;
    }

    static boolean isPackedF32LinearInput(AccelTensor input) {
        return input != null
                && input.rank() > 0
                && input.size(-1) > 0L
                && input.quantType() == AccelTensor.QuantType.F32
                && input.isContiguous();
    }

    static AccelTensor linearRowView(AccelTensor input, long row) {
        if (!isPackedF32LinearInput(input)) {
            throw new IllegalArgumentException("linear row views require packed F32 linear input");
        }
        long hidden = input.size(-1);
        long rows = input.numel() / hidden;
        if (row < 0L || row >= rows) {
            throw new IndexOutOfBoundsException("row " + row + " outside linear input rows " + rows);
        }
        long rowBytes = Math.multiplyExact(hidden, (long) Float.BYTES);
        MemorySegment rowSegment = input.dataPtr().asSlice(Math.multiplyExact(row, rowBytes), rowBytes);
        return AccelTensor.view(rowSegment, new long[] { 1L, hidden }, input);
    }

    static AccelTensor addBiasIfNeeded(AccelTensor tensor, AccelTensor bias) {
        if (bias == null) {
            return tensor;
        }
        AccelTensor biased = AccelOps.add(tensor, bias);
        tensor.close();
        return biased;
    }

    /**
     * In-place GeGLU combine for Gemma-style FFNs.
     * Writes gelu(gate) * up directly into the reusable workspace buffer.
     */
    static void fusedGeglu(AccelTensor out, AccelTensor gate, AccelTensor up) {
        MemorySegment outSeg = out.dataPtr();
        MemorySegment gateSeg = gate.dataPtr();
        MemorySegment upSeg = up.dataPtr();
        long n = gate.numel();

        for (long i = 0; i < n; i++) {
            float g = gateSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float u = upSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float inner = GELU_INNER_SCALE * (g + GELU_CUBIC_COEFF * g * g * g);
            float gelu = 0.5f * g * (1.0f + (float) Math.tanh(inner));
            outSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, gelu * u);
        }
    }

    static AccelTensor embeddingLookup(AccelTensor embedTable, long[] tokenIds) {
        int seqLen = tokenIds.length;
        AccelTensor selected = embedTable.indexSelect(tokenIds);
        long hiddenSize = selected.size(1);
        return selected.reshape(1L, seqLen, hiddenSize);
    }

    static AccelTensor selectLastToken(AccelTensor hidden, int seqLen) {
        if (seqLen < 1) {
            return hidden;
        }
        long hiddenSize = hidden.size(-1);
        return hidden.slice(1, seqLen - 1L, seqLen).reshape(1L, hiddenSize);
    }

    static AccelTensor selectTokenAt(AccelTensor hidden, int tokenIndex) {
        long hiddenSize = hidden.size(-1);
        long clamped = Math.max(0L, tokenIndex);
        return hidden.slice(1, clamped, clamped + 1L).reshape(1L, hiddenSize);
    }

    static int topIndex(float[] logits) {
        int best = -1;
        float bestVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] > bestVal) {
                bestVal = logits[i];
                best = i;
            }
        }
        return best;
    }

    static boolean isSingleTokenHalfLinearCandidate(AccelTensor input, AccelTensor weight) {
        if (weight.quantType() != AccelTensor.QuantType.BF16
                && weight.quantType() != AccelTensor.QuantType.F16) {
            return false;
        }
        if (input.rank() < 1) {
            return false;
        }
        long k = input.size(-1);
        long rows = input.numel() / Math.max(1L, k);
        return rows == 1L;
    }

    static long multiplySaturating(long left, long right) {
        try {
            return Math.multiplyExact(left, right);
        } catch (ArithmeticException e) {
            return Long.MAX_VALUE;
        }
    }
}
