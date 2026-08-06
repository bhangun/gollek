/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.elementCount;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.selectTokenAt;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.topIndex;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.ArrayList;
import java.util.List;

final class DirectForwardLogits {
    private DirectForwardLogits() {
    }

    @FunctionalInterface
    interface LinearProjection {
        AccelTensor apply(AccelTensor input, AccelTensor weight, AccelTensor bias);
    }

    static AccelTensor reusableOutputTensor(ForwardWorkspace ws,
                                            AccelTensor input,
                                            AccelTensor lmHeadW) {
        if (ws == null || input == null || input.rank() == 0 || lmHeadW == null || lmHeadW.rank() != 2) {
            return null;
        }
        long[] outputShape = input.shapeWithLastDim(lmHeadW.size(0));
        ws.ensureLogitsCapacity(elementCount(outputShape));
        return AccelTensor.view(ws.getLogitsSeg(), outputShape);
    }

    static float[] materializeAndClose(AccelTensor logits) {
        try {
            long tLogitsCopy0 = System.nanoTime();
            float[] result = logits.toFloatArray();
            DirectInferenceProfiler.recordLogitsMaterializationNanos(System.nanoTime() - tLogitsCopy0);
            return result;
        } finally {
            logits.close();
        }
    }

    static void logPrefillDiagnostics(float[] result, ModelConfig config, boolean verboseTokensEnabled) {
        if (!verboseTokensEnabled || config.getNumAttentionHeads() <= 0) {
            return;
        }

        double sum = 0;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float f : result) {
            sum += f;
            if (f < min) {
                min = f;
            }
            if (f > max) {
                max = f;
            }
        }

        System.err.printf("[DEBUG] Logits stats: min=%f, max=%f, sum=%f, size=%d%n", min, max, sum, result.length);

        List<Integer> topIndices = new ArrayList<>();
        for (int i = 0; i < result.length; i++) {
            topIndices.add(i);
        }
        topIndices.sort((a, b) -> Float.compare(result[b], result[a]));
        for (int k = 0; k < Math.min(5, topIndices.size()); k++) {
            int id = topIndices.get(k);
            System.err.printf("  Top %d: ID=%d, val=%f%n", k, id, result[id]);
        }
    }

    static void debugSequencePositionLogits(AccelTensor hidden,
                                            AccelTensor lmHeadW,
                                            int seqLen,
                                            LinearProjection linear) {
        AccelTensor first = selectTokenAt(hidden, 0);
        AccelTensor last = selectTokenAt(hidden, seqLen - 1);
        AccelTensor firstLogits = null;
        AccelTensor lastLogits = null;
        try {
            firstLogits = linear.apply(first, lmHeadW, null);
            lastLogits = linear.apply(last, lmHeadW, null);
            int firstTop = topIndex(firstLogits.toFloatArray());
            int lastTop = topIndex(lastLogits.toFloatArray());
            System.err.printf("[DEBUG] Positional logits: firstTokenTop=%d lastTokenTop=%d%n", firstTop, lastTop);
            System.err.flush();
        } finally {
            if (firstLogits != null) {
                firstLogits.close();
            }
            if (lastLogits != null) {
                lastLogits.close();
            }
            first.close();
            last.close();
        }
    }
}
