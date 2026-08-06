/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;

final class DirectForwardDiagnostics {
    private DirectForwardDiagnostics() {
    }

    static void logTensorStats(AccelTensor tensor, String label) {
        logArrayStats(tensor.toFloatArray(), label);
    }

    static void logSegmentStats(MemorySegment segment, long[] shape, String label) {
        logTensorStats(AccelTensor.view(segment, shape), label);
    }

    static void logArrayStats(float[] values, String label) {
        double sum = 0.0;
        double sumSq = 0.0;
        double sumAbs = 0.0;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float value : values) {
            sum += value;
            sumSq += (double) value * value;
            sumAbs += Math.abs(value);
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }
        double n = Math.max(1, values.length);
        double mean = sum / n;
        double rms = Math.sqrt(sumSq / n);
        double meanAbs = sumAbs / n;
        System.err.printf("[DEBUG] %s stats: mean=%f meanAbs=%f rms=%f min=%f max=%f size=%d%n",
                label, mean, meanAbs, rms, min, max, values.length);
        System.err.flush();
    }
}
