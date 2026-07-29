/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;

final class DirectForwardElementwiseOps {
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private DirectForwardElementwiseOps() {
    }

    static void rmsNormRowsMetal(MetalBinding metalBinding, MemorySegment out, MemorySegment in,
            MemorySegment weight, int rows, int hiddenSize, float eps, boolean addOne) {
        int rc = metalBinding.rmsNormRows(out, in, weight, rows, hiddenSize, eps, addOne);
        if (rc != 0) {
            throw new IllegalStateException("Metal rmsNormRows failed with code " + rc);
        }
    }

    static void residualAdd(Logger log, MetalBinding metalBinding, MemorySegment left, AccelTensor right,
            MemorySegment out, int seqLen, long hiddenSize, boolean useMetalElementwise) {
        if (useMetalElementwise) {
            try {
                int elements = Math.toIntExact(seqLen * hiddenSize);
                int rc = metalBinding.add(out, left, right.dataPtr(), elements);
                if (rc != 0) {
                    throw new IllegalStateException("Metal add failed with code " + rc);
                }
                return;
            } catch (IllegalStateException | UnsupportedOperationException e) {
                log.debugf("Falling back from Metal add to direct CPU accumulation: %s", e.getMessage());
            }
        }

        addSegments(left, right.dataPtr(), out, seqLen * hiddenSize);
    }

    static void scaleSegmentInPlace(Logger log, MetalBinding metalBinding, MemorySegment seg,
            int seqLen, long hiddenSize, float scale, boolean useMetalLayerScalarScale) {
        long elements = seqLen * hiddenSize;
        if (useMetalLayerScalarScale) {
            try {
                int rc = metalBinding.scale(seg, seg, scale, Math.toIntExact(elements));
                if (rc == 0) {
                    return;
                }
                log.debugf("Falling back from Metal layer scalar scale: native returned %d", rc);
            } catch (IllegalStateException | UnsupportedOperationException | ArithmeticException e) {
                log.debugf("Falling back from Metal layer scalar scale: %s", e.getMessage());
            }
        }
        long i = 0;
        long upperBound = FLOAT_SPECIES.loopBound(elements);
        for (; i < upperBound; i += FLOAT_SPECIES.length()) {
            FloatVector vec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, seg, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            vec.mul(scale).intoMemorySegment(seg, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }
        for (; i < elements; i++) {
            float value = seg.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            seg.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, value * scale);
        }
    }

    private static void addSegments(MemorySegment left, MemorySegment right, MemorySegment out, long elements) {
        long i = 0;
        long upperBound = FLOAT_SPECIES.loopBound(elements);
        for (; i < upperBound; i += FLOAT_SPECIES.length()) {
            FloatVector leftVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, left, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            FloatVector rightVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, right, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            leftVec.add(rightVec).intoMemorySegment(out, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }
        for (; i < elements; i++) {
            float sum = left.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i)
                    + right.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, sum);
        }
    }
}
