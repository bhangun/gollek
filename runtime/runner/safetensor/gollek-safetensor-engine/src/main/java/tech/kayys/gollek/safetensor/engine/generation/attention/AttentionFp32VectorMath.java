/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

final class AttentionFp32VectorMath {
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private AttentionFp32VectorMath() {
    }

    static float dotProduct(MemorySegment q, long qOffset, MemorySegment k, long kOffset, int dim) {
        int i = 0;
        FloatVector sum = FloatVector.zero(FLOAT_SPECIES);
        int upperBound = FLOAT_SPECIES.loopBound(dim);
        for (; i < upperBound; i += FLOAT_SPECIES.length()) {
            FloatVector qVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, q, qOffset + (long) i * Float.BYTES, ByteOrder.nativeOrder());
            FloatVector kVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, k, kOffset + (long) i * Float.BYTES, ByteOrder.nativeOrder());
            sum = sum.add(qVec.mul(kVec));
        }
        float result = sum.reduceLanes(VectorOperators.ADD);
        for (; i < dim; i++) {
            result += q.getAtIndex(ValueLayout.JAVA_FLOAT, (qOffset / Float.BYTES) + i)
                    * k.getAtIndex(ValueLayout.JAVA_FLOAT, (kOffset / Float.BYTES) + i);
        }
        return result;
    }

    static void updateAccumulator(float[] acc, MemorySegment valueSegment, long valueOffset,
            float previousWeight, float currentWeight, int dim) {
        int i = 0;
        FloatVector previous = FloatVector.broadcast(FLOAT_SPECIES, previousWeight);
        FloatVector current = FloatVector.broadcast(FLOAT_SPECIES, currentWeight);
        int upperBound = FLOAT_SPECIES.loopBound(dim);
        for (; i < upperBound; i += FLOAT_SPECIES.length()) {
            FloatVector accVec = FloatVector.fromArray(FLOAT_SPECIES, acc, i);
            FloatVector valueVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, valueSegment, valueOffset + (long) i * Float.BYTES, ByteOrder.nativeOrder());
            accVec.mul(previous).add(valueVec.mul(current)).intoArray(acc, i);
        }
        for (; i < dim; i++) {
            acc[i] = acc[i] * previousWeight
                    + valueSegment.getAtIndex(ValueLayout.JAVA_FLOAT, (valueOffset / Float.BYTES) + i) * currentWeight;
        }
    }

    static void writeNormalizedAccumulator(MemorySegment out, long outIndex, float[] acc, float invL, int headDim) {
        int i = 0;
        FloatVector inv = FloatVector.broadcast(FLOAT_SPECIES, invL);
        int upperBound = FLOAT_SPECIES.loopBound(headDim);
        for (; i < upperBound; i += FLOAT_SPECIES.length()) {
            FloatVector.fromArray(FLOAT_SPECIES, acc, i)
                    .mul(inv)
                    .intoMemorySegment(out, (outIndex + i) * Float.BYTES, ByteOrder.nativeOrder());
        }
        for (; i < headDim; i++) {
            out.setAtIndex(ValueLayout.JAVA_FLOAT, outIndex + i, acc[i] * invL);
        }
    }
}
