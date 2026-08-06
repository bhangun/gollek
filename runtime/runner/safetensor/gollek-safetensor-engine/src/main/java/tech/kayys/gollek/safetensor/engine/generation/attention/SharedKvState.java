/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;

public class SharedKvState implements AutoCloseable {
    private AccelTensor keyBuffer;
    private AccelTensor valueBuffer;
    private AccelTensor packedKeyBuffer;
    private AccelTensor packedValueBuffer;
    private final int numHeads;
    private final int headDim;
    private int lengthTokens;
    private int capacityTokens;

    public SharedKvState(AccelTensor key, AccelTensor value) {
        this(key, value, Math.toIntExact(key.size(2)), Math.toIntExact(key.size(3)),
                Math.toIntExact(key.size(1)), Math.toIntExact(key.size(1)));
    }

    private SharedKvState(AccelTensor keyBuffer, AccelTensor valueBuffer,
            int numHeads, int headDim, int lengthTokens, int capacityTokens) {
        this.keyBuffer = keyBuffer;
        this.valueBuffer = valueBuffer;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.lengthTokens = lengthTokens;
        this.capacityTokens = capacityTokens;
    }

    public AccelTensor key() {
        return SharedKvBufferOps.viewTokenMajor(keyBuffer, lengthTokens);
    }

    public AccelTensor value() {
        return SharedKvBufferOps.viewTokenMajor(valueBuffer, lengthTokens);
    }

    public void releaseView(AccelTensor tensor) {
        if (tensor != null
                && tensor != keyBuffer
                && tensor != valueBuffer
                && !tensor.isClosed()) {
            tensor.close();
        }
    }

    public void append(AccelTensor deltaKey, AccelTensor deltaValue) {
        int deltaTokens = Math.toIntExact(deltaKey.size(1));
        if (deltaTokens <= 0) {
            closeIfView(deltaKey, keyBuffer);
            closeIfView(deltaValue, valueBuffer);
            return;
        }

        ensureCapacity(lengthTokens + deltaTokens);
        SharedKvBufferOps.appendTokenMajor(keyBuffer, deltaKey, lengthTokens, numHeads, headDim);
        SharedKvBufferOps.appendTokenMajor(valueBuffer, deltaValue, lengthTokens, numHeads, headDim);
        if (packedKeyBuffer != null && packedValueBuffer != null) {
            SharedKvBufferOps.appendTokenMajorToHeadMajor(packedKeyBuffer, deltaKey, lengthTokens, numHeads, headDim);
            SharedKvBufferOps.appendTokenMajorToHeadMajor(packedValueBuffer, deltaValue, lengthTokens, numHeads,
                    headDim);
        }
        lengthTokens += deltaTokens;
        closeIfView(deltaKey, keyBuffer);
        closeIfView(deltaValue, valueBuffer);
    }

    public MemorySegment packedKeyData() {
        ensurePackedBuffers();
        return packedKeyBuffer.dataPtr();
    }

    public MemorySegment packedValueData() {
        ensurePackedBuffers();
        return packedValueBuffer.dataPtr();
    }

    public int packedCapacityTokens() {
        ensurePackedBuffers();
        return capacityTokens;
    }

    private void ensureCapacity(int requiredTokens) {
        if (requiredTokens <= capacityTokens) {
            return;
        }
        int newCapacity = Math.max(requiredTokens, Math.max(4, capacityTokens * 2));
        keyBuffer = SharedKvBufferOps.growTokenMajor(keyBuffer, newCapacity, numHeads, headDim, lengthTokens);
        valueBuffer = SharedKvBufferOps.growTokenMajor(valueBuffer, newCapacity, numHeads, headDim, lengthTokens);
        if (packedKeyBuffer != null) {
            packedKeyBuffer = SharedKvBufferOps.growHeadMajor(packedKeyBuffer, newCapacity, numHeads, headDim,
                    lengthTokens);
        }
        if (packedValueBuffer != null) {
            packedValueBuffer = SharedKvBufferOps.growHeadMajor(packedValueBuffer, newCapacity, numHeads, headDim,
                    lengthTokens);
        }
        capacityTokens = newCapacity;
    }

    private void ensurePackedBuffers() {
        if (packedKeyBuffer != null && packedValueBuffer != null) {
            return;
        }
        long batch = keyBuffer.size(0);
        packedKeyBuffer = AccelTensor.zeros(batch, numHeads, capacityTokens, headDim);
        packedValueBuffer = AccelTensor.zeros(batch, numHeads, capacityTokens, headDim);
        SharedKvBufferOps.packTokenMajorToHeadMajor(keyBuffer, packedKeyBuffer, lengthTokens, numHeads, headDim);
        SharedKvBufferOps.packTokenMajorToHeadMajor(valueBuffer, packedValueBuffer, lengthTokens, numHeads, headDim);
    }

    private static void closeIfView(AccelTensor tensor, AccelTensor backingBuffer) {
        if (tensor != null && tensor != backingBuffer && !tensor.isClosed()) {
            tensor.close();
        }
    }

    @Override
    public void close() {
        if (keyBuffer != null && !keyBuffer.isClosed()) {
            keyBuffer.close();
        }
        if (valueBuffer != null && !valueBuffer.isClosed()) {
            valueBuffer.close();
        }
        if (packedKeyBuffer != null && !packedKeyBuffer.isClosed()) {
            packedKeyBuffer.close();
        }
        if (packedValueBuffer != null && !packedValueBuffer.isClosed()) {
            packedValueBuffer.close();
        }
    }
}
