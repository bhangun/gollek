/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardContiguousTensor(AccelTensor source, AccelTensor tensor) implements AutoCloseable {

    static DirectForwardContiguousTensor from(AccelTensor source) {
        AccelTensor checkedSource = Objects.requireNonNull(source, "source");
        return new DirectForwardContiguousTensor(checkedSource, checkedSource.contiguous());
    }

    boolean ownsTemporary() {
        return tensor != source;
    }

    @Override
    public void close() {
        if (ownsTemporary() && !tensor.isClosed()) {
            tensor.close();
        }
    }
}
