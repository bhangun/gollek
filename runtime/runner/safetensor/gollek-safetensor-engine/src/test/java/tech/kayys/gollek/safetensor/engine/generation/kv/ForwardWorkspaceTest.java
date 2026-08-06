/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.kv;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ForwardWorkspaceTest {

    @Test
    void allocatesCombinedScratchForEachLogicalToken() {
        try (ForwardWorkspace workspace = new ForwardWorkspace()) {
            workspace.ensureCapacity(10, 5, 7);

            assertNotNull(workspace.getCombinedSeg());
            assertTrue(workspace.getCombinedSeg().byteSize() >= 2L * 7L * Float.BYTES);
        }
    }

    @Test
    void reusesIndependentScratchPools() {
        try (ForwardWorkspace workspace = new ForwardWorkspace()) {
            workspace.ensureCapacity(4, 4, 8);
            workspace.ensureProjectionScratchCapacity(32);
            workspace.ensureLogitsCapacity(3);
            workspace.ensureAttentionMetadataCapacity(4, 1);

            assertNotNull(workspace.getHiddenASeg());
            assertNotNull(workspace.getGateSeg());
            assertNotNull(workspace.getLogitsSeg());
            assertNotNull(workspace.getAttentionBlockTableSeg());
            assertNotNull(workspace.getAttentionContextLensSeg());
        }
    }
}
