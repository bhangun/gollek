/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class SharedKvStateTest {

    @Test
    void appendsIntoExistingPackedBuffersWhenStateGrows() {
        try (SharedKvState state = new SharedKvState(tensor(1.0f, 1, 2, 2, 2), tensor(101.0f, 1, 2, 2, 2));
                AccelTensor deltaKey = tensor(9.0f, 1, 1, 2, 2);
                AccelTensor deltaValue = tensor(109.0f, 1, 1, 2, 2)) {
            state.packedKeyData();

            state.append(deltaKey, deltaValue);

            AccelTensor keyView = state.key();
            try {
                assertArrayEquals(new float[] {
                        1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f
                }, keyView.toFloatArray(), 0.0001f);
            } finally {
                state.releaseView(keyView);
            }
            assertEquals(4, state.packedCapacityTokens());
            assertArrayEquals(new float[] {
                    1.0f, 2.0f, 5.0f, 6.0f, 9.0f, 10.0f, 0.0f, 0.0f,
                    3.0f, 4.0f, 7.0f, 8.0f, 11.0f, 12.0f, 0.0f, 0.0f
            }, toFloats(state.packedKeyData(), 16), 0.0001f);
            assertArrayEquals(new float[] {
                    101.0f, 102.0f, 105.0f, 106.0f, 109.0f, 110.0f, 0.0f, 0.0f,
                    103.0f, 104.0f, 107.0f, 108.0f, 111.0f, 112.0f, 0.0f, 0.0f
            }, toFloats(state.packedValueData(), 16), 0.0001f);
        }
    }

    private static AccelTensor tensor(float start, long... shape) {
        AccelTensor tensor = AccelTensor.zeros(shape);
        for (int i = 0; i < tensor.numel(); i++) {
            tensor.setFlat(i, start + i);
        }
        return tensor;
    }

    private static float[] toFloats(MemorySegment segment, int count) {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) {
            values[i] = segment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return values;
    }
}
