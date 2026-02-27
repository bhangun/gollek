package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link LibTorchGenerationParams}.
 */
class LibTorchGenerationParamsTest {

    @Test
    void builderSetsAllDefaults() {
        LibTorchGenerationParams params = LibTorchGenerationParams.builder().build();
        assertEquals(512, params.getMaxTokens());
        assertEquals(0.8f, params.getTemperature(), 0.01f);
        assertEquals(0.95f, params.getTopP(), 0.01f);
        assertEquals(40, params.getTopK());
        assertEquals(1.1f, params.getRepeatPenalty(), 0.01f);
        assertEquals(64, params.getRepeatLastN());
    }

    @Test
    void builderSetsCustomValues() {
        LibTorchGenerationParams params = LibTorchGenerationParams.builder()
                .maxTokens(1024)
                .temperature(0.5f)
                .topP(0.9f)
                .topK(50)
                .repeatPenalty(1.2f)
                .repeatLastN(128)
                .build();

        assertEquals(1024, params.getMaxTokens());
        assertEquals(0.5f, params.getTemperature(), 0.01f);
        assertEquals(0.9f, params.getTopP(), 0.01f);
        assertEquals(50, params.getTopK());
        assertEquals(1.2f, params.getRepeatPenalty(), 0.01f);
        assertEquals(128, params.getRepeatLastN());
    }

    @Test
    void zeroTemperatureIsGreedy() {
        LibTorchGenerationParams params = LibTorchGenerationParams.builder()
                .temperature(0.0f)
                .build();
        assertEquals(0.0f, params.getTemperature(), 0.001f);
    }

    @Test
    void topKOfOneIsGreedy() {
        LibTorchGenerationParams params = LibTorchGenerationParams.builder()
                .topK(1)
                .build();
        assertEquals(1, params.getTopK());
    }
}
