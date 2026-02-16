import org.junit.jupiter.api.Test;

import tech.kayys.golek.converter.model.QuantizationType;

import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for QuantizationType enum.
 *
 * @author Bhangun
 */
class QuantizationTypeTest {

    @Test
    @DisplayName("Should have correct properties for F16")
    void testF16Properties() {
        QuantizationType f16 = QuantizationType.F16;

        assertEquals("f16", f16.getNativeName());
        assertEquals("16-bit floating point", f16.getDescription());
        assertEquals(QuantizationType.QualityLevel.VERY_HIGH, f16.getQualityLevel());
        assertEquals(2.0, f16.getCompressionRatio(), 0.01);
        assertTrue(f16.getUseCase().contains("quality"));
    }

    @Test
    @DisplayName("Should have correct properties for Q4_K_M")
    void testQ4KMProperties() {
        QuantizationType q4km = QuantizationType.Q4_K_M;

        assertEquals("q4_k_m", q4km.getNativeName());
        assertEquals("4-bit K-quant (medium)", q4km.getDescription());
        assertEquals(QuantizationType.QualityLevel.MEDIUM_HIGH, q4km.getQualityLevel());
        assertEquals(8.5, q4km.getCompressionRatio(), 0.01);
        assertTrue(q4km.getUseCase().contains("recommended"));
    }

    @Test
    @DisplayName("Should find quantization type by native name")
    void testFromNativeName() {
        assertEquals(QuantizationType.F16, QuantizationType.fromNativeName("f16"));
        assertEquals(QuantizationType.Q4_K_M, QuantizationType.fromNativeName("q4_k_m"));
        assertEquals(QuantizationType.Q8_0, QuantizationType.fromNativeName("q8_0"));
        assertNull(QuantizationType.fromNativeName("invalid"));
        assertNull(QuantizationType.fromNativeName(null));
        assertNull(QuantizationType.fromNativeName(""));
    }

    @Test
    @DisplayName("Should recommend appropriate quantization based on model size")
    void testRecommendation() {
        // Small model with quality priority
        QuantizationType smallQuality = QuantizationType.recommend(1.0, true);
        assertTrue(smallQuality == QuantizationType.F16 || smallQuality == QuantizationType.Q4_K_M);

        // Small model with size priority
        QuantizationType smallSize = QuantizationType.recommend(1.0, false);
        assertTrue(smallSize == QuantizationType.F16 || smallSize == QuantizationType.Q4_K_M);

        // Medium model with quality priority
        QuantizationType mediumQuality = QuantizationType.recommend(7.0, true);
        assertEquals(QuantizationType.Q5_K_M, mediumQuality);

        // Medium model with size priority
        QuantizationType mediumSize = QuantizationType.recommend(7.0, false);
        assertEquals(QuantizationType.Q4_K_M, mediumSize);

        // Large model with quality priority
        QuantizationType largeQuality = QuantizationType.recommend(15.0, true);
        assertEquals(QuantizationType.Q4_K_M, largeQuality);

        // Large model with size priority
        QuantizationType largeSize = QuantizationType.recommend(15.0, false);
        assertTrue(largeSize == QuantizationType.Q4_K_S || largeSize == QuantizationType.Q3_K_M);

        // Very large model
        QuantizationType veryLarge = QuantizationType.recommend(100.0, false);
        assertTrue(veryLarge == QuantizationType.Q4_K_S || veryLarge == QuantizationType.Q3_K_M);
    }

    @Test
    @DisplayName("Should have correct quality level scores")
    void testQualityLevelScores() {
        assertEquals(5.0, QuantizationType.QualityLevel.HIGHEST.getScore(), 0.01);
        assertEquals(4.0, QuantizationType.QualityLevel.VERY_HIGH.getScore(), 0.01);
        assertEquals(3.0, QuantizationType.QualityLevel.HIGH.getScore(), 0.01);
        assertEquals(2.5, QuantizationType.QualityLevel.MEDIUM_HIGH.getScore(), 0.01);
        assertEquals(2.0, QuantizationType.QualityLevel.MEDIUM.getScore(), 0.01);
        assertEquals(1.5, QuantizationType.QualityLevel.MEDIUM_LOW.getScore(), 0.01);
        assertEquals(1.0, QuantizationType.QualityLevel.LOW.getScore(), 0.01);
    }

    @Test
    @DisplayName("Should have consistent compression ratios")
    void testCompressionRatios() {
        for (QuantizationType type : QuantizationType.values()) {
            assertTrue(type.getCompressionRatio() > 0,
                    "Compression ratio should be positive for " + type.name());
        }

        // Verify some specific ratios
        assertEquals(1.0, QuantizationType.F32.getCompressionRatio(), 0.01);
        assertEquals(2.0, QuantizationType.F16.getCompressionRatio(), 0.01);
        assertEquals(8.5, QuantizationType.Q4_K_M.getCompressionRatio(), 0.01);
        assertEquals(4.0, QuantizationType.Q8_0.getCompressionRatio(), 0.01);
    }

    @Test
    @DisplayName("Should have quality levels in correct order")
    void testQualityOrdering() {
        // Quality levels should be ordered from highest to lowest
        assertTrue(
                QuantizationType.QualityLevel.HIGHEST.getScore() > QuantizationType.QualityLevel.VERY_HIGH.getScore());
        assertTrue(QuantizationType.QualityLevel.VERY_HIGH.getScore() > QuantizationType.QualityLevel.HIGH.getScore());
        assertTrue(
                QuantizationType.QualityLevel.HIGH.getScore() > QuantizationType.QualityLevel.MEDIUM_HIGH.getScore());
        assertTrue(
                QuantizationType.QualityLevel.MEDIUM_HIGH.getScore() > QuantizationType.QualityLevel.MEDIUM.getScore());
        assertTrue(
                QuantizationType.QualityLevel.MEDIUM.getScore() > QuantizationType.QualityLevel.MEDIUM_LOW.getScore());
        assertTrue(QuantizationType.QualityLevel.MEDIUM_LOW.getScore() > QuantizationType.QualityLevel.LOW.getScore());
    }
}