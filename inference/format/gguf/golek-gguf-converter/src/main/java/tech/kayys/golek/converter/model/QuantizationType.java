package tech.kayys.golek.converter.model;

import lombok.Getter;

/**
 * GGUF quantization types with metadata.
 * 
 * <p>
 * Provides information about each quantization method including
 * quality level, compression ratio, and use cases.
 * 
 * @author Bhangun
 * @version 1.0.0
 */
@Getter
public enum QuantizationType {

        // No quantization
        F32("f32", "32-bit floating point",
                        QualityLevel.HIGHEST, 1.0, "Maximum precision, largest size"),

        F16("f16", "16-bit floating point",
                        QualityLevel.VERY_HIGH, 2.0, "Near-original quality with 50% size reduction"),

        // 4-bit quantization
        Q4_0("q4_0", "4-bit quantization (fast)",
                        QualityLevel.MEDIUM, 8.0, "Good balance of size and quality"),

        Q4_1("q4_1", "4-bit quantization (higher quality)",
                        QualityLevel.MEDIUM_HIGH, 7.5, "Better quality than Q4_0, slightly larger"),

        // 5-bit quantization
        Q5_0("q5_0", "5-bit quantization (fast)",
                        QualityLevel.MEDIUM_HIGH, 6.4, "Better than 4-bit with minimal size increase"),

        Q5_1("q5_1", "5-bit quantization (higher quality)",
                        QualityLevel.HIGH, 6.0, "Excellent quality-to-size ratio"),

        // 8-bit quantization
        Q8_0("q8_0", "8-bit quantization",
                        QualityLevel.VERY_HIGH, 4.0, "Minimal quality loss, 4x compression"),

        Q8_1("q8_1", "8-bit quantization (higher quality)",
                        QualityLevel.VERY_HIGH, 3.8, "Near-lossless with good compression"),

        // K-quants (advanced quantization with better quality)
        Q2_K("q2_k", "2-bit K-quant",
                        QualityLevel.LOW, 16.0, "Extreme compression, noticeable quality loss"),

        Q3_K_S("q3_k_s", "3-bit K-quant (small)",
                        QualityLevel.MEDIUM_LOW, 12.0, "Aggressive compression with acceptable quality"),

        Q3_K_M("q3_k_m", "3-bit K-quant (medium)",
                        QualityLevel.MEDIUM, 11.0, "Balanced 3-bit quantization"),

        Q3_K_L("q3_k_l", "3-bit K-quant (large)",
                        QualityLevel.MEDIUM, 10.5, "Higher quality 3-bit quantization"),

        Q4_K_S("q4_k_s", "4-bit K-quant (small)",
                        QualityLevel.MEDIUM, 9.0, "Recommended for most use cases"),

        Q4_K_M("q4_k_m", "4-bit K-quant (medium)",
                        QualityLevel.MEDIUM_HIGH, 8.5, "Best overall quality-to-size ratio"),

        Q5_K_S("q5_k_s", "5-bit K-quant (small)",
                        QualityLevel.HIGH, 7.0, "High quality with good compression"),

        Q5_K_M("q5_k_m", "5-bit K-quant (medium)",
                        QualityLevel.HIGH, 6.5, "Excellent quality for production use"),

        Q6_K("q6_k", "6-bit K-quant",
                        QualityLevel.VERY_HIGH, 5.3, "Near-original quality with 5x compression");

        private final String nativeName;
        private final String description;
        private final QualityLevel qualityLevel;
        private final double compressionRatio;
        private final String useCase;

        QuantizationType(String nativeName, String description,
                        QualityLevel qualityLevel, double compressionRatio, String useCase) {
                this.nativeName = nativeName;
                this.description = description;
                this.qualityLevel = qualityLevel;
                this.compressionRatio = compressionRatio;
                this.useCase = useCase;
        }

        /**
         * Find quantization type by native name.
         * 
         * @param nativeName native name (e.g., "q4_k_m")
         * @return quantization type or null if not found
         */
        public static QuantizationType fromNativeName(String nativeName) {
                if (nativeName == null) {
                        return null;
                }

                String normalized = nativeName.toLowerCase().trim();
                for (QuantizationType type : values()) {
                        if (type.nativeName.equals(normalized)) {
                                return type;
                        }
                }
                return null;
        }

        /**
         * Get recommended quantization type for a given model size and use case.
         * 
         * @param modelSizeGb       model size in gigabytes
         * @param prioritizeQuality true to prioritize quality over size
         * @return recommended quantization type
         */
        public static QuantizationType recommend(double modelSizeGb, boolean prioritizeQuality) {
                if (modelSizeGb < 3.0) {
                        // Small models: can afford higher precision
                        return prioritizeQuality ? F16 : Q4_K_M;
                } else if (modelSizeGb < 7.0) {
                        // Medium models: balance quality and size
                        return prioritizeQuality ? Q5_K_M : Q4_K_M;
                } else if (modelSizeGb < 13.0) {
                        // Large models: favor compression
                        return prioritizeQuality ? Q4_K_M : Q4_K_S;
                } else {
                        // Very large models: aggressive compression
                        return prioritizeQuality ? Q4_K_S : Q3_K_M;
                }
        }

        /**
         * Quality level classification.
         */
        public enum QualityLevel {
                HIGHEST(5),
                VERY_HIGH(4),
                HIGH(3),
                MEDIUM_HIGH(2.5),
                MEDIUM(2),
                MEDIUM_LOW(1.5),
                LOW(1);

                @Getter
                private final double score;

                QualityLevel(double score) {
                        this.score = score;
                }
        }
}
