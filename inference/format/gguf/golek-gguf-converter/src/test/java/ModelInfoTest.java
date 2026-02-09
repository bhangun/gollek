import org.junit.jupiter.api.Test;

import tech.kayys.golek.converter.model.ModelInfo;
import tech.kayys.golek.converter.model.QuantizationType;
import tech.kayys.golek.spi.model.ModelFormat;

import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ModelInfo class.
 *
 * @author Bhangun
 */
class ModelInfoTest {

        @Test
        @DisplayName("Should create ModelInfo with all parameters")
        void testModelInfoCreation() {
                ModelInfo info = ModelInfo.builder()
                                .modelType("llama")
                                .architecture("LlamaForCausalLM")
                                .parameterCount(7_241_748_480L)
                                .numLayers(32)
                                .hiddenSize(4096)
                                .vocabSize(32000)
                                .contextLength(4096)
                                .quantization("q4_k_m")
                                .fileSize(3_834_792_141L) // ~3.8GB
                                .format(ModelFormat.PYTORCH)
                                .build();

                assertEquals("llama", info.getModelType());
                assertEquals("LlamaForCausalLM", info.getArchitecture());
                assertEquals(7_241_748_480L, info.getParameterCount());
                assertEquals(32, info.getNumLayers());
                assertEquals(4096, info.getHiddenSize());
                assertEquals(32000, info.getVocabSize());
                assertEquals(4096, info.getContextLength());
                assertEquals("q4_k_m", info.getQuantization());
                assertEquals(3_834_792_141L, info.getFileSize());
                assertEquals(ModelFormat.PYTORCH, info.getFormat());
        }

        @Test
        @DisplayName("Should format parameter count correctly")
        void testParameterCountFormatting() {
                ModelInfo small = ModelInfo.builder()
                                .parameterCount(500_000L) // 0.5M
                                .build();
                assertEquals("500.0M", small.getParameterCountFormatted());

                ModelInfo medium = ModelInfo.builder()
                                .parameterCount(2_500_000_000L) // 2.5B
                                .build();
                assertEquals("2.5B", medium.getParameterCountFormatted());

                ModelInfo large = ModelInfo.builder()
                                .parameterCount(13_000_000_000L) // 13B
                                .build();
                assertEquals("13.0B", large.getParameterCountFormatted());

                ModelInfo zero = ModelInfo.builder()
                                .parameterCount(0L)
                                .build();
                assertEquals("Unknown", zero.getParameterCountFormatted());
        }

        @Test
        @DisplayName("Should format file size correctly")
        void testFileSizeFormatting() {
                ModelInfo small = ModelInfo.builder()
                                .fileSize(512 * 1024) // 512KB
                                .build();
                assertEquals("0.50 MB", small.getFileSizeFormatted());

                ModelInfo medium = ModelInfo.builder()
                                .fileSize(2L * 1024 * 1024 * 1024) // 2GB
                                .build();
                assertEquals("2.00 GB", medium.getFileSizeFormatted());

                ModelInfo large = ModelInfo.builder()
                                .fileSize(15L * 1024 * 1024 * 1024) // 15GB
                                .build();
                assertEquals("15.00 GB", large.getFileSizeFormatted());

                ModelInfo zero = ModelInfo.builder()
                                .fileSize(0L)
                                .build();
                assertEquals("Unknown", zero.getFileSizeFormatted());
        }

        @Test
        @DisplayName("Should calculate file size in GB correctly")
        void testFileSizeInGb() {
                ModelInfo info = ModelInfo.builder()
                                .fileSize(4_294_967_296L) // Exactly 4GB
                                .build();

                assertEquals(4.0, info.getFileSizeGb(), 0.01);
        }

        @Test
        @DisplayName("Should identify large models correctly")
        void testIsLargeModel() {
                ModelInfo small = ModelInfo.builder()
                                .parameterCount(5_000_000_000L) // 5B
                                .build();
                assertFalse(small.isLargeModel());

                ModelInfo large = ModelInfo.builder()
                                .parameterCount(12_000_000_000L) // 12B
                                .build();
                assertTrue(large.isLargeModel());

                ModelInfo huge = ModelInfo.builder()
                                .parameterCount(70_000_000_000L) // 70B
                                .build();
                assertTrue(huge.isLargeModel());
        }

        @Test
        @DisplayName("Should estimate memory requirements correctly")
        void testMemoryEstimation() {
                ModelInfo info = ModelInfo.builder()
                                .parameterCount(7_000_000_000L) // 7B parameters
                                .build();

                // Estimate for F16 (2 bytes per parameter)
                double f16Memory = info.estimateMemoryGb(QuantizationType.F16);
                assertTrue(f16Memory > 10.0); // Should be more than 14GB with overhead

                // Estimate for Q4_K_M (4/8.5 bytes per parameter)
                double q4kmMemory = info.estimateMemoryGb(QuantizationType.Q4_K_M);
                assertTrue(q4kmMemory > 3.0 && q4kmMemory < 8.0); // Should be reasonable

                // Estimate for Q8_0 (4/4 bytes per parameter)
                double q8Memory = info.estimateMemoryGb(QuantizationType.Q8_0);
                assertTrue(q8Memory > 25.0); // Should be close to original size with overhead
        }

        @Test
        @DisplayName("Should handle null values gracefully")
        void testNullValueHandling() {
                ModelInfo info = ModelInfo.builder().build(); // All fields null/default

                assertNull(info.getModelType());
                assertNull(info.getArchitecture());
                assertEquals(0, info.getParameterCount());
                assertEquals(0, info.getNumLayers());
                assertEquals(0, info.getHiddenSize());
                assertEquals(0, info.getVocabSize());
                assertEquals(0, info.getContextLength());
                assertNull(info.getQuantization());
                assertEquals(0, info.getFileSize());
                assertNull(info.getFormat());
        }

        @Test
        @DisplayName("Should generate meaningful toString representation")
        void testToString() {
                ModelInfo info = ModelInfo.builder()
                                .modelType("llama")
                                .architecture("LlamaForCausalLM")
                                .parameterCount(7_241_748_480L)
                                .fileSize(3_834_792_141L)
                                .format(ModelFormat.PYTORCH)
                                .build();

                String str = info.toString();
                assertTrue(str.contains("ModelInfo"));
                assertTrue(str.contains("type=llama"));
                assertTrue(str.contains("arch=LlamaForCausalLM"));
                assertTrue(str.contains("params=7.2B"));
                assertTrue(str.contains("size=3.83"));
                assertTrue(str.contains("format=PyTorch"));
        }

        @Test
        @DisplayName("Should handle unknown format gracefully")
        void testUnknownFormat() {
                ModelInfo info = ModelInfo.builder()
                                .format(ModelFormat.UNKNOWN)
                                .build();

                assertEquals(ModelFormat.UNKNOWN, info.getFormat());
        }
}