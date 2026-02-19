package tech.kayys.gollek.spi.inference;

import org.junit.jupiter.api.Test;
import java.time.Duration;
import java.util.Collections;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class BatchingSPITest {

    @Test
    void testBatchConfigValidation() {
        BatchConfig config = BatchConfig.defaultDynamic();

        // This should pass
        config.validate();

        BatchConfig invalidBatchSize = new BatchConfig(
                BatchStrategy.DYNAMIC, 0, Duration.ofMillis(50), 4, 4, 128, false);
        assertThatThrownBy(invalidBatchSize::validate)
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("maxBatchSize must be > 0");

        BatchConfig invalidWaitTime = new BatchConfig(
                BatchStrategy.DYNAMIC, 8, Duration.ofMillis(-10), 4, 4, 128, false);
        assertThatThrownBy(invalidWaitTime::validate)
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("maxWaitTime must be >= 0");
    }

    @Test
    void testBatchResponseCounts() {
        BatchResult success = new BatchResult("1",
                InferenceResponse.builder().requestId("req-1").content("success").build(), null);
        BatchResult failure = new BatchResult("2", null,
                tech.kayys.gollek.spi.error.ErrorPayload.builder().type("ERROR").message("test").build());

        BatchResponse response = new BatchResponse("batch-1", java.util.List.of(success, failure),
                new BatchMetrics(2, 1, 0, 0));

        assertThat(response.successCount()).isEqualTo(1);
        assertThat(response.failureCount()).isEqualTo(1);
    }
}
