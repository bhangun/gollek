package tech.kayys.golek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;
import tech.kayys.golek.engine.inference.InferenceService;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;

import static org.mockito.ArgumentMatchers.any;

@QuarkusTest
public class RunCommandTest {

    @Inject
    RunCommand runCommand;

    @InjectMock
    InferenceService inferenceService;

    @Test
    public void testRunCommand() {
        // Mock response
        InferenceResponse mockResponse = InferenceResponse.builder()
                .requestId("test-id")
                .model("test-model")
                .content("Blue like the ocean")
                .build();

        Mockito.when(inferenceService.inferAsync(any(InferenceRequest.class)))
                .thenReturn(Uni.createFrom().item(mockResponse));

        // Set CLI options directly as they are injected by picocli,
        // but here we are testing the Runnable logic as a bean.
        // Since fields are package-private or private and injected by picocli,
        // we might need to rely on reflection or integration tests if we can't set them
        // easily.
        // However, looking at the previous implementation, the fields were
        // package-private.

        // Actually, Picocli fields are often private. Let's check the source again.
        // If they are package-private, I can set them. If they are private, I need
        // reflection.
        // I wrote them as package-private (default visibility) in the previous step.
        // "String modelId;" etc.

        runCommand.modelId = "test-model";
        runCommand.prompt = "Why is the sky blue?";
        runCommand.tenantId = "test-tenant";

        // Execute
        runCommand.run();

        // Verify
        Mockito.verify(inferenceService).inferAsync(any(InferenceRequest.class));
    }
}
