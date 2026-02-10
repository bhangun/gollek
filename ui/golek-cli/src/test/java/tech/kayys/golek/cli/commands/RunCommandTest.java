package tech.kayys.golek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.model.ModelInfo;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;

import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@QuarkusTest
public class RunCommandTest {

    @Inject
    RunCommand runCommand;

    @InjectMock
    GolekSdk sdk;

    @Test
    public void testRunCommand() throws Exception {
        // Mock response
        InferenceResponse mockResponse = InferenceResponse.builder()
                .requestId("test-id")
                .model("test-model")
                .content("Blue like the ocean")
                .build();

        Mockito.when(sdk.createCompletion(any(InferenceRequest.class)))
                .thenReturn(mockResponse);

        Mockito.when(sdk.getModelInfo(eq("test-model")))
                .thenReturn(Optional.of(ModelInfo.builder().modelId("test-model").build()));

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

        // Execute
        runCommand.run();

        // Verify
        Mockito.verify(sdk).createCompletion(any(InferenceRequest.class));
    }
}
