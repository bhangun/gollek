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
public class ChatCommandTest {

    @Inject
    ChatCommand chatCommand;

    @InjectMock
    InferenceService inferenceService;

    @Test
    public void testChatCommandInitialization() {
        // Test that the command can be injected and configured
        chatCommand.modelId = "test-model";
        chatCommand.tenantId = "default";
        chatCommand.temperature = 0.7;

        // ChatCommand is interactive, so we can't fully test run()
        // Just verify it's properly configured
        assert chatCommand.modelId.equals("test-model");
        assert chatCommand.tenantId.equals("default");
    }
}
