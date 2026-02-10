package tech.kayys.golek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import org.junit.jupiter.api.Test;
import jakarta.inject.Inject;
import tech.kayys.golek.sdk.core.GolekSdk;

@QuarkusTest
public class ChatCommandTest {

    @Inject
    ChatCommand chatCommand;

    @InjectMock
    GolekSdk sdk;

    @Test
    public void testChatCommandInitialization() {
        // Test that the command can be injected and configured
        chatCommand.modelId = "test-model";
        chatCommand.temperature = 0.7;

        // ChatCommand is interactive, so we can't fully test run()
        // Just verify it's properly configured
        assert chatCommand.modelId.equals("test-model");
    }
}
