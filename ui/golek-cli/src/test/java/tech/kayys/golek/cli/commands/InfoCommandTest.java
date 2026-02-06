package tech.kayys.golek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

@QuarkusTest
public class InfoCommandTest {

    @Inject
    InfoCommand infoCommand;

    @Test
    public void testInfoCommand() {
        // Just verify it runs without exception for now
        infoCommand.run();
    }
}
