package tech.kayys.gollek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.Test;
import jakarta.inject.Inject;

@QuarkusTest
public class ProvidersCommandTest {

    @Inject
    ProvidersCommand providersCommand;

    @Test
    public void testProvidersCommandEmpty() {
        if (providersCommand.parentCommand == null) {
            providersCommand.parentCommand = new tech.kayys.gollek.cli.GollekCommand();
        }
        providersCommand.parentCommand.verbose = false;
        // Just verify it runs without exception
        providersCommand.run();
    }

    @Test
    public void testProvidersCommandVerbose() {
        if (providersCommand.parentCommand == null) {
            providersCommand.parentCommand = new tech.kayys.gollek.cli.GollekCommand();
        }
        providersCommand.parentCommand.verbose = true;
        // Just verify it runs without exception
        providersCommand.run();
    }
}
