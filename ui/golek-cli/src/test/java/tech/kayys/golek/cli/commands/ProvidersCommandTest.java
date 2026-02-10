package tech.kayys.golek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.Test;
import jakarta.inject.Inject;

@QuarkusTest
public class ProvidersCommandTest {

    @Inject
    ProvidersCommand providersCommand;

    @Test
    public void testProvidersCommandEmpty() {
        providersCommand.verbose = false;
        // Just verify it runs without exception
        providersCommand.run();
    }

    @Test
    public void testProvidersCommandVerbose() {
        providersCommand.verbose = true;
        // Just verify it runs without exception
        providersCommand.run();
    }
}
