package tech.kayys.golek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;
import jakarta.enterprise.inject.Instance;
import tech.kayys.golek.spi.provider.LLMProvider;
import tech.kayys.golek.spi.provider.ProviderCapabilities;
import tech.kayys.golek.spi.provider.ProviderHealth;

import java.util.Collections;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

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
