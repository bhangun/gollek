package tech.kayys.golek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;
import tech.kayys.golek.model.core.LocalModelRepository;
import tech.kayys.golek.spi.model.ModelManifest;

import java.time.Instant;
import java.util.Collections;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@QuarkusTest
public class ShowCommandTest {

    @Inject
    ShowCommand showCommand;

    @InjectMock
    LocalModelRepository modelRepository;

    @Test
    public void testShowCommandModelFound() {
        ModelManifest model = new ModelManifest(
                "test-model",
                "Test Model",
                "1.0",
                "default",
                Collections.emptyMap(),
                Collections.emptyList(),
                null,
                Collections.emptyMap(),
                Instant.now(),
                Instant.now());

        Mockito.when(modelRepository.findById(any(String.class), any(String.class)))
                .thenReturn(Uni.createFrom().item(model));

        showCommand.modelId = "test-model";
        showCommand.tenantId = "default";

        showCommand.run();

        Mockito.verify(modelRepository).findById(eq("test-model"), eq("default"));
    }

    @Test
    public void testShowCommandModelNotFound() {
        Mockito.when(modelRepository.findById(any(String.class), any(String.class)))
                .thenReturn(Uni.createFrom().nullItem());

        showCommand.modelId = "nonexistent";
        showCommand.tenantId = "default";

        showCommand.run();

        Mockito.verify(modelRepository).findById(eq("nonexistent"), eq("default"));
    }
}
