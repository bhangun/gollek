package tech.kayys.golek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;
import tech.kayys.golek.model.core.LocalModelRepository;
import tech.kayys.golek.model.core.Pageable;
import tech.kayys.golek.spi.model.ModelManifest;

import java.time.Instant;
import java.util.Collections;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@QuarkusTest
public class ListCommandTest {

    @Inject
    ListCommand listCommand;

    @InjectMock
    LocalModelRepository modelRepository;

    @Test
    public void testListCommandEmpty() {
        Mockito.when(modelRepository.list(any(String.class), any(Pageable.class)))
                .thenReturn(Uni.createFrom().item(Collections.emptyList()));

        listCommand.tenantId = "default";
        listCommand.format = "table";
        listCommand.limit = 50;

        listCommand.run();

        Mockito.verify(modelRepository).list(eq("default"), any(Pageable.class));
    }

    @Test
    public void testListCommandWithModels() {
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

        Mockito.when(modelRepository.list(any(String.class), any(Pageable.class)))
                .thenReturn(Uni.createFrom().item(List.of(model)));

        listCommand.tenantId = "default";
        listCommand.format = "table";
        listCommand.limit = 50;

        listCommand.run();

        Mockito.verify(modelRepository).list(eq("default"), any(Pageable.class));
    }

    @Test
    public void testListCommandJsonFormat() {
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

        Mockito.when(modelRepository.list(any(String.class), any(Pageable.class)))
                .thenReturn(Uni.createFrom().item(List.of(model)));

        listCommand.tenantId = "default";
        listCommand.format = "json";
        listCommand.limit = 50;

        listCommand.run();

        Mockito.verify(modelRepository).list(eq("default"), any(Pageable.class));
    }
}
