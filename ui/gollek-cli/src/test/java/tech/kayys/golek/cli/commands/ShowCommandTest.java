package tech.kayys.gollek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.core.model.ModelInfo;

import java.time.Instant;
import java.util.Collections;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@QuarkusTest
public class ShowCommandTest {

    @Inject
    ShowCommand showCommand;

    @InjectMock
    GollekSdk sdk;

    @Test
    public void testShowCommandModelFound() throws Exception {
        ModelInfo model = ModelInfo.builder()
                .modelId("test-model")
                .name("Test Model")
                .version("1.0")
                .apiKey("community")
                .format("GGUF")
                .metadata(Collections.emptyMap())
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();

        Mockito.when(sdk.getModelInfo(eq("test-model")))
                .thenReturn(Optional.of(model));

        showCommand.modelId = "test-model";

        showCommand.run();

        Mockito.verify(sdk).getModelInfo(eq("test-model"));
    }

    @Test
    public void testShowCommandModelNotFound() throws Exception {
        Mockito.when(sdk.getModelInfo(any(String.class)))
                .thenReturn(Optional.empty());

        showCommand.modelId = "nonexistent";

        showCommand.run();

        Mockito.verify(sdk).getModelInfo(eq("nonexistent"));
    }
}
