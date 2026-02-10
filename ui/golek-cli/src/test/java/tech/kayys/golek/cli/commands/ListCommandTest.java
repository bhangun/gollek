package tech.kayys.golek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.model.ModelInfo;
import java.util.Collections;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@QuarkusTest
public class ListCommandTest {

        @Inject
        ListCommand listCommand;

        @InjectMock
        GolekSdk sdk;

        @Test
        public void testListCommandEmpty() throws Exception {
                Mockito.when(sdk.listModels(any(Integer.class), any(Integer.class)))
                                .thenReturn(Collections.emptyList());

                listCommand.format = "table";
                listCommand.limit = 50;

                listCommand.run();

                Mockito.verify(sdk).listModels(eq(0), eq(50));
        }

        @Test
        public void testListCommandWithModels() throws Exception {
                ModelInfo model = ModelInfo.builder()
                                .modelId("test-model")
                                .name("Test Model")
                                .version("1.0")
                                .tenantId("default")
                                .build();

                Mockito.when(sdk.listModels(any(Integer.class), any(Integer.class)))
                                .thenReturn(List.of(model));

                listCommand.format = "table";
                listCommand.limit = 50;

                listCommand.run();

                Mockito.verify(sdk).listModels(eq(0), eq(50));
        }

        @Test
        public void testListCommandJsonFormat() throws Exception {
                ModelInfo model = ModelInfo.builder()
                                .modelId("test-model")
                                .name("Test Model")
                                .version("1.0")
                                .tenantId("default")
                                .build();

                Mockito.when(sdk.listModels(any(Integer.class), any(Integer.class)))
                                .thenReturn(List.of(model));

                listCommand.format = "json";
                listCommand.limit = 50;

                listCommand.run();

                Mockito.verify(sdk).listModels(eq(0), eq(50));
        }
}
