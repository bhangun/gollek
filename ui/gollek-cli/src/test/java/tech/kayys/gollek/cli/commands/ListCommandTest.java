package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.sdk.route.RunnerRouteBenchmarkCache;
import tech.kayys.gollek.sdk.route.RunnerRoutePolicy;
import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.safetensor.engine.route.*;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;
import tech.kayys.gollek.cli.commands.ListCommand;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.sdk.model.ModelListRequest;

import java.util.Collections;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@QuarkusTest
public class ListCommandTest {

        @Inject
        ListCommand listCommand;

        @InjectMock
        GollekSdk sdk;

        @Test
        public void testListCommandEmpty() throws Exception {
                Mockito.when(sdk.listModels(any(ModelListRequest.class)))
                                .thenReturn(Collections.emptyList());

                listCommand.format = "table";
                listCommand.limit = 50;

                listCommand.run();

                Mockito.verify(sdk).listModels(any(ModelListRequest.class));
        }

        @Test
        public void testListCommandWithModels() throws Exception {
                ModelInfo model = ModelInfo.builder()
                                .modelId("test-model")
                                .name("Test Model")
                                .version("1.0")
                                .build();

                Mockito.when(sdk.listModels(any(ModelListRequest.class)))
                                .thenReturn(List.of(model));

                listCommand.format = "table";
                listCommand.limit = 50;

                listCommand.run();

                Mockito.verify(sdk).listModels(any(ModelListRequest.class));
        }

        @Test
        public void testListCommandJsonFormat() throws Exception {
                ModelInfo model = ModelInfo.builder()
                                .modelId("test-model")
                                .name("Test Model")
                                .version("1.0")
                                .build();

                Mockito.when(sdk.listModels(any(ModelListRequest.class)))
                                .thenReturn(List.of(model));

                listCommand.format = "json";
                listCommand.limit = 50;

                listCommand.run();

                Mockito.verify(sdk).listModels(any(ModelListRequest.class));
        }
}
