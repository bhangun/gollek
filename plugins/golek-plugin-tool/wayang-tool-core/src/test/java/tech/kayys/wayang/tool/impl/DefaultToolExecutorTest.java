package tech.kayys.wayang.tool.impl;

import io.quarkus.test.InjectMock;
import io.quarkus.test.junit.QuarkusTest;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;
import jakarta.inject.Inject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import tech.kayys.wayang.tool.spi.Tool;
import tech.kayys.wayang.tool.spi.ToolExecutor;
import tech.kayys.wayang.tool.spi.ToolRegistry;

import java.util.Collections;
import java.util.Map;

@QuarkusTest
public class DefaultToolExecutorTest {

    @Inject
    ToolExecutor toolExecutor;

    @InjectMock
    ToolRegistry toolRegistry;

    @BeforeEach
    void setup() {
        Tool mockTool = Mockito.mock(Tool.class);
        Mockito.when(mockTool.execute(Mockito.any(), Mockito.any()))
                .thenReturn(Uni.createFrom().item(Map.of("result", "success")));

        Mockito.when(toolRegistry.getTool("mock-tool"))
                .thenReturn(Uni.createFrom().item(mockTool));

        Mockito.when(toolRegistry.getTool("missing-tool"))
                .thenReturn(Uni.createFrom().failure(new IllegalArgumentException("Not found")));
    }

    @Test
    public void testExecuteSuccess() {
        tech.kayys.wayang.tool.dto.ToolExecutionResult result = toolExecutor.execute("mock-tool", Map.of(), Map.of())
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem()
                .getItem();

        org.junit.jupiter.api.Assertions.assertEquals("success", result.output().get("result"));
    }

    @Test
    public void testExecuteMissingTool() {
        toolExecutor.execute("missing-tool", Map.of(), Map.of())
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .assertFailedWith(IllegalArgumentException.class);
    }
}
