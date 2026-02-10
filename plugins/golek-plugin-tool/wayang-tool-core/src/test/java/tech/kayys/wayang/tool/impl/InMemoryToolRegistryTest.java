package tech.kayys.wayang.tool.impl;

import io.quarkus.test.junit.QuarkusTest;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import tech.kayys.wayang.tool.spi.Tool;
import tech.kayys.wayang.tool.spi.ToolRegistry;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@QuarkusTest
public class InMemoryToolRegistryTest {

    @Inject
    ToolRegistry toolRegistry;

    @Test
    public void testRegisterAndRetrieve() {
        Tool tool = new Tool() {
            @Override
            public String id() {
                return "test-tool";
            }

            @Override
            public String name() {
                return "Test Tool";
            }

            @Override
            public String description() {
                return "A test tool";
            }

            @Override
            public Map<String, Object> inputSchema() {
                return Collections.emptyMap();
            }

            @Override
            public io.smallrye.mutiny.Uni<Map<String, Object>> execute(Map<String, Object> arguments,
                    Map<String, Object> context) {
                return io.smallrye.mutiny.Uni.createFrom().item(Collections.emptyMap());
            }
        };

        toolRegistry.register(tool);

        Tool retrieved = toolRegistry.getTool("test-tool")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem()
                .getItem();

        Assertions.assertNotNull(retrieved);
        Assertions.assertEquals("test-tool", retrieved.id());
    }

    @Test
    public void testListTools() {
        List<Tool> tools = toolRegistry.listTools()
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem()
                .getItem();

        Assertions.assertNotNull(tools);
        // Size depends on test execution order/isolation, but should be at least 0
    }

    @Test
    public void testGetNonExistentTool() {
        toolRegistry.getTool("non-existent")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .assertFailedWith(IllegalArgumentException.class);
    }
}
