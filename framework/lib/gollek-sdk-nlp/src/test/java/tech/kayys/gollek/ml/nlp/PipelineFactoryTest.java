package tech.kayys.gollek.ml.nlp;

import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

public class PipelineFactoryTest {

    @Test
    public void testFactoryRegistration() {
        // Register a dummy task
        PipelineFactory.register("dummy-task", config -> new Pipeline<String, String>() {
            @Override
            public String process(String input) { return "processed_" + input; }
            @Override
            public String task() { return "dummy-task"; }
            @Override
            public String model() { return "model-123"; }
        });

        Set<String> tasks = PipelineFactory.availableTasks();
        assertTrue(tasks.contains("dummy-task"));

        Pipeline<String, String> pipeline = PipelineFactory.create("dummy-task", "model-123");
        assertNotNull(pipeline);
        assertEquals("processed_hello", pipeline.process("hello"));
    }

    @Test
    public void testBuiltInTasksExist() {
        Set<String> tasks = PipelineFactory.availableTasks();
        assertTrue(tasks.contains("text-generation"));
        assertTrue(tasks.contains("text-classification"));
        assertTrue(tasks.contains("embedding"));
    }

    @Test
    public void testUnknownTaskThrows() {
        assertThrows(PipelineException.class, () -> 
            PipelineFactory.create("invalid-task", "any"));
    }
}
