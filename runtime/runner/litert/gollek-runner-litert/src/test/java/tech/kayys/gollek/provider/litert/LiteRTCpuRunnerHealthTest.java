package tech.kayys.gollek.provider.litert;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LiteRTCpuRunnerHealthTest {

    @Test
    void uninitializedRunnerIsNotHealthy() {
        assertFalse(new LiteRTCpuRunner().health());
    }

    @Test
    void initializedLlmRunnerIsHealthyWithoutCompiledModel() throws Exception {
        LiteRTCpuRunner runner = new LiteRTCpuRunner();
        try {
            setField(runner, "initialized", true);
            setField(runner, "llmRunner",
                    new LiteRTInferenceRunner(null, Path.of("gemma-4-E2B-it.litertlm"), null, true, 4));

            assertTrue(runner.health());
        } finally {
            runner.close();
        }
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }
}
