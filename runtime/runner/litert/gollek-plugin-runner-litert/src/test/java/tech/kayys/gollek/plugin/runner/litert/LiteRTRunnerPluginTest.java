package tech.kayys.gollek.plugin.runner.litert;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.plugin.runner.RequestType;
import tech.kayys.gollek.plugin.runner.RunnerContext;
import tech.kayys.gollek.plugin.runner.RunnerExecutionException;
import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LiteRTRunnerPluginTest {

    @Test
    void validateDoesNotRequireLegacyTfliteInterpreter() {
        LiteRTRunnerPlugin plugin = new LiteRTRunnerPlugin();

        assertTrue(plugin.validate().isValid());
        assertTrue(plugin.isAvailable());
    }

    @Test
    void supportedFormatsIncludeActualLiteRtModelFiles() {
        LiteRTRunnerPlugin plugin = new LiteRTRunnerPlugin();

        assertTrue(plugin.supportedFormats().contains(".litertlm"));
        assertTrue(plugin.supportedFormats().contains(".tflite"));
        assertTrue(plugin.supportedFormats().contains(".task"));
    }

    @Test
    void metadataReportsRealProviderFacade() {
        LiteRTRunnerPlugin plugin = new LiteRTRunnerPlugin();

        assertTrue(plugin.metadata().containsKey("provider"));
        assertTrue(plugin.metadata().containsKey("has_execution_runtime"));
    }

    @Test
    void executeBeforeInitializeThrows() {
        LiteRTRunnerPlugin plugin = new LiteRTRunnerPlugin();
        RunnerRequest request = RunnerRequest.builder()
                .type(RequestType.INFER)
                .parameter("prompt", "hello")
                .build();

        assertThrows(RunnerExecutionException.class, () -> plugin.execute(request, RunnerContext.empty()));
    }

    @Test
    void inferenceWithoutModelFailsInsteadOfReturningFakeSuccess() throws Exception {
        LiteRTRunnerPlugin plugin = new LiteRTRunnerPlugin();
        plugin.initialize(RunnerContext.empty());

        RunnerRequest request = RunnerRequest.builder()
                .type(RequestType.INFER)
                .parameter("prompt", "hello")
                .build();

        RunnerResult<?> result = plugin.execute(request, RunnerContext.empty());

        assertFalse(result.isSuccess());
        assertTrue(result.getErrorMessage().orElse("").contains("model id or model_path"));
    }

    @Test
    void inferenceWithMissingModelPathFailsBeforeProviderExecution() throws Exception {
        LiteRTRunnerPlugin plugin = new LiteRTRunnerPlugin();
        plugin.initialize(RunnerContext.empty());

        RunnerRequest request = RunnerRequest.builder()
                .type(RequestType.INFER)
                .parameter("prompt", "hello")
                .parameter("model_path", "/tmp/gollek-missing-model.litertlm")
                .build();

        RunnerResult<?> result = plugin.execute(request, RunnerContext.empty());

        assertFalse(result.isSuccess());
        assertTrue(result.getErrorMessage().orElse("").contains("does not exist"));
    }

    @Test
    void classificationDoesNotReturnPlaceholderSuccess() throws Exception {
        LiteRTRunnerPlugin plugin = new LiteRTRunnerPlugin();
        plugin.initialize(RunnerContext.empty());

        RunnerResult<?> result = plugin.execute(new RunnerRequest(RequestType.CLASSIFY), RunnerContext.empty());

        assertFalse(result.isSuccess());
        assertTrue(result.getErrorMessage().orElse("").contains("not implemented"));
    }
}
