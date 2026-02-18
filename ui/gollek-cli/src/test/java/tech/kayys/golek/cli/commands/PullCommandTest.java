package tech.kayys.gollek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;
import tech.kayys.gollek.sdk.core.GollekSdk;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@QuarkusTest
public class PullCommandTest {

    @Inject
    PullCommand pullCommand;

    @InjectMock
    GollekSdk sdk;

    @Test
    public void testPullCommandOllama() throws Exception {
        pullCommand.modelSpec = "llama2";
        pullCommand.insecure = false;

        pullCommand.run();

        Mockito.verify(sdk).pullModel(eq("llama2"), any());
    }

    @Test
    public void testPullCommandOllamaWithPrefix() throws Exception {
        pullCommand.modelSpec = "ollama:mistral";
        pullCommand.insecure = false;

        pullCommand.run();

        Mockito.verify(sdk).pullModel(eq("ollama:mistral"), any());
    }

    @Test
    public void testPullCommandHuggingFace() throws Exception {
        pullCommand.modelSpec = "hf:TheBloke/Llama-2";
        pullCommand.insecure = false;

        pullCommand.run();

        Mockito.verify(sdk).pullModel(eq("hf:TheBloke/Llama-2"), any());
    }
}
