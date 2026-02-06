package tech.kayys.golek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import io.smallrye.mutiny.Multi;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;
import tech.kayys.golek.provider.ollama.OllamaClient;
import tech.kayys.golek.provider.ollama.OllamaPullRequest;
import tech.kayys.golek.provider.ollama.OllamaPullProgress;

import static org.mockito.ArgumentMatchers.any;

@QuarkusTest
public class PullCommandTest {

    @Inject
    PullCommand pullCommand;

    @InjectMock
    OllamaClient ollamaClient;

    @Test
    public void testPullCommandOllama() {
        OllamaPullProgress progress = new OllamaPullProgress();
        progress.setStatus("pulling");
        progress.setTotal(100L);
        progress.setCompleted(100L);

        Mockito.when(ollamaClient.pullModel(any(OllamaPullRequest.class)))
                .thenReturn(Multi.createFrom().items(progress));

        pullCommand.modelSpec = "llama2";
        pullCommand.insecure = false;

        pullCommand.run();

        Mockito.verify(ollamaClient).pullModel(any(OllamaPullRequest.class));
    }

    @Test
    public void testPullCommandOllamaWithPrefix() {
        OllamaPullProgress progress = new OllamaPullProgress();
        progress.setStatus("complete");

        Mockito.when(ollamaClient.pullModel(any(OllamaPullRequest.class)))
                .thenReturn(Multi.createFrom().items(progress));

        pullCommand.modelSpec = "ollama:mistral";
        pullCommand.insecure = false;

        pullCommand.run();

        Mockito.verify(ollamaClient).pullModel(any(OllamaPullRequest.class));
    }

    @Test
    public void testPullCommandHuggingFace() {
        // HuggingFace pull is not implemented yet, should print error message
        pullCommand.modelSpec = "hf:TheBloke/Llama-2";
        pullCommand.insecure = false;

        pullCommand.run();

        // No interaction with ollamaClient expected
        Mockito.verifyNoInteractions(ollamaClient);
    }
}
