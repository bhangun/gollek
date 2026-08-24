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

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.model.ModelPullRequest;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@QuarkusTest
public class PullCommandTest {

    @Inject
    PullCommand pullCommand;

    @InjectMock
    GollekSdk sdk;

    @Test
    public void testPullCommandWithSimpleName() throws Exception {
        pullCommand.modelSpec = "llama3.2";
        pullCommand.insecure = false;

        pullCommand.run();

        Mockito.verify(sdk).pullModel(any(ModelPullRequest.class), any());
    }

    @Test
    public void testPullCommandCustomRegistry() throws Exception {
        pullCommand.modelSpec = "registry:llama3.2";
        pullCommand.insecure = false;

        pullCommand.run();

        Mockito.verify(sdk).pullModel(any(ModelPullRequest.class), any());
    }
}
