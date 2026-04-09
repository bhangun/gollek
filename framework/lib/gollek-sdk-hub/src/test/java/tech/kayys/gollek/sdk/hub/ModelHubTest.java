package tech.kayys.gollek.sdk.hub;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Linear;
import tech.kayys.gollek.ml.nn.Parameter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class ModelHubTest {

    @TempDir
    Path tempDir;

    @Test
    public void testApplyWeights() {
        Linear linear = new Linear(2, 2);
        // Initial weights are probably zero or random depending on implementation
        // Let's force some weights
        Map<String, GradTensor> weights = new HashMap<>();
        
        // Linear usually has "weight" and "bias" parameters
        float[] weightData = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] biasData = {0.5f, 0.5f};
        
        weights.put("weight", GradTensor.of(weightData, 2, 2));
        weights.put("bias", GradTensor.of(biasData, 2));
        
        ModelHub.applyWeights(linear, weights);
        
        Parameter w = linear.namedParameters().get("weight");
        Parameter b = linear.namedParameters().get("bias");
        
        assertArrayEquals(weightData, w.data().data(), 1e-6f);
        assertArrayEquals(biasData, b.data().data(), 1e-6f);
    }

    @Test
    public void testApplyWeightsWithPrefix() {
        Linear linear = new Linear(2, 2);
        Map<String, GradTensor> weights = new HashMap<>();
        
        float[] weightData = {1.0f, 2.0f, 3.0f, 4.0f};
        // Simulated prefix from a larger model dump (e.g. "model.layers.0.weight")
        weights.put("model.fc.weight", GradTensor.of(weightData, 2, 2));
        
        ModelHub.applyWeights(linear, weights);
        
        Parameter w = linear.namedParameters().get("weight");
        assertArrayEquals(weightData, w.data().data(), 1e-6f);
    }

    @Test
    public void testResolveLocalPath() throws IOException {
        Path modelDir = tempDir.resolve("my-model");
        Files.createDirectories(modelDir);
        Files.createFile(modelDir.resolve("model.safetensors"));
        
        // This should resolve because it's a direct path
        Map<String, GradTensor> weights = ModelHub.loadWeights(modelDir.toAbsolutePath().toString());
        assertNotNull(weights);
    }
}
