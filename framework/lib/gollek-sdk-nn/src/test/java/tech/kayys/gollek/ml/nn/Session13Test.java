package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.inference.KVCache;
import tech.kayys.gollek.ml.inference.TokenSampler;
import tech.kayys.gollek.ml.models.DDPM;
import tech.kayys.gollek.ml.registry.ModelRegistry;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Session 13 tests: DDPM, TokenSampler, KVCache, ModelRegistry.
 */
class Session13Test {

    // ── DDPM ──────────────────────────────────────────────────────────────

    @Test
    void ddpmTrainingLossIsPositive() {
        DDPM ddpm = new DDPM(16, 32, 100); // small for speed
        GradTensor x0 = GradTensor.rand(4, 16); // [0,1] data
        GradTensor loss = ddpm.trainingLoss(x0);
        assertEquals(0, loss.ndim());
        assertTrue(loss.item() > 0f);
        assertTrue(Float.isFinite(loss.item()));
    }

    @Test
    void ddpmSampleOutputShape() {
        DDPM ddpm = new DDPM(8, 16, 10); // tiny for speed
        GradTensor samples = ddpm.sample(3);
        assertArrayEquals(new long[]{3, 8}, samples.shape());
    }

    @Test
    void ddpmHasParameters() {
        DDPM ddpm = new DDPM(16, 32, 100);
        assertTrue(ddpm.parameterCount() > 0);
    }

    // ── TokenSampler ──────────────────────────────────────────────────────

    @Test
    void greedyReturnsArgmax() {
        float[] logits = {0.1f, 0.5f, 0.9f, 0.2f};
        assertEquals(2, TokenSampler.greedy(logits));
    }

    @Test
    void temperatureSamplingInRange() {
        float[] logits = new float[100];
        for (int i = 0; i < logits.length; i++) logits[i] = (float) Math.random();
        int tok = TokenSampler.temperature(logits, 1.0f);
        assertTrue(tok >= 0 && tok < logits.length);
    }

    @Test
    void topKSamplingInTopK() {
        float[] logits = {10f, 1f, 1f, 1f, 1f}; // token 0 dominates
        // With k=1, must return token 0
        assertEquals(0, TokenSampler.topK(logits, 1, 1.0f));
    }

    @Test
    void topPSamplingInRange() {
        float[] logits = new float[50];
        for (int i = 0; i < logits.length; i++) logits[i] = (float) Math.random();
        int tok = TokenSampler.topP(logits, 0.9f, 1.0f);
        assertTrue(tok >= 0 && tok < logits.length);
    }

    @Test
    void beamSearchReturnsSequence() {
        // Simple logit function: always predicts token 1 most likely
        java.util.function.Function<int[], float[]> logitsFn = ids -> {
            float[] l = new float[5]; l[1] = 10f; return l;
        };
        int[] result = TokenSampler.beamSearch(logitsFn, new int[]{0}, 2, 5, 3);
        assertNotNull(result);
        assertTrue(result.length > 1);
    }

    // ── KVCache ───────────────────────────────────────────────────────────

    @Test
    void kvCacheUpdateAndRetrieve() {
        try (KVCache cache = new KVCache(2, 4, 8, 32)) {
            GradTensor k = GradTensor.randn(4, 1, 8); // [nHeads, 1, headDim]
            GradTensor v = GradTensor.randn(4, 1, 8);
            cache.update(0, k, v);

            assertEquals(1, cache.seqLen(0));
            assertArrayEquals(new long[]{4, 1, 8}, cache.getKeys(0).shape());
            assertArrayEquals(new long[]{4, 1, 8}, cache.getValues(0).shape());
        }
    }

    @Test
    void kvCacheAccumulatesTokens() {
        try (KVCache cache = new KVCache(1, 2, 4, 16)) {
            for (int i = 0; i < 5; i++) {
                cache.update(0, GradTensor.randn(2, 1, 4), GradTensor.randn(2, 1, 4));
            }
            assertEquals(5, cache.seqLen(0));
            assertArrayEquals(new long[]{2, 5, 4}, cache.getKeys(0).shape());
        }
    }

    @Test
    void kvCacheReset() {
        try (KVCache cache = new KVCache(1, 2, 4, 16)) {
            cache.update(0, GradTensor.randn(2, 1, 4), GradTensor.randn(2, 1, 4));
            cache.reset();
            assertEquals(0, cache.seqLen(0));
        }
    }

    @Test
    void kvCacheFullThrows() {
        try (KVCache cache = new KVCache(1, 1, 2, 2)) {
            cache.update(0, GradTensor.randn(1, 1, 2), GradTensor.randn(1, 1, 2));
            cache.update(0, GradTensor.randn(1, 1, 2), GradTensor.randn(1, 1, 2));
            assertThrows(IllegalStateException.class,
                () -> cache.update(0, GradTensor.randn(1, 1, 2), GradTensor.randn(1, 1, 2)));
        }
    }

    // ── ModelRegistry ─────────────────────────────────────────────────────

    @Test
    void registryRegisterAndLoad(@TempDir Path tmpDir) throws IOException {
        ModelRegistry registry = new ModelRegistry(tmpDir);
        NNModule model = new Sequential(new Linear(4, 2));

        String version = registry.register("test-model", model,
            Map.of("accuracy", "0.95"));

        assertNotNull(version);
        assertTrue(registry.modelNames().contains("test-model"));

        Map<String, GradTensor> loaded = registry.load("test-model", version);
        assertEquals(model.stateDict().keySet(), loaded.keySet());
    }

    @Test
    void registryTagAndLoadByTag(@TempDir Path tmpDir) throws IOException {
        ModelRegistry registry = new ModelRegistry(tmpDir);
        NNModule model = new Sequential(new Linear(4, 2));

        String version = registry.register("clf", model, Map.of());
        registry.tag(version, "production");

        Map<String, GradTensor> loaded = registry.load("clf", "production");
        assertNotNull(loaded);
    }

    @Test
    void registryListVersions(@TempDir Path tmpDir) throws IOException {
        ModelRegistry registry = new ModelRegistry(tmpDir);
        NNModule model = new Sequential(new Linear(2, 2));

        registry.register("m", model, Map.of("v", "1"));
        try {
            Thread.sleep(2); // ensure different timestamps
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        registry.register("m", model, Map.of("v", "2"));

        assertEquals(2, registry.list("m").size());
    }

    @Test
    void registryUnknownTagThrows(@TempDir Path tmpDir) throws IOException {
        ModelRegistry registry = new ModelRegistry(tmpDir);
        registry.register("m", new Sequential(new Linear(2, 2)), Map.of());
        assertThrows(java.util.NoSuchElementException.class,
            () -> registry.load("m", "nonexistent-tag"));
    }
}
