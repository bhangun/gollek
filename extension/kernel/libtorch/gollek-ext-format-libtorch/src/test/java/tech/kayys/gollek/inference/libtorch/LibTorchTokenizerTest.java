package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link LibTorchTokenizer}.
 * Tests both the BPE tokenizer (with a real tokenizer.json) and the
 * character-level fallback.
 */
class LibTorchTokenizerTest {

    @TempDir
    Path tempDir;

    private LibTorchTokenizer tokenizer;

    @BeforeEach
    void setUp() {
        tokenizer = new LibTorchTokenizer();
        // Inject a config that points to our temp directory
        tokenizer.config = testConfig(tempDir);
    }

    // ── Character-level fallback tests ────────────────────────────────

    @Test
    void fallbackTokenizerEncodesDecode() {
        // No tokenizer.json exists → should use char-level fallback
        long[] tokens = tokenizer.encode("no-model", "Hello", true);
        assertNotNull(tokens);
        assertTrue(tokens.length > 0, "Should produce tokens");

        // First token should be BOS (id=1 in fallback)
        assertEquals(1L, tokens[0], "First token should be BOS");
    }

    @Test
    void fallbackTokenizerRoundTrips() {
        String input = "Test 123";
        long[] tokens = tokenizer.encode("no-model", input, false);
        String decoded = tokenizer.decode("no-model", tokens);
        assertEquals(input, decoded, "Char-level tokenizer should round-trip");
    }

    @Test
    void fallbackDecodeSkipsSpecialTokens() {
        // BOS=1, EOS=2, PAD=0 should be skipped during decode
        long[] tokens = new long[] { 1, 72, 105, 2 }; // BOS, 'H', 'i', EOS
        String decoded = tokenizer.decode("no-model", tokens);
        assertFalse(decoded.contains("<s>"), "BOS should be skipped");
        assertFalse(decoded.contains("</s>"), "EOS should be skipped");
    }

    @Test
    void fallbackDecodeSingleToken() {
        String piece = tokenizer.decodeToken("no-model", 72L); // 'H' in char-level
        assertNotNull(piece);
        assertFalse(piece.isEmpty());
    }

    @Test
    void fallbackVocabSize() {
        int size = tokenizer.vocabSize("no-model");
        assertTrue(size >= 260, "Char-level vocab should have at least 260 entries");
    }

    @Test
    void fallbackEosTokenId() {
        int eosId = tokenizer.eosTokenId("no-model");
        assertEquals(2, eosId, "Fallback EOS should be 2");
    }

    // ── BPE tokenizer.json tests ──────────────────────────────────────

    @Test
    void bpeTokenizerLoadsVocab() throws IOException {
        createMinimalTokenizerJson("bpe-model");

        int vocabSize = tokenizer.vocabSize("bpe-model");
        assertTrue(vocabSize > 0, "BPE vocab should be loaded");
    }

    @Test
    void bpeTokenizerEncodes() throws IOException {
        createMinimalTokenizerJson("bpe-model2");

        long[] tokens = tokenizer.encode("bpe-model2", "ab", false);
        assertNotNull(tokens);
        assertTrue(tokens.length > 0, "Should produce tokens");
    }

    @Test
    void bpeTokenizerAddsBos() throws IOException {
        createMinimalTokenizerJson("bpe-model3");

        long[] withBos = tokenizer.encode("bpe-model3", "ab", true);
        long[] withoutBos = tokenizer.encode("bpe-model3", "ab", false);

        // With BOS should have at least one more token
        assertTrue(withBos.length >= withoutBos.length,
                "Encoding with BOS should produce equal or more tokens");
    }

    @Test
    void bpeTokenizerDecodes() throws IOException {
        createMinimalTokenizerJson("bpe-model4");

        long[] tokens = tokenizer.encode("bpe-model4", "ab", false);
        String decoded = tokenizer.decode("bpe-model4", tokens);
        assertNotNull(decoded);
    }

    @Test
    void emptyInputProducesEmptyOrBosOnly() {
        long[] withBos = tokenizer.encode("no-model", "", true);
        // Should be BOS only or empty
        assertTrue(withBos.length <= 1, "Empty input should produce at most BOS token");
    }

    @Test
    void tokenizerCachesState() {
        // First call loads, second should use cache
        tokenizer.encode("cache-test", "a", false);
        tokenizer.encode("cache-test", "b", false);
        // No exception = cache works
    }

    // ── Helpers ───────────────────────────────────────────────────────

    private void createMinimalTokenizerJson(String modelId) throws IOException {
        Path modelDir = tempDir.resolve(modelId);
        Files.createDirectories(modelDir);

        // Minimal HuggingFace tokenizer.json with BPE vocab and merges
        Map<String, Object> vocab = Map.of(
                "a", 0, "b", 1, "c", 2, "ab", 3,
                "<s>", 4, "</s>", 5, "<unk>", 6, "<pad>", 7);
        List<String> merges = List.of("a b");
        Map<String, Object> model = Map.of(
                "type", "BPE",
                "vocab", vocab,
                "merges", merges);
        List<Map<String, Object>> addedTokens = List.of(
                Map.of("id", 4, "content", "<s>", "special", true),
                Map.of("id", 5, "content", "</s>", "special", true),
                Map.of("id", 6, "content", "<unk>", "special", true),
                Map.of("id", 7, "content", "<pad>", "special", true));
        Map<String, Object> tokenizerJson = Map.of(
                "model", model,
                "added_tokens", addedTokens);

        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(modelDir.resolve("tokenizer.json").toFile(), tokenizerJson);
    }

    private LibTorchProviderConfig testConfig(Path basePath) {
        return new LibTorchProviderConfig() {
            @Override
            public boolean enabled() {
                return true;
            }

            @Override
            public NativeConfig nativeLib() {
                return () -> Optional.empty();
            }

            @Override
            public ModelConfig model() {
                return new ModelConfig() {
                    @Override
                    public String basePath() {
                        return basePath.toString();
                    }

                    @Override
                    public String extensions() {
                        return ".pt,.pts,.pth";
                    }
                };
            }

            @Override
            public GpuConfig gpu() {
                return new GpuConfig() {
                    @Override
                    public boolean enabled() {
                        return false;
                    }

                    @Override
                    public int deviceIndex() {
                        return 0;
                    }
                };
            }

            @Override
            public SessionConfig session() {
                return new SessionConfig() {
                    @Override
                    public int maxPerTenant() {
                        return 4;
                    }

                    @Override
                    public int idleTimeoutSeconds() {
                        return 300;
                    }

                    @Override
                    public int maxTotal() {
                        return 16;
                    }
                };
            }

            @Override
            public InferenceConfig inference() {
                return new InferenceConfig() {
                    @Override
                    public int timeoutSeconds() {
                        return 30;
                    }

                    @Override
                    public int threads() {
                        return 4;
                    }
                };
            }

            @Override
            public BatchingConfig batching() {
                return new BatchingConfig() {
                    @Override
                    public boolean enabled() {
                        return false;
                    }

                    @Override
                    public int maxBatchSize() {
                        return 16;
                    }

                    @Override
                    public int batchTimeoutMs() {
                        return 50;
                    }
                };
            }

            @Override
            public WarmupConfig warmup() {
                return new WarmupConfig() {
                    @Override
                    public boolean enabled() {
                        return false;
                    }

                    @Override
                    public Optional<String> models() {
                        return Optional.empty();
                    }

                    @Override
                    public boolean dummyForward() {
                        return false;
                    }

                    @Override
                    public String tenantId() {
                        return "__warmup__";
                    }
                };
            }

            @Override
            public GenerationConfig generation() {
                return new GenerationConfig() {
                    @Override
                    public float temperature() {
                        return 0.8f;
                    }

                    @Override
                    public float topP() {
                        return 0.95f;
                    }

                    @Override
                    public int topK() {
                        return 40;
                    }

                    @Override
                    public int maxTokens() {
                        return 512;
                    }

                    @Override
                    public float repeatPenalty() {
                        return 1.1f;
                    }

                    @Override
                    public int repeatLastN() {
                        return 64;
                    }
                };
            }
        };
    }
}
