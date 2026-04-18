package tech.kayys.gollek.gguf.tokenizer;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.gguf.loader.GGUFModel;

import java.lang.reflect.Method;
import java.lang.foreign.MemorySegment;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class GGUFTokenizerDetectionTest {

    private static GGUFModel makeModel(Map<String,Object> meta, List<String> tokens) {
        return new GGUFModel(1, meta, Collections.emptyList(), 0L, (MemorySegment) null, (java.lang.foreign.Arena) null);
    }

    @Test
    @DisplayName("Detects GPT-2 byte-level vocab (returns false)")
    void detectGpt2ByteLevel() throws Exception {
        List<String> tokens = new ArrayList<>();
        tokens.add("Ġhello");
        tokens.add("Ġworld");
        tokens.add("!");

        Map<String,Object> meta = new HashMap<>();
        meta.put("tokenizer.ggml.tokens", tokens);
        meta.put("tokenizer.ggml.model", "gpt2");

        GGUFTokenizer t = new GGUFTokenizer(makeModel(meta, tokens));

        // invoke private method detectRawUtf8Vocab
        Method m = GGUFTokenizer.class.getDeclaredMethod("detectRawUtf8Vocab", Map.class, String.class, String.class, List.class);
        m.setAccessible(true);
        boolean res = (boolean) m.invoke(t, meta, "llama", "gpt2", tokens);
        assertFalse(res, "Expected GPT-2 byte-level to be detected as not raw UTF-8");
    }

    @Test
    @DisplayName("Detects Qwen/Tiktoken raw UTF-8 vocab (returns true)")
    void detectQwenRawUtf8() throws Exception {
        List<String> tokens = new ArrayList<>();
        tokens.add(" hello");
        tokens.add(" world");
        tokens.add("!");

        Map<String,Object> meta = new HashMap<>();
        meta.put("tokenizer.ggml.tokens", tokens);
        meta.put("general.architecture", "qwen2");

        GGUFTokenizer t = new GGUFTokenizer(makeModel(meta, tokens));
        Method m = GGUFTokenizer.class.getDeclaredMethod("detectRawUtf8Vocab", Map.class, String.class, String.class, List.class);
        m.setAccessible(true);
        boolean res = (boolean) m.invoke(t, meta, "qwen2", "tiktoken", tokens);
        assertTrue(res, "Expected Qwen/Tiktoken vocab to be detected as raw UTF-8");
    }

    @Test
    @DisplayName("Detects cl100k style via encoding metadata")
    void detectCl100kEncoding() throws Exception {
        List<String> tokens = new ArrayList<>();
        tokens.add("hello");
        tokens.add(" world");

        Map<String,Object> meta = new HashMap<>();
        meta.put("tokenizer.ggml.tokens", tokens);
        meta.put("tokenizer.encoding", "cl100k_base");

        GGUFTokenizer t = new GGUFTokenizer(makeModel(meta, tokens));
        Method m = GGUFTokenizer.class.getDeclaredMethod("detectRawUtf8Vocab", Map.class, String.class, String.class, List.class);
        m.setAccessible(true);
        boolean res = (boolean) m.invoke(t, meta, "unknown", "unknown", tokens);
        assertTrue(res, "Expected cl100k_base encoding to be treated as raw UTF-8");
    }

    @Test
    @DisplayName("Detects tiktoken model hint via tokenizer.model metadata")
    void detectTiktokenModelHint() throws Exception {
        List<String> tokens = new ArrayList<>();
        tokens.add(" hello");
        tokens.add("world");

        Map<String,Object> meta = new HashMap<>();
        meta.put("tokenizer.ggml.tokens", tokens);
        meta.put("tokenizer.model", "tiktoken");

        GGUFTokenizer t = new GGUFTokenizer(makeModel(meta, tokens));
        Method m = GGUFTokenizer.class.getDeclaredMethod("detectRawUtf8Vocab", Map.class, String.class, String.class, List.class);
        m.setAccessible(true);
        boolean res = (boolean) m.invoke(t, meta, "unknown", "tiktoken", tokens);
        assertTrue(res, "Expected tokenizer.model=tiktoken to be treated as raw UTF-8");
    }

    @Test
    @DisplayName("Detects cl100k alias without _base suffix")
    void detectCl100kAlias() throws Exception {
        List<String> tokens = new ArrayList<>();
        tokens.add("hello");
        tokens.add(" world");

        Map<String,Object> meta = new HashMap<>();
        meta.put("tokenizer.ggml.tokens", tokens);
        meta.put("tokenizer.encoding", "cl100k");

        GGUFTokenizer t = new GGUFTokenizer(makeModel(meta, tokens));
        Method m = GGUFTokenizer.class.getDeclaredMethod("detectRawUtf8Vocab", Map.class, String.class, String.class, List.class);
        m.setAccessible(true);
        boolean res = (boolean) m.invoke(t, meta, "unknown", "unknown", tokens);
        assertTrue(res, "Expected cl100k to be treated as raw UTF-8");
    }

    @Test
    @DisplayName("Byte-level fallback: tokens with Ġ markers should be non-raw")
    void detectByteLevelFallback() throws Exception {
        List<String> tokens = new ArrayList<>();
        tokens.add("Ġthe");
        tokens.add("Ġquick");
        tokens.add("Ġbrown");

        Map<String,Object> meta = new HashMap<>();
        meta.put("tokenizer.ggml.tokens", tokens);

        GGUFTokenizer t = new GGUFTokenizer(makeModel(meta, tokens));
        Method m = GGUFTokenizer.class.getDeclaredMethod("detectRawUtf8Vocab", Map.class, String.class, String.class, List.class);
        m.setAccessible(true);
        boolean res = (boolean) m.invoke(t, meta, "unknown", "unknown", tokens);
        assertFalse(res, "Expected byte-level tokens (Ġ) to be detected as not raw UTF-8");
    }
}
