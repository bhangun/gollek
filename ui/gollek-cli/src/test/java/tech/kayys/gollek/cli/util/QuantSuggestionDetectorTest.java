package tech.kayys.gollek.cli.util;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies that {@link QuantSuggestionDetector} correctly extracts parameter
 * counts from model name strings.
 */
class QuantSuggestionDetectorTest {

    @Test
    @DisplayName("Parse standard B-suffix patterns")
    void testStandardPatterns() {
        assertEquals(7.0, QuantSuggestionDetector.parseParamCount("Qwen2.5-7B-Instruct"));
        assertEquals(70.0, QuantSuggestionDetector.parseParamCount("Llama-3.1-70B-Instruct"));
        assertEquals(13.0, QuantSuggestionDetector.parseParamCount("Llama-2-13b-chat-hf"));
        assertEquals(0.5, QuantSuggestionDetector.parseParamCount("Qwen2.5-0.5B-Instruct"));
        assertEquals(1.5, QuantSuggestionDetector.parseParamCount("Qwen2.5-1.5B-Instruct"));
        assertEquals(3.8, QuantSuggestionDetector.parseParamCount("Phi-3.5-mini-3.8B"));
        assertEquals(72.0, QuantSuggestionDetector.parseParamCount("Qwen2.5-72B-Instruct"));
        assertEquals(2.0, QuantSuggestionDetector.parseParamCount("gemma-2b-it"));
        assertEquals(8.0, QuantSuggestionDetector.parseParamCount("Meta-Llama-3-8B"));
    }

    @Test
    @DisplayName("Returns -1 for models without B-suffix")
    void testNoParamCount() {
        assertEquals(-1, QuantSuggestionDetector.parseParamCount("gpt-4o"));
        assertEquals(-1, QuantSuggestionDetector.parseParamCount("claude-3-opus"));
        assertEquals(-1, QuantSuggestionDetector.parseParamCount(null));
        assertEquals(-1, QuantSuggestionDetector.parseParamCount(""));
    }

    @Test
    @DisplayName("Picks largest B-value when multiple present")
    void testMultipleMatches() {
        // "Qwen2.5" has "2.5" but no B, then "7B" — should pick 7
        assertEquals(7.0, QuantSuggestionDetector.parseParamCount("Qwen2.5-7B-Instruct"));
    }

    @Test
    @DisplayName("Handles HuggingFace paths with org prefix")
    void testHuggingFacePaths() {
        assertEquals(7.0, QuantSuggestionDetector.parseParamCount("meta-llama/Llama-2-7b-chat-hf"));
        assertEquals(70.0, QuantSuggestionDetector.parseParamCount("Qwen/Qwen2.5-72B-Instruct".replace("72", "70")));
    }

    @Test
    @DisplayName("suggestIfNeeded skips when --quantize is already set")
    void testSkipsWhenQuantizeSet() {
        boolean result = QuantSuggestionDetector.suggestIfNeeded(
                "Llama-3.1-70B-Instruct", null, "bnb", false);
        assertFalse(result, "Should not suggest when --quantize is already set");
    }

    @Test
    @DisplayName("suggestIfNeeded returns true for 7B+ models")
    void testSuggestsFor7BPlus() {
        boolean result = QuantSuggestionDetector.suggestIfNeeded(
                "Llama-3.1-70B-Instruct", null, null, true); // quiet=true to suppress output
        // quiet=true suppresses output AND returns false
        assertFalse(result, "quiet mode should suppress suggestions");
    }
}
