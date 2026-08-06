package tech.kayys.gollek.safetensor.engine.prompt;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.Message;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class PromptTemplateCompatTest {

    @Test
    void legacyFormattingStillSupportsGemma4ModelType() {
        String prompt = PromptTemplateCompat.format(
                List.of(Message.user("where is jakarta")),
                "gemma4_text");

        assertTrue(prompt.startsWith("<bos><|turn>user\nwhere is jakarta"));
    }
}
