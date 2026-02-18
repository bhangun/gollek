package tech.kayys.gollek.plugin.prompt;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.Message;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link PromptControlPlugin}.
 */
class PromptControlPluginTest {

    private PromptControlPlugin plugin;

    @BeforeEach
    void setUp() {
        plugin = new PromptControlPlugin();
    }

    @Test
    void buildPrompt_formatsRolesCorrectly() {
        var messages = List.of(
                Message.system("You are a helpful assistant."),
                Message.user("Hello!"),
                Message.assistant("Hi there!")
        );

        String prompt = plugin.buildPrompt(messages);

        assertTrue(prompt.contains("### System:\nYou are a helpful assistant."));
        assertTrue(prompt.contains("### User:\nHello!"));
        assertTrue(prompt.contains("### Assistant:\nHi there!"));
    }

    @Test
    void buildPrompt_handlesEmptyMessages() {
        String prompt = plugin.buildPrompt(List.of());
        assertEquals("", prompt);
    }

    @Test
    void manageContextWindow_noTruncationNeeded() {
        var messages = List.of(
                Message.user("Short message.")
        );

        List<Message> result = plugin.manageContextWindow(messages, 100);
        assertEquals(1, result.size());
    }

    @Test
    void manageContextWindow_truncatesOldestWhenExceeded() {
        var messages = new ArrayList<Message>();
        messages.add(Message.system("Sys"));
        messages.add(Message.user("Old message that is quite long and should be dropped"));
        messages.add(Message.user("Recent"));

        // Budget only fits system + recent
        List<Message> result = plugin.manageContextWindow(messages, 10);

        // System should always be kept
        assertTrue(result.stream().anyMatch(m -> m.getRole() == Message.Role.SYSTEM));
        // Recent message should be kept
        assertTrue(result.stream().anyMatch(m -> "Recent".equals(m.getContent())));
    }

    @Test
    void injectDefaultSystemPrompt_addsWhenMissing() {
        var pluginWithDefault = new PromptControlPlugin();

        var messages = List.of(
                Message.user("Hello!")
        );

        // No default set, so no injection
        List<Message> result = pluginWithDefault.injectDefaultSystemPrompt(messages);
        assertEquals(1, result.size());
    }

    @Test
    void estimateTokens_roughEstimate() {
        assertEquals(0, plugin.estimateTokens(null));
        assertEquals(0, plugin.estimateTokens(""));
        assertEquals(3, plugin.estimateTokens("Hello World!")); // 12 chars / 4 = 3
    }
}
