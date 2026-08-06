package tech.kayys.gollek.plugin.prompt;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link PromptTemplate}.
 */
class PromptTemplateTest {

    @Test
    void render_substitutesVariables() {
        var template = PromptTemplate.system("test", "Hello, {{name}}! Your task is: {{task}}");

        String result = template.render(Map.of("name", "Agent", "task", "Summarize"));

        assertEquals("Hello, Agent! Your task is: Summarize", result);
    }

    @Test
    void render_leavesUnknownVariables() {
        var template = PromptTemplate.user("q", "Answer: {{answer}}");

        String result = template.render(Map.of());

        assertEquals("Answer: {{answer}}", result);
    }

    @Test
    void factoryMethods_setCorrectRole() {
        assertEquals("system", PromptTemplate.system("s", "t").role());
        assertEquals("user", PromptTemplate.user("u", "t").role());
        assertEquals("tool", PromptTemplate.tool("tl", "t").role());
    }
}
