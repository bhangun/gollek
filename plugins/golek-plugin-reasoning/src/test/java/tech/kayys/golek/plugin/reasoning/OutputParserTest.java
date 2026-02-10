package tech.kayys.golek.plugin.reasoning;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import tech.kayys.golek.spi.tool.ToolCall;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link OutputParser}.
 */
class OutputParserTest {

    private OutputParser parser;

    @BeforeEach
    void setUp() {
        parser = new OutputParser();
    }

    @Test
    void parseToolCalls_detectsJsonToolCall() {
        String output = """
                I need to search for that.
                {"tool_call": {"name": "search", "arguments": {"query": "test"}}}
                """;

        List<ToolCall> calls = parser.parseToolCalls(output);

        assertEquals(1, calls.size());
        assertEquals("search", calls.get(0).getFunction().getName());
    }

    @Test
    void parseToolCalls_returnsEmptyForPlainText() {
        List<ToolCall> calls = parser.parseToolCalls("Just a regular answer.");
        assertTrue(calls.isEmpty());
    }

    @Test
    void parseToolCalls_handlesNull() {
        assertTrue(parser.parseToolCalls(null).isEmpty());
        assertTrue(parser.parseToolCalls("").isEmpty());
    }

    @Test
    void parseToolCalls_detectsXmlToolCall() {
        String output = """
                <tool_call>
                  <name>calculator</name>
                  <arguments>{"a": 1, "b": 2}</arguments>
                </tool_call>
                """;

        List<ToolCall> calls = parser.parseToolCalls(output);

        assertEquals(1, calls.size());
        assertEquals("calculator", calls.get(0).getFunction().getName());
    }

    @Test
    void isMalformed_detectsTruncatedJson() {
        assertTrue(parser.isMalformed("{\"tool_call\": {\"name\": \"test\""));
        assertFalse(parser.isMalformed("Normal text"));
    }

    @Test
    void extractFinalAnswer_stripsThinkingBlocks() {
        String output = "<think>Let me reason about this...</think>The answer is 42.";

        String answer = parser.extractFinalAnswer(output);

        assertEquals("The answer is 42.", answer);
    }

    @Test
    void extractFinalAnswer_handlesNull() {
        assertEquals("", parser.extractFinalAnswer(null));
    }
}
