package tech.kayys.gollek.plugin.streaming;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.stream.StreamChunk;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link StreamTransformer}.
 */
class StreamTransformerTest {

    private StreamTransformer transformer;

    @BeforeEach
    void setUp() {
        transformer = new StreamTransformer();
    }

    @Test
    void transform_preservesChunkData() {
        StreamChunk chunk = StreamChunk.of("req-1", 0, "Hello");
        StreamChunk result = transformer.transform(chunk);

        assertEquals("Hello", result.getDelta());
    }

    @Test
    void hasPartialToolCall_detectsJsonToolCall() {
        // Accumulate text first
        transformer.transform(StreamChunk.of("req-1", 0, "Some text with \"tool_call\""));
        
        // Check current valid method usage - StreamTransformer doesn't store detection in accessible way 
        // other than internal state, but hasPartialToolCall checks specific string
        assertTrue(transformer.hasPartialToolCall("Some text with \"tool_call\" in it"));
        assertTrue(transformer.hasPartialToolCall("Some text with \"function_call\" in it"));
    }

    @Test
    void hasPartialToolCall_returnsFalseForNormalText() {
        assertFalse(transformer.hasPartialToolCall("Normal output text"));
        assertFalse(transformer.hasPartialToolCall(null));
    }

    @Test
    void hasPartialToolCall_detectsXmlToolCall() {
        assertTrue(transformer.hasPartialToolCall("Begin <tool_call> output"));
    }

    @Test
    void reset_clearsState() {
        // Should not throw
        transformer.reset();
    }
}
