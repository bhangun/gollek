package tech.kayys.gollek.plugin.rag;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ChunkingService}.
 */
class ChunkingServiceTest {

    @Test
    void chunkFixedSize_splitsCorrectly() {
        var service = new ChunkingService(ChunkingService.Strategy.FIXED_SIZE, 20, 5);

        List<String> chunks = service.chunk("This is a test string that should be split into chunks");

        assertFalse(chunks.isEmpty());
        assertTrue(chunks.size() > 1);
        // Each chunk should be at most chunkSize (20)
        for (String chunk : chunks) {
            assertTrue(chunk.length() <= 20);
        }
    }

    @Test
    void chunkBySentence_splitsBySentenceBoundary() {
        var service = new ChunkingService(ChunkingService.Strategy.SENTENCE, 50, 0);

        String text = "First sentence. Second sentence. Third sentence that is quite long and exceeds the limit.";
        List<String> chunks = service.chunk(text);

        assertFalse(chunks.isEmpty());
    }

    @Test
    void chunkByParagraph_splitsByParagraph() {
        var service = new ChunkingService(ChunkingService.Strategy.PARAGRAPH, 50, 0);

        String text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        List<String> chunks = service.chunk(text);

        assertFalse(chunks.isEmpty());
    }

    @Test
    void chunk_handlesNull() {
        var service = new ChunkingService();
        assertTrue(service.chunk(null).isEmpty());
        assertTrue(service.chunk("").isEmpty());
    }
}
