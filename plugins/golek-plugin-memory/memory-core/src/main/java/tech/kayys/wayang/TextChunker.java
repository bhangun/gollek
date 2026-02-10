package tech.kayys.gamelan.executor.memory;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Utilities for chunking text into smaller pieces for embedding
 */
@ApplicationScoped
public class TextChunker {

    private static final Logger LOG = LoggerFactory.getLogger(TextChunker.class);

    @ConfigProperty(name = "gamelan.embedding.chunk.size", defaultValue = "512")
    int chunkSize;

    @ConfigProperty(name = "gamelan.embedding.chunk.overlap", defaultValue = "50")
    int chunkOverlap;

    /**
     * Split text into chunks with overlap
     *
     * @param text Text to chunk
     * @return List of text chunks
     */
    public List<String> chunk(String text) {
        List<String> chunks = new ArrayList<>();

        if (text == null || text.isEmpty()) {
            return chunks;
        }

        // Split by sentences first
        String[] sentences = text.split("(?<=[.!?])\\s+");

        StringBuilder currentChunk = new StringBuilder();
        int currentLength = 0;

        for (String sentence : sentences) {
            int sentenceLength = sentence.length();

            if (currentLength + sentenceLength > chunkSize && currentLength > 0) {
                // Save current chunk
                chunks.add(currentChunk.toString().trim());

                // Start new chunk with overlap
                currentChunk = new StringBuilder();
                currentLength = 0;

                // Add overlap from previous chunk
                String prevChunk = chunks.get(chunks.size() - 1);
                if (prevChunk.length() > chunkOverlap) {
                    String overlap = prevChunk.substring(prevChunk.length() - chunkOverlap);
                    currentChunk.append(overlap).append(" ");
                    currentLength = overlap.length();
                }
            }

            currentChunk.append(sentence).append(" ");
            currentLength += sentenceLength;
        }

        // Add final chunk
        if (currentLength > 0) {
            chunks.add(currentChunk.toString().trim());
        }

        LOG.debug("Chunked text of {} chars into {} chunks", text.length(), chunks.size());

        return chunks;
    }

    /**
     * Estimate token count (rough approximation)
     *
     * @param text Text to count
     * @return Estimated token count
     */
    public int estimateTokenCount(String text) {
        // Rough approximation: 1 token ≈ 4 characters
        return text.length() / 4;
    }
}