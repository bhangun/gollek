/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.rag;

import java.util.List;

/**
 * Chunking service for splitting documents into retrieval-friendly chunks.
 */
public class ChunkingService {

    /**
     * Chunking strategy.
     */
    public enum Strategy {
        /** Fixed number of characters per chunk */
        FIXED_SIZE,
        /** Chunk at sentence boundaries */
        SENTENCE,
        /** Chunk at paragraph boundaries */
        PARAGRAPH
    }

    private final Strategy strategy;
    private final int chunkSize;
    private final int overlap;

    public ChunkingService() {
        this(Strategy.FIXED_SIZE, 512, 64);
    }

    public ChunkingService(Strategy strategy, int chunkSize, int overlap) {
        this.strategy = strategy;
        this.chunkSize = chunkSize;
        this.overlap = overlap;
    }

    /**
     * Chunk a document into smaller pieces.
     *
     * @param text the document text
     * @return list of text chunks
     */
    public List<String> chunk(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }

        return switch (strategy) {
            case FIXED_SIZE -> chunkFixedSize(text);
            case SENTENCE -> chunkBySentence(text);
            case PARAGRAPH -> chunkByParagraph(text);
        };
    }

    private List<String> chunkFixedSize(String text) {
        var chunks = new java.util.ArrayList<String>();
        int start = 0;
        while (start < text.length()) {
            int end = Math.min(start + chunkSize, text.length());
            chunks.add(text.substring(start, end));
            if (end == text.length()) {
                break;
            }
            start = end - overlap;
            // Safety check: ensure start always progresses to avoid infinite loops if
            // overlap >= chunkSize
            if (start <= 0 && end > 0 && overlap >= chunkSize) {
                start = end; // Force progress if overlap is invalid
            }
        }
        return chunks;
    }

    private List<String> chunkBySentence(String text) {
        var chunks = new java.util.ArrayList<String>();
        String[] sentences = text.split("(?<=[.!?])\\s+");
        var current = new StringBuilder();

        for (String sentence : sentences) {
            if (current.length() + sentence.length() > chunkSize && !current.isEmpty()) {
                chunks.add(current.toString().trim());
                current = new StringBuilder();
            }
            current.append(sentence).append(" ");
        }

        if (!current.isEmpty()) {
            chunks.add(current.toString().trim());
        }

        return chunks;
    }

    private List<String> chunkByParagraph(String text) {
        var chunks = new java.util.ArrayList<String>();
        String[] paragraphs = text.split("\\n\\s*\\n");
        var current = new StringBuilder();

        for (String para : paragraphs) {
            if (current.length() + para.length() > chunkSize && !current.isEmpty()) {
                chunks.add(current.toString().trim());
                current = new StringBuilder();
            }
            current.append(para).append("\n\n");
        }

        if (!current.isEmpty()) {
            chunks.add(current.toString().trim());
        }

        return chunks;
    }
}
