/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.gollek.plugin.streaming;

import tech.kayys.gollek.spi.stream.StreamChunk;

/**
 * Transforms raw token stream chunks into typed events.
 * <p>
 * Detects:
 * <ul>
 *   <li>Text content chunks</li>
 *   <li>Partial tool call patterns</li>
 *   <li>Complete tool call boundaries</li>
 *   <li>Stream completion signals</li>
 * </ul>
 */
public class StreamTransformer {

    private final StringBuilder accumulator = new StringBuilder();
    private boolean inToolCall = false;
    private int braceDepth = 0;

    /**
     * Transform a stream chunk, enriching it with detection metadata.
     *
     * @param chunk the incoming chunk
     * @return the transformed chunk
     */
    public StreamChunk transform(StreamChunk chunk) {
        String delta = chunk.getDelta();
        if (delta == null) return chunk;

        accumulator.append(delta);

        // Track brace depth for JSON tool call detection
        for (char c : delta.toCharArray()) {
            if (c == '{') {
                braceDepth++;
                if (!inToolCall && looksLikeToolCallStart()) {
                    inToolCall = true;
                }
            } else if (c == '}') {
                braceDepth--;
                if (inToolCall && braceDepth == 0) {
                    inToolCall = false;
                    // Tool call complete — could enrich metadata here
                }
            }
        }

        return chunk;
    }

    /**
     * Check if the accumulated text contains a partial tool call.
     */
    public boolean hasPartialToolCall(String text) {
        if (text == null) return false;
        return text.contains("\"tool_call\"") || text.contains("\"function_call\"")
                || text.contains("<tool_call>");
    }

    /**
     * Reset the transformer state for a new stream.
     */
    public void reset() {
        accumulator.setLength(0);
        inToolCall = false;
        braceDepth = 0;
    }

    private boolean looksLikeToolCallStart() {
        String acc = accumulator.toString();
        int lastBrace = acc.lastIndexOf('{');
        if (lastBrace < 0) return false;

        String before = acc.substring(Math.max(0, lastBrace - 50), lastBrace);
        return before.contains("tool_call") || before.contains("function_call");
    }
}
