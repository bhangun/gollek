/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */

package tech.kayys.gollek.spi.plugin;

import tech.kayys.gollek.spi.stream.StreamChunk;

import java.time.Duration;

/**
 * SPI for plugins that control streaming and partial output handling.
 * <p>
 * Responsible for:
 * <ul>
 * <li>Token streaming transformation</li>
 * <li>Partial tool call detection and streaming</li>
 * <li>Cancel / interrupt handling</li>
 * <li>Backpressure management</li>
 * <li>Timeout control</li>
 * </ul>
 * <p>
 * llama.cpp only streams raw tokens; this plugin transforms them into
 * typed events: text chunks, tool calls, and structured events.
 */
public interface StreamingPlugin extends GollekPlugin {

    /**
     * Process a streaming chunk, potentially transforming or enriching it.
     *
     * @param chunk the incoming stream chunk
     * @return the processed chunk (may be modified, enriched, or filtered)
     */
    StreamChunk onChunk(StreamChunk chunk);

    /**
     * Handle a partial tool call detection during streaming.
     * Called when a partial tool call pattern is detected in the stream.
     *
     * @param accumulatedText the accumulated text so far in this stream
     * @return true if a partial tool call is detected
     */
    boolean detectPartialToolCall(String accumulatedText);

    /**
     * Called when the stream is cancelled by the client.
     * Perform any necessary cleanup.
     *
     * @param reason the cancellation reason (may be null)
     */
    void onCancel(String reason);

    /**
     * Called when the stream completes successfully.
     *
     * @param totalChunks total number of chunks processed
     */
    void onComplete(int totalChunks);

    /**
     * Called when the stream encounters an error.
     *
     * @param error the error that occurred
     */
    void onError(Throwable error);

    /**
     * Get the stream timeout duration.
     * Default: 30 seconds
     */
    default Duration timeout() {
        return Duration.ofSeconds(30);
    }

    /**
     * Get the backpressure strategy for this stream.
     * Default: BUFFER
     */
    default BackpressureMode backpressureMode() {
        return BackpressureMode.BUFFER;
    }

    /**
     * Get the maximum buffer size for backpressure buffering.
     * Default: 1024 chunks
     */
    default int maxBufferSize() {
        return 1024;
    }

}
