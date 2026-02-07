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
 * @author bhangun
 */

package tech.kayys.golek.core.inference;

import tech.kayys.golek.spi.inference.InferenceResponseInterface;
import tech.kayys.golek.spi.stream.StreamChunk;
import org.reactivestreams.Publisher;

import java.time.Instant;
import java.util.Map;

public final class StreamingResponse implements InferenceResponseInterface {

    private final Publisher<StreamChunk> publisher;

    public StreamingResponse(Publisher<StreamChunk> publisher) {
        this.publisher = publisher;
    }

    @Override
    public String getRequestId() {
        // Return a default or derived request ID
        return "stream-" + System.currentTimeMillis();
    }

    @Override
    public String getContent() {
        // Streaming responses don't have content directly, it's in the stream
        return "";
    }

    @Override
    public String getModel() {
        // Return a default or derived model name
        return "streaming-model";
    }

    @Override
    public int getTokensUsed() {
        // Return a default value, actual count would be computed from stream
        return 0;
    }

    @Override
    public long getDurationMs() {
        // Return a default value, actual duration would be computed
        return 0;
    }

    @Override
    public Instant getTimestamp() {
        return Instant.now();
    }

    @Override
    public Map<String, Object> getMetadata() {
        return Map.of(); // Empty map as default
    }

    @Override
    public boolean isStreaming() {
        return true;
    }

    public Publisher<StreamChunk> stream() {
        return publisher;
    }
}