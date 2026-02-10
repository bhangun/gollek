/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.streaming;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.plugin.StreamingPlugin;
import tech.kayys.golek.spi.plugin.BackpressureMode;
import tech.kayys.golek.spi.stream.StreamChunk;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Plugin for streaming control and partial output management.
 * <p>
 * Transforms raw token streams into typed events (text chunks, partial tool
 * calls,
 * complete tool calls, final answers). Manages backpressure, cancel/interrupt,
 * and timeouts.
 */
@ApplicationScoped
public class StreamingControlPlugin implements StreamingPlugin {

    private static final Logger LOG = Logger.getLogger(StreamingControlPlugin.class);
    private static final String PLUGIN_ID = "tech.kayys/streaming-control";

    private boolean enabled = true;
    private Duration timeout = Duration.ofSeconds(30);
    private BackpressureMode backpressureMode = BackpressureMode.BUFFER;
    private int maxBufferSize = 1024;

    private final StreamTransformer transformer = new StreamTransformer();
    private final AtomicInteger chunkCount = new AtomicInteger(0);
    private final AtomicLong totalBytes = new AtomicLong(0);

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public void initialize(PluginContext context) {
        context.getConfig("enabled").ifPresent(v -> this.enabled = Boolean.parseBoolean(v));
        context.getConfig("timeoutSeconds").ifPresent(v -> this.timeout = Duration.ofSeconds(Integer.parseInt(v)));
        context.getConfig("backpressureMode").ifPresent(v -> this.backpressureMode = BackpressureMode.valueOf(v));
        context.getConfig("maxBufferSize").ifPresent(v -> this.maxBufferSize = Integer.parseInt(v));
        LOG.infof("Initialized %s (timeout: %ds, backpressure: %s)",
                PLUGIN_ID, timeout.getSeconds(), backpressureMode);
    }

    @Override
    public StreamChunk onChunk(StreamChunk chunk) {
        chunkCount.incrementAndGet();
        if (chunk.getDelta() != null) {
            totalBytes.addAndGet(chunk.getDelta().length());
        }

        // Transform the chunk (detect partial tool calls, etc.)
        return transformer.transform(chunk);
    }

    @Override
    public boolean detectPartialToolCall(String accumulatedText) {
        return transformer.hasPartialToolCall(accumulatedText);
    }

    @Override
    public void onCancel(String reason) {
        LOG.infof("Stream cancelled: %s (processed %d chunks, %d bytes)",
                reason != null ? reason : "unknown", chunkCount.get(), totalBytes.get());
        resetCounters();
    }

    @Override
    public void onComplete(int totalChunks) {
        LOG.debugf("Stream completed: %d chunks, %d bytes", totalChunks, totalBytes.get());
        resetCounters();
    }

    @Override
    public void onError(Throwable error) {
        LOG.warnf(error, "Stream error after %d chunks", chunkCount.get());
        resetCounters();
    }

    @Override
    public Duration timeout() {
        return timeout;
    }

    @Override
    public BackpressureMode backpressureMode() {
        return backpressureMode;
    }

    @Override
    public int maxBufferSize() {
        return maxBufferSize;
    }

    private void resetCounters() {
        chunkCount.set(0);
        totalBytes.set(0);
        transformer.reset();
    }
}
