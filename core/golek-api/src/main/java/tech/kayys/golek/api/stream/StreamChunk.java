package tech.kayys.golek.api.stream;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;

/**
 * Single chunk in a streaming response
 */
public final class StreamChunk {

    private final String requestId;
    private final int index;
    private final String delta;
    private final boolean isFinal;
    private final Instant timestamp;

    @JsonCreator
    public StreamChunk(
            @JsonProperty("requestId") String requestId,
            @JsonProperty("index") int index,
            @JsonProperty("delta") String delta,
            @JsonProperty("isFinal") boolean isFinal,
            @JsonProperty("timestamp") Instant timestamp) {
        this.requestId = requestId;
        this.index = index;
        this.delta = delta;
        this.isFinal = isFinal;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
    }

    public String getRequestId() {
        return requestId;
    }

    public int getIndex() {
        return index;
    }

    public String getDelta() {
        return delta;
    }

    public boolean isFinal() {
        return isFinal;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public static StreamChunk of(String requestId, int index, String delta) {
        return new StreamChunk(requestId, index, delta, false, Instant.now());
    }

    public static StreamChunk finalChunk(String requestId, int index, String delta) {
        return new StreamChunk(requestId, index, delta, true, Instant.now());
    }

    @Override
    public String toString() {
        return "StreamChunk{" +
                "index=" + index +
                ", isFinal=" + isFinal +
                ", deltaLength=" + (delta != null ? delta.length() : 0) +
                '}';
    }
}