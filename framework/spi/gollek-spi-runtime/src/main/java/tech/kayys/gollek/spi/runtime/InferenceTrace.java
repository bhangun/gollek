/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.time.Instant;

/**
 * Token-level audit record for distributed inference execution.
 * Captures the exact processing state to enable deterministic replay
 * and bottleneck analysis.
 * 
 * @param requestId     unique ID of the request
 * @param tokenId       ID of the generated/input token
 * @param nodeId        ID of the node that processed the token
 * @param kernelType    attention kernel used (FLASH, PAGED, etc.)
 * @param kvKey            identifier of the KV shard used
 * @param modelAttribution mapping of model IDs to their contribution weight for this token
 * @param expertAttribution mapping of expert IDs to their gating weights (for MoE models)
 * @param latencyNs        nanoseconds taken to process this token
 * @param timestamp        time of processing
 * @param isRemote         whether attention was executed remotely
 */
public record InferenceTrace(
    String requestId,
    int tokenId,
    String nodeId,
    KernelType kernelType,
    KVKey kvKey,
    java.util.Map<String, Double> modelAttribution,
    java.util.Map<String, Float> expertAttribution,
    long latencyNs,
    Instant timestamp,
    boolean isRemote
) {
    /**
     * Minimal record for local tracking.
     */
    public static InferenceTrace local(String requestId, int tokenId, long latencyNs) {
        return new InferenceTrace(
            requestId, tokenId, "local", KernelType.STANDARD, 
            null, java.util.Map.of("local", 1.0), java.util.Map.of(), 
            latencyNs, Instant.now(), false);
    }
}
