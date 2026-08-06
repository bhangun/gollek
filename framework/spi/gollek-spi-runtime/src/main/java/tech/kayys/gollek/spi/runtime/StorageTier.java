/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Storage tiers for the distributed KV memory fabric.
 * Defines the latency and accessibility of a KV shard.
 */
public enum StorageTier {
    RAM,       // System RAM (Local CPU)
    GPU,       // VRAM (Local GPU)
    DISK,      // Local compressed/mapped storage
    REMOTE     // Another node in the cluster
}
