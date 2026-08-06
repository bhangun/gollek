/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility for managing KV shards in a distributed environment.
 * Helps with layer-wise and sequence-wise partitioning.
 */
public class KVShardUtils {

    /**
     * Splits a model's layers into fixed-size shards for distribution.
     * 
     * @param numLayers total layers in the model
     * @param shardCount number of shards to split into
     * @return a list of shard IDs (e.g. "layers_0_7", "layers_8_15")
     */
    public static List<String> calculateLayerShards(int numLayers, int shardCount) {
        List<String> shards = new ArrayList<>();
        int layersPerShard = (int) Math.ceil((double) numLayers / shardCount);
        
        for (int i = 0; i < shardCount; i++) {
            int start = i * layersPerShard;
            int end = Math.min(start + layersPerShard - 1, numLayers - 1);
            if (start < numLayers) {
                shards.add(String.format("layers_%d_%d", start, end));
            }
        }
        return shards;
    }

    /**
     * Generates a shard-specific KVKey.
     */
    public static KVKey createShardKey(KVKey baseKey, String shardId) {
        return new KVKey(
            baseKey.modelId(),
            baseKey.prefixHash(),
            baseKey.tenantId(),
            shardId,
            baseKey.version()
        );
    }
}
