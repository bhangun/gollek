package tech.kayys.gollek.safetensor.engine.generation.kv;

import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

class KVCacheAllocationTest {

    @Test
    void testProactiveAllocation() {
        BlockManager bm = new BlockManager();
        KVCacheManager manager = new KVCacheManager();
        manager.globalBlockManager = bm;
        
        KVCacheManager.KVCacheSession session = manager.createSession(128);
        
        ModelConfig config = new ModelConfig();
        // Set basic dimensions (using overrides as fields are private)
        config.setArchitectures(java.util.List.of("LlamaForCausalLM"));
        config.overrideNumAttentionHeads(1);
        config.overrideNumKeyValueHeads(1);
        config.overrideHeadDim(64);
        
        session.allocate(config, tech.kayys.gollek.safetensor.generation.GenerationConfig.defaults());
        
        // Initially 1 block allocated per layer (tokens 0-15)
        assertEquals(1, session.getBlockTable(0).size());
        
        // Proactively ensure capacity for 35 tokens (should need 3 blocks: 0-15, 16-31, 32-47)
        session.ensureCapacity(35);
        assertEquals(3, session.getBlockTable(0).size());
        
        // Ensure capacity for 16 tokens (already satisfied)
        session.ensureCapacity(16);
        assertEquals(3, session.getBlockTable(0).size());

        // Ensure capacity for 49 tokens (should need 4 blocks)
        session.ensureCapacity(49);
        assertEquals(4, session.getBlockTable(0).size());
    }

    @Test
    void testInt4KvAllocationUsesPackedBlocks() {
        BlockManager bm = new BlockManager();
        KVCacheManager manager = new KVCacheManager();
        manager.globalBlockManager = bm;

        KVCacheManager.KVCacheSession session = manager.createSession(128);

        ModelConfig config = new ModelConfig();
        config.setArchitectures(java.util.List.of("LlamaForCausalLM"));
        config.overrideNumAttentionHeads(1);
        config.overrideNumKeyValueHeads(1);
        config.overrideHeadDim(64);

        tech.kayys.gollek.safetensor.generation.GenerationConfig cfg =
                tech.kayys.gollek.safetensor.generation.GenerationConfig.builder()
                        .kvCacheQuant(tech.kayys.gollek.safetensor.generation.GenerationConfig.KvCacheQuantization.INT4)
                        .build();

        session.allocate(config, cfg);

        assertTrue(session.isQuantizedInt4());
        assertFalse(session.isQuantizedInt8());
        assertEquals(BlockManager.KvStorageType.INT4, session.blockManager().getStorageType());
        assertEquals(512, session.blockManager().getBlockSizeBytes(),
                "16 tokens * 64 dims packed to 4-bit should consume 512 bytes per block");
        assertNotNull(session.blockManager().getKScaleBlock(session.getBlockTable(0).getFirst()));
    }
}
