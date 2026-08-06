package tech.kayys.gollek.plugin.runner.gguf;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.gguf.loader.GGUFModel;
import tech.kayys.alkhawarizm.gguf.loader.GGUFTensorInfo;
import tech.kayys.alkhawarizm.gguf.runtime.GgufTensorOps;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JavaNativeGgufBackendTest {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final String EXPLICIT_MIN_ROWS = "gollek.gguf.java_native.prepare_min_rows";
    private static final String AUTO_PREPARE = "gollek.gguf.java_native.auto_prepare";
    private static final String AUTO_MIN_ROWS = "gollek.gguf.java_native.auto_prepare_min_rows";
    private static final String AUTO_BUDGET_BYTES = "gollek.gguf.java_native.auto_prepare_budget_bytes";

    @Test
    void autoPreparesDecoderCachesWhenPlanFitsBudget() {
        Map<String, String> previous = saveProperties();
        setProperty(EXPLICIT_MIN_ROWS, null);
        setProperty(AUTO_PREPARE, "true");
        setProperty(AUTO_MIN_ROWS, "1");
        setProperty(AUTO_BUDGET_BYTES, "1k");

        try (Arena arena = Arena.ofShared()) {
            JavaNativeGgufBackend backend = new JavaNativeGgufBackend(decoderModel(arena), 52, 0);

            GgufTensorOps.PreparedMatrixCachePlan plan = backend.preparedCachePlan();
            GgufTensorOps.PreparedMatrixCacheStats stats = backend.preparedCacheStats();

            assertEquals(2, plan.preparedCandidates());
            assertEquals(72L, plan.estimatedPreparedBytes());
            assertTrue(stats.ready());
            assertEquals(2, stats.preparedTensors());
            assertEquals(2, stats.cacheEntries());
            assertEquals(72L, stats.cacheBytes());
            assertEquals("auto-prepared", backend.metadata().get("preparedMatrixCacheMode"));
            assertTrue(backend.preparedCacheSummary().contains("mode=auto-prepared"));

            backend.close();
        } finally {
            restoreProperties(previous);
        }
    }

    @Test
    void autoPreparesBudgetedDecoderCacheSubsetWhenFullPlanExceedsBudget() {
        Map<String, String> previous = saveProperties();
        setProperty(EXPLICIT_MIN_ROWS, null);
        setProperty(AUTO_PREPARE, "true");
        setProperty(AUTO_MIN_ROWS, "1");
        setProperty(AUTO_BUDGET_BYTES, "40");

        try (Arena arena = Arena.ofShared()) {
            JavaNativeGgufBackend backend = new JavaNativeGgufBackend(decoderModel(arena), 52, 0);

            GgufTensorOps.PreparedMatrixCachePlan plan = backend.preparedCachePlan();
            GgufTensorOps.PreparedMatrixCacheStats stats = backend.preparedCacheStats();

            assertEquals(1, plan.preparedCandidates());
            assertEquals(36L, plan.estimatedPreparedBytes());
            assertTrue(stats.ready());
            assertEquals(1, stats.preparedTensors());
            assertEquals(1, stats.cacheEntries());
            assertEquals(36L, stats.cacheBytes());
            assertEquals("auto-budget-prepared", backend.metadata().get("preparedMatrixCacheMode"));
            assertTrue(backend.preparedCacheSummary().contains("mode=auto-budget-prepared"));

            backend.close();
        } finally {
            restoreProperties(previous);
        }
    }

    @Test
    void autoSkipsDecoderCachePreparationWhenPlanExceedsBudget() {
        Map<String, String> previous = saveProperties();
        setProperty(EXPLICIT_MIN_ROWS, null);
        setProperty(AUTO_PREPARE, "true");
        setProperty(AUTO_MIN_ROWS, "1");
        setProperty(AUTO_BUDGET_BYTES, "1");

        try (Arena arena = Arena.ofShared()) {
            JavaNativeGgufBackend backend = new JavaNativeGgufBackend(decoderModel(arena), 52, 0);

            GgufTensorOps.PreparedMatrixCachePlan plan = backend.preparedCachePlan();
            GgufTensorOps.PreparedMatrixCacheStats stats = backend.preparedCacheStats();

            assertEquals(2, plan.preparedCandidates());
            assertEquals(72L, plan.estimatedPreparedBytes());
            assertEquals(0, stats.scannedTensors());
            assertEquals(0, stats.preparedTensors());
            assertEquals("auto-skipped-budget", backend.metadata().get("preparedMatrixCacheMode"));
            assertTrue(backend.preparedCacheSummary().contains("mode=auto-skipped-budget"));

            backend.close();
        } finally {
            restoreProperties(previous);
        }
    }

    @Test
    void explicitDecoderCachePreparationKeepsExistingForceBehavior() {
        Map<String, String> previous = saveProperties();
        setProperty(EXPLICIT_MIN_ROWS, "1");
        setProperty(AUTO_PREPARE, "false");
        setProperty(AUTO_MIN_ROWS, "1");
        setProperty(AUTO_BUDGET_BYTES, "1");

        try (Arena arena = Arena.ofShared()) {
            JavaNativeGgufBackend backend = new JavaNativeGgufBackend(decoderModel(arena), 52, 0);

            GgufTensorOps.PreparedMatrixCacheStats stats = backend.preparedCacheStats();

            assertTrue(stats.ready());
            assertEquals(2, stats.preparedTensors());
            assertEquals(72L, stats.cacheBytes());
            assertEquals("explicit-prepared", backend.metadata().get("preparedMatrixCacheMode"));
            assertTrue(backend.preparedCacheSummary().contains("mode=explicit-prepared"));

            backend.close();
        } finally {
            restoreProperties(previous);
        }
    }

    private static GGUFModel decoderModel(Arena arena) {
        MemorySegment segment = arena.allocate(52);
        writeQ4_0Block(segment.asSlice(0, 18), (byte) 0x98);
        writeQ8Block(segment.asSlice(18, 34), (byte) 1);
        GGUFTensorInfo q4 = new GGUFTensorInfo(
                "blk.0.attn_q.weight",
                new long[]{32, 1},
                2,
                0,
                18);
        GGUFTensorInfo q8 = new GGUFTensorInfo(
                "blk.0.attn_v.weight",
                new long[]{32, 1},
                8,
                18,
                34);
        GGUFTensorInfo ignored = new GGUFTensorInfo(
                "token_embd.weight",
                new long[]{32, 1},
                8,
                18,
                34);
        return new GGUFModel(3, Map.of(), List.of(q4, q8, ignored), 0, segment, null);
    }

    private static void writeQ4_0Block(MemorySegment block, byte packedQuant) {
        block.set(LE_SHORT, 0, (short) 0x3c00);
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, packedQuant);
        }
    }

    private static void writeQ8Block(MemorySegment block, byte quant) {
        block.set(LE_SHORT, 0, (short) 0x3c00);
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, quant);
        }
    }

    private static Map<String, String> saveProperties() {
        Map<String, String> previous = new HashMap<>();
        previous.put(EXPLICIT_MIN_ROWS, System.getProperty(EXPLICIT_MIN_ROWS));
        previous.put(AUTO_PREPARE, System.getProperty(AUTO_PREPARE));
        previous.put(AUTO_MIN_ROWS, System.getProperty(AUTO_MIN_ROWS));
        previous.put(AUTO_BUDGET_BYTES, System.getProperty(AUTO_BUDGET_BYTES));
        return previous;
    }

    private static void restoreProperties(Map<String, String> previous) {
        previous.forEach(JavaNativeGgufBackendTest::setProperty);
    }

    private static void setProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }
}
