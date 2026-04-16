package tech.kayys.gollek.cache;

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.cache.PromptPrefixHasher;
import tech.kayys.gollek.cache.CachedKVEntry;
import tech.kayys.gollek.cache.PrefixHash;
import tech.kayys.gollek.cache.PromptCacheStore;
import tech.kayys.gollek.cache.InProcessPromptCacheStore;
import tech.kayys.gollek.cache.NoOpPromptCacheStore;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the Gollek prompt-cache module.
 *
 * <p>Tests are organised into nested classes per component so they can be
 * run independently. No Quarkus container is required for most tests —
 * only the store integration tests use {@code @QuarkusTest}.
 */
class PromptCacheTest {

    // =========================================================================
    // PromptPrefixHasher
    // =========================================================================

    @Nested
    @DisplayName("PromptPrefixHasher")
    class HasherTests {

        // Minimal KVCacheConfig stub
        private PromptPrefixHasher hasher;

        @BeforeEach
        void setup() {
            hasher = HasherTestHelper.hasherWithBlockSize(16, "xxhash64");
        }

        @Test
        @DisplayName("empty token array returns empty boundaries")
        void emptyTokenIds() {
            assertTrue(hasher.hashBoundaries(new int[0]).isEmpty());
        }

        @Test
        @DisplayName("prompt shorter than one block returns no boundaries")
        void shortPrompt() {
            int[] tokens = new int[10]; // < blockSize=16
            assertTrue(hasher.hashBoundaries(tokens).isEmpty());
        }

        @Test
        @DisplayName("exactly one block returns one boundary at length 16")
        void oneBlock() {
            int[] tokens = sameTokens(16);
            Map<Integer, Long> b = hasher.hashBoundaries(tokens);
            assertEquals(1, b.size());
            assertTrue(b.containsKey(16));
        }

        @Test
        @DisplayName("48-token prompt returns boundaries at 16, 32, 48")
        void threeBlocks() {
            int[] tokens = sameTokens(48);
            Map<Integer, Long> b = hasher.hashBoundaries(tokens);
            assertEquals(3, b.size());
            assertTrue(b.containsKey(16));
            assertTrue(b.containsKey(32));
            assertTrue(b.containsKey(48));
        }

        @Test
        @DisplayName("same token sequence → identical hashes (deterministic)")
        void deterministic() {
            int[] t1 = sequentialTokens(32);
            int[] t2 = sequentialTokens(32);
            assertEquals(
                hasher.hashBoundaries(t1).get(32),
                hasher.hashBoundaries(t2).get(32)
            );
        }

        @Test
        @DisplayName("different token sequences → different hashes at same boundary")
        void differentInputs() {
            int[] t1 = sequentialTokens(32);
            int[] t2 = reverseTokens(32);
            assertNotEquals(
                hasher.hashBoundaries(t1).get(32),
                hasher.hashBoundaries(t2).get(32)
            );
        }

        @Test
        @DisplayName("shared prefix produces identical hash at shared boundary")
        void sharedPrefix() {
            int[] base   = sequentialTokens(48);
            int[] longer = new int[64];
            System.arraycopy(base, 0, longer, 0, 48);
            for (int i = 48; i < 64; i++) longer[i] = 999;

            Map<Integer, Long> b1 = hasher.hashBoundaries(base);
            Map<Integer, Long> b2 = hasher.hashBoundaries(longer);

            // Hash at position 32 must be identical for both sequences
            assertEquals(b1.get(32), b2.get(32), "Shared prefix hash at boundary 32 must match");
            assertEquals(b1.get(48), b2.get(48), "Shared prefix hash at boundary 48 must match");
        }

        @Test
        @DisplayName("tail tokens do not affect boundary hashes before them")
        void tailIndependence() {
            int[] base  = sequentialTokens(32);
            int[] withTail = new int[35]; // 32 + 3 extra (not on boundary)
            System.arraycopy(base, 0, withTail, 0, 32);
            withTail[32] = 100; withTail[33] = 101; withTail[34] = 102;

            assertEquals(
                hasher.hashBoundaries(base).get(32),
                hasher.hashBoundaries(withTail).get(32),
                "Non-boundary tail must not affect prior boundary hash"
            );
        }

        @Test
        @DisplayName("murmur3 algorithm produces different (valid) hashes")
        void murmur3() {
            PromptPrefixHasher m3 = HasherTestHelper.hasherWithBlockSize(16, "murmur3");
            Map<Integer, Long> b = m3.hashBoundaries(sequentialTokens(32));
            assertEquals(2, b.size());
            assertNotNull(b.get(16));
            assertNotNull(b.get(32));
        }
    }

    // =========================================================================
    // PrefixHash record
    // =========================================================================

    @Nested
    @DisplayName("PrefixHash")
    class PrefixHashTests {

        @Test
        @DisplayName("storageKey format is correct")
        void storageKey() {
            PrefixHash k = new PrefixHash("llama3", 0xABCDL, 32, "global");
            String key = k.storageKey();
            assertTrue(key.startsWith("gollek:pc:global:llama3:32:"), "Key format: " + key);
        }

        @Test
        @DisplayName("withScope returns new record with updated scope")
        void withScope() {
            PrefixHash k = new PrefixHash("llama3", 0xABCDL, 32);
            PrefixHash s = k.withScope("session:abc");
            assertEquals("session:abc", s.scope());
            assertEquals("global", k.scope()); // original unchanged
        }

        @Test
        @DisplayName("null modelId throws")
        void nullModelId() {
            assertThrows(NullPointerException.class,
                () -> new PrefixHash(null, 0L, 16, "global"));
        }

        @Test
        @DisplayName("zero prefixLength throws")
        void zeroPrefixLength() {
            assertThrows(IllegalArgumentException.class,
                () -> new PrefixHash("m", 0L, 0, "global"));
        }
    }

    // =========================================================================
    // CachedKVEntry record
    // =========================================================================

    @Nested
    @DisplayName("CachedKVEntry")
    class CachedKVEntryTests {

        @Test
        @DisplayName("accessed() increments hitCount and refreshes lastAccessedAt")
        void accessed() throws InterruptedException {
            PrefixHash key   = new PrefixHash("m", 1L, 16);
            CachedKVEntry e  = CachedKVEntry.of(key, List.of(0, 1), 16);
            Thread.sleep(2);
            CachedKVEntry e2 = e.accessed();
            assertEquals(1L, e2.hitCount());
            assertTrue(e2.lastAccessedAt().isAfter(e.lastAccessedAt()));
        }

        @Test
        @DisplayName("blockIds list is immutable")
        void immutableBlockIds() {
            PrefixHash key  = new PrefixHash("m", 1L, 16);
            CachedKVEntry e = CachedKVEntry.of(key, List.of(4, 7), 16);
            assertThrows(UnsupportedOperationException.class,
                () -> e.blockIds().add(99));
        }

        @Test
        @DisplayName("factory copies the blockIds list defensively")
        void defensiveCopy() {
            var mutable = new java.util.ArrayList<>(List.of(1, 2));
            PrefixHash key  = new PrefixHash("m", 1L, 16);
            CachedKVEntry e = CachedKVEntry.of(key, mutable, 16);
            mutable.add(99);
            assertEquals(2, e.blockIds().size());
        }
    }

    // =========================================================================
    // NoOpPromptCacheStore
    // =========================================================================

    @Nested
    @DisplayName("NoOpPromptCacheStore")
    class NoOpStoreTests {

        private final NoOpPromptCacheStore noOp = new NoOpPromptCacheStore();

        @Test
        void alwaysMiss() {
            PrefixHash key = new PrefixHash("m", 1L, 16);
            assertEquals(Optional.empty(), noOp.lookup(key));
        }

        @Test
        void storeDoesNothing() {
            PrefixHash key  = new PrefixHash("m", 1L, 16);
            CachedKVEntry e = CachedKVEntry.of(key, List.of(1), 16);
            assertDoesNotThrow(() -> noOp.store(e));
            assertEquals(Optional.empty(), noOp.lookup(key));
        }

        @Test
        void statsAreEmpty() {
            assertEquals("noop", noOp.stats().strategy());
            assertEquals(0, noOp.stats().hits());
        }
    }

    // =========================================================================
    // InProcessPromptCacheStore (unit-level, no Quarkus)
    // =========================================================================

    @Nested
    @DisplayName("InProcessPromptCacheStore")
    class InProcessStoreTests {

        private InProcessPromptCacheStore store;

        @BeforeEach
        void setup() {
            store = InProcessTestHelper.buildStore();
        }

        @Test
        @DisplayName("store then lookup returns same entry")
        void roundTrip() {
            PrefixHash    key   = new PrefixHash("llama3", 0x1234L, 32, "global");
            CachedKVEntry entry = CachedKVEntry.of(key, List.of(0, 1), 32);
            store.store(entry);
            Optional<CachedKVEntry> result = store.lookup(key);
            assertTrue(result.isPresent());
            assertEquals(32, result.get().tokenCount());
        }

        @Test
        @DisplayName("lookup of unknown key returns empty")
        void missOnUnknownKey() {
            PrefixHash key = new PrefixHash("llama3", 0xFFFF_FFFFL, 32, "global");
            assertTrue(store.lookup(key).isEmpty());
        }

        @Test
        @DisplayName("invalidateByModel removes correct entries")
        void invalidateByModel() {
            PrefixHash k1 = new PrefixHash("llama3", 1L, 16, "global");
            PrefixHash k2 = new PrefixHash("qwen",   2L, 16, "global");
            store.store(CachedKVEntry.of(k1, List.of(0), 16));
            store.store(CachedKVEntry.of(k2, List.of(1), 16));

            store.invalidateByModel("llama3");

            assertTrue(store.lookup(k1).isEmpty(),  "llama3 entry should be gone");
            assertTrue(store.lookup(k2).isPresent(), "qwen entry should remain");
        }

        @Test
        @DisplayName("invalidateAll removes everything")
        void invalidateAll() {
            for (int i = 0; i < 10; i++) {
                PrefixHash k = new PrefixHash("m", (long) i, 16, "global");
                store.store(CachedKVEntry.of(k, List.of(i), 16));
            }
            store.invalidateAll();
            for (int i = 0; i < 10; i++) {
                PrefixHash k = new PrefixHash("m", (long) i, 16, "global");
                assertTrue(store.lookup(k).isEmpty());
            }
        }

        @Test
        @DisplayName("lookup increments hitCount on entry")
        void hitCountIncrement() {
            PrefixHash    key   = new PrefixHash("m", 7L, 16, "global");
            CachedKVEntry entry = CachedKVEntry.of(key, List.of(3), 16);
            store.store(entry);

            store.lookup(key);
            Optional<CachedKVEntry> second = store.lookup(key);
            assertTrue(second.isPresent());
            assertTrue(second.get().hitCount() > 0);
        }

        @Test
        @DisplayName("stats reflect stores and hits")
        void stats() {
            PrefixHash k = new PrefixHash("m", 42L, 16, "global");
            store.store(CachedKVEntry.of(k, List.of(0), 16));
            store.lookup(k);
            store.lookup(k);

            var stats = store.stats();
            assertEquals("in-process", stats.strategy());
            assertTrue(stats.stores() >= 1);
            assertTrue(stats.hits()   >= 2);
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static int[] sameTokens(int n) {
        int[] t = new int[n];
        java.util.Arrays.fill(t, 42);
        return t;
    }

    private static int[] sequentialTokens(int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) t[i] = i + 1;
        return t;
    }

    private static int[] reverseTokens(int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) t[i] = n - i;
        return t;
    }
}

// ---------------------------------------------------------------------------
// Test helper factories (inner classes avoid public API pollution)
// ---------------------------------------------------------------------------

class HasherTestHelper {
    static PromptPrefixHasher hasherWithBlockSize(int blockSize, String algo) {
        // Build minimal stubs using default KVCacheConfig and inline PromptCacheConfig
        tech.kayys.gollek.kvcache.KVCacheConfig kv =
            tech.kayys.gollek.kvcache.KVCacheConfig.builder()
                .blockSize(blockSize)
                .totalBlocks(64)
                .numLayers(4)
                .numHeads(4)
                .headDim(16)
                .build();

        // Minimal PromptCacheConfig via anonymous impl
        tech.kayys.gollek.cache.PromptCacheConfig cfg =
            new tech.kayys.gollek.cache.PromptCacheConfig() {
                public boolean enabled()                      { return true; }
                public String  strategy()                     { return "in-process"; }
                public String  scope()                        { return "global"; }
                public int     minCacheableTokens()           { return 1; }
                public int     maxEntries()                   { return 1024; }
                public long    maxTotalTokens()               { return 100_000; }
                public java.time.Duration ttl()               { return java.time.Duration.ofMinutes(30); }
                public String  evictionPolicy()               { return "lru"; }
                public InProcess inProcess()                  { return new InProcess() {
                    public int     initialCapacity()          { return 64; }
                    public boolean recordStats()              { return true; }
                };}
                public Redis redis()                          { return new Redis() {
                    public String hosts()                     { return "localhost:6379"; }
                    public String keyPrefix()                 { return "gollek:pc:"; }
                    public int    maxTotalConnections()       { return 8; }
                    public java.time.Duration readTimeout()   { return java.time.Duration.ofSeconds(1); }
                    public boolean asyncWrites()              { return true; }
                    public String  serializer()               { return "json"; }
                };}
                public Disk disk()                            { return new Disk() {
                    public String  path()                     { return "/tmp/gollek-test"; }
                    public String  backend()                  { return "mmap"; }
                    public long    maxSizeMb()                { return 256; }
                    public boolean syncOnWrite()              { return false; }
                };}
                public String hashAlgo()                      { return algo; }
                public boolean warmOnStartup()                { return false; }
                public java.util.Optional<String> warmModelIds() { return java.util.Optional.empty(); }
                public boolean asyncStore()                   { return false; }
            };

        return new tech.kayys.gollek.cache.PromptPrefixHasher(kv, cfg);
    }
}

class InProcessTestHelper {
    static tech.kayys.gollek.cache.InProcessPromptCacheStore buildStore() {
        tech.kayys.gollek.cache.PromptCacheConfig cfg =
            HasherTestHelper.hasherWithBlockSize(16, "xxhash64") != null
                ? stubConfig() : stubConfig();

        // Build a no-op metrics stub
        io.micrometer.core.instrument.MeterRegistry registry =
            new io.micrometer.core.instrument.simple.SimpleMeterRegistry();

        tech.kayys.gollek.cache.PromptCacheMetrics metrics =
            new tech.kayys.gollek.cache.PromptCacheMetrics(registry, stubConfig());

        return new tech.kayys.gollek.cache.InProcessPromptCacheStore(stubConfig(), metrics);
    }

    private static tech.kayys.gollek.cache.PromptCacheConfig stubConfig() {
        return HasherTestHelper.hasherWithBlockSize(16, "xxhash64") != null
            ? buildStub() : buildStub(); // force-call to reuse helper
    }

    static tech.kayys.gollek.cache.PromptCacheConfig buildStub() {
        return new tech.kayys.gollek.cache.PromptCacheConfig() {
            public boolean enabled()                      { return true; }
            public String  strategy()                     { return "in-process"; }
            public String  scope()                        { return "global"; }
            public int     minCacheableTokens()           { return 1; }
            public int     maxEntries()                   { return 512; }
            public long    maxTotalTokens()               { return 100_000; }
            public java.time.Duration ttl()               { return java.time.Duration.ZERO; }
            public String  evictionPolicy()               { return "lru"; }
            public InProcess inProcess()                  { return new InProcess() {
                public int initialCapacity()              { return 32; }
                public boolean recordStats()              { return true; }
            };}
            public Redis redis()                          { return new Redis() {
                public String hosts()                     { return "localhost:6379"; }
                public String keyPrefix()                 { return "gollek:pc:"; }
                public int    maxTotalConnections()       { return 8; }
                public java.time.Duration readTimeout()   { return java.time.Duration.ofSeconds(1); }
                public boolean asyncWrites()              { return true; }
                public String  serializer()               { return "json"; }
            };}
            public Disk disk()                            { return new Disk() {
                public String  path()                     { return "/tmp/gollek-test"; }
                public String  backend()                  { return "mmap"; }
                public long    maxSizeMb()                { return 64; }
                public boolean syncOnWrite()              { return false; }
            };}
            public String hashAlgo()                      { return "xxhash64"; }
            public boolean warmOnStartup()                { return false; }
            public java.util.Optional<String> warmModelIds() { return java.util.Optional.empty(); }
            public boolean asyncStore()                   { return false; }
        };
    }
}
