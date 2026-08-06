package tech.kayys.gollek.cache;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.jboss.logging.Logger;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class PromptCacheSnapshotReaderTest {

    @Test
    void readsMultipleEntriesFromSnapshot() throws Exception {
        CacheEntrySerializer serializer = new CacheEntrySerializer(new ObjectMapper());
        CachedKVEntry entry1 = new CachedKVEntry(
                new PrefixHash("model-a", 123L, 16, "global"),
                List.of(1, 2, 3),
                16,
                Instant.now(),
                Instant.now(),
                0L,
                "global");
        CachedKVEntry entry2 = new CachedKVEntry(
                new PrefixHash("model-b", 456L, 32, "global"),
                List.of(4, 5),
                32,
                Instant.now(),
                Instant.now(),
                2L,
                "global");

        byte magic = 0x47;
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(magic);
        out.write(serializer.serialize(entry1).getBytes(StandardCharsets.UTF_8));
        out.write(magic);
        out.write(serializer.serialize(entry2).getBytes(StandardCharsets.UTF_8));

        Path tmp = Files.createTempFile("prompt-cache", ".dat");
        Files.write(tmp, out.toByteArray());

        List<CachedKVEntry> read = new ArrayList<>();
        PromptCacheSnapshotReader.read(
                tmp,
                serializer,
                magic,
                4096,
                Logger.getLogger(PromptCacheSnapshotReaderTest.class),
                snapshot -> read.add(snapshot.entry())
        );

        assertEquals(2, read.size());
        assertEquals("model-a", read.get(0).key().modelId());
        assertEquals("model-b", read.get(1).key().modelId());
    }
}
