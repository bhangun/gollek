package tech.kayys.gollek.cache;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.function.Consumer;

/**
 * Reader for prompt-cache disk snapshots (prompt-cache.dat).
 *
 * <p>Each entry is encoded as: [MAGIC(1)] [JSON bytes]. JSON is a single object
 * and may not include delimiters, so we scan for balanced braces to locate the end.
 */
public final class PromptCacheSnapshotReader {

    private PromptCacheSnapshotReader() {}

    public record SnapshotEntry(long offset, int length, CachedKVEntry entry) {}

    public static void read(Path dataFile,
                            CacheEntrySerializer serializer,
                            byte magic,
                            int maxEntryBytes,
                            Logger log,
                            Consumer<SnapshotEntry> consumer) {
        if (dataFile == null || !Files.isRegularFile(dataFile)) {
            return;
        }

        try (FileChannel channel = FileChannel.open(dataFile, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize <= 0) return;

            MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
            long offset = 0;

            while (offset < fileSize && buffer.remaining() > 1) {
                byte b = buffer.get();
                if (b != magic) break;

                int jsonLen = scanJsonLength(buffer, maxEntryBytes - 1);
                if (jsonLen <= 0) break;

                byte[] buf = new byte[jsonLen];
                buffer.get(buf);

                try {
                    String json = new String(buf, StandardCharsets.UTF_8).trim();
                    CachedKVEntry entry = serializer.deserialize(json);
                    consumer.accept(new SnapshotEntry(offset, 1 + jsonLen, entry));
                } catch (Exception e) {
                    log.debugf("[PromptCacheSnapshot] parse failed at offset=%d: %s",
                            offset, e.getMessage());
                    break;
                }

                offset += 1L + jsonLen;
            }
        } catch (IOException e) {
            log.warnf("[PromptCacheSnapshot] Failed to read %s: %s", dataFile, e.getMessage());
        }
    }

    private static int scanJsonLength(ByteBuffer view, int maxBytes) {
        int start = view.position();
        ByteBuffer scan = view.duplicate();
        scan.position(start);

        boolean started = false;
        boolean inString = false;
        boolean escape = false;
        int depth = 0;
        int len = 0;

        while (scan.hasRemaining() && len < maxBytes) {
            char c = (char) (scan.get() & 0xff);
            len++;

            if (!started) {
                if (Character.isWhitespace(c)) continue;
                if (c != '{') return -1;
                started = true;
                depth = 1;
                continue;
            }

            if (inString) {
                if (escape) {
                    escape = false;
                } else if (c == '\\') {
                    escape = true;
                } else if (c == '"') {
                    inString = false;
                }
                continue;
            }

            if (c == '"') {
                inString = true;
            } else if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) {
                    break;
                }
            }
        }

        if (!started || depth != 0) return -1;
        return len;
    }
}
