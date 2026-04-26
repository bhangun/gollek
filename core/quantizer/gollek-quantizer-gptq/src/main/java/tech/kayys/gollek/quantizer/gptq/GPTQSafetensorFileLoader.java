package tech.kayys.gollek.quantizer.gptq;

import com.fasterxml.jackson.databind.ObjectMapper;

import tech.kayys.gollek.safetensor.loader.SafetensorHeader;
import tech.kayys.gollek.safetensor.loader.SafetensorHeaderParser;
import tech.kayys.gollek.safetensor.loader.SafetensorLoadResult;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Standalone (non-CDI) SafeTensors file loader for the GPTQ quantizer module.
 *
 * <p>This class wraps the core safetensor-loader infrastructure
 * ({@link SafetensorHeaderParser}, FFM API) without requiring CDI injection.
 * It is designed for use in standalone contexts where the Quarkus container
 * is not available.
 *
 * <p>Usage:
 * <pre>{@code
 * try (GPTQSafetensorFileLoader loader = new GPTQSafetensorFileLoader()) {
 *     GPTQSafetensorShard shard = loader.loadShard(path);
 *     MemorySegment segment = shard.getTensorSegment("model.layers.0.qweight");
 *     // ...
 * }
 * }</pre>
 */
public class GPTQSafetensorFileLoader implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(GPTQSafetensorFileLoader.class);

    private static final int DEFAULT_READ_CHUNK = 8 * 1024 * 1024; // 8 MiB
    private static final boolean DEFAULT_PREFER_MMAP = true;

    private final ObjectMapper objectMapper;
    private final boolean preferMmap;
    private final int readChunkBytes;

    public GPTQSafetensorFileLoader() {
        this(DEFAULT_PREFER_MMAP, DEFAULT_READ_CHUNK);
    }

    public GPTQSafetensorFileLoader(boolean preferMmap, int readChunkBytes) {
        this.objectMapper = new ObjectMapper();
        this.preferMmap = preferMmap;
        this.readChunkBytes = readChunkBytes;
    }

    /**
     * Loads a single safetensor file and wraps it as a GPTQ shard.
     *
     * @param filePath path to the .safetensors file
     * @return a GPTQSafetensorShard wrapping the loaded result
     * @throws IOException on file access or parsing errors
     */
    public GPTQSafetensorShard loadShard(Path filePath) throws IOException {
        Path resolved = filePath.toAbsolutePath().normalize();
        validatePath(resolved);

        SafetensorLoadResult result;
        if (preferMmap) {
            result = loadMmap(resolved);
        } else {
            result = loadCopy(resolved);
        }

        log.info("Loaded safetensor file: {} [{} tensors, mode={}]",
                resolved.getFileName(), result.tensorCount(), result.mode());

        return new GPTQSafetensorShard(resolved, result);
    }

    /**
     * Loads only the header of a safetensor file without mapping tensor data.
     * Useful for model introspection (config auto-detection).
     *
     * @param filePath path to the .safetensors file
     * @return the parsed header
     * @throws IOException on file access or parsing errors
     */
    public SafetensorHeader loadHeaderOnly(Path filePath) throws IOException {
        Path resolved = filePath.toAbsolutePath().normalize();
        validatePath(resolved);

        try (Arena arena = Arena.ofConfined();
             FileChannel channel = FileChannel.open(resolved, StandardOpenOption.READ)) {

            long fileSize = channel.size();
            if (fileSize == 0) {
                throw new IOException("File is empty: " + resolved);
            }

            MemorySegment seg = channel.map(FileChannel.MapMode.READ_ONLY, 0L, fileSize, arena);
            SafetensorHeaderParser parser = SafetensorHeaderParser.create(objectMapper);
            return parser.parse(seg, resolved);
        }
    }

    // ── MMAP load ─────────────────────────────────────────────────────────────

    private SafetensorLoadResult loadMmap(Path resolved) throws IOException {
        Arena arena = Arena.ofAuto();
        try (FileChannel channel = FileChannel.open(resolved, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize == 0) {
                arena.close();
                throw new IOException("File is empty: " + resolved);
            }

            MemorySegment fileSegment = channel.map(
                    FileChannel.MapMode.READ_ONLY, 0L, fileSize, arena);

            // Channel can be closed after mmap — the segment remains valid
            channel.close();

            SafetensorHeaderParser parser = SafetensorHeaderParser.create(objectMapper);
            SafetensorHeader header = parser.parse(fileSegment, resolved);

            return new SafetensorLoadResult(
                    resolved, header, fileSegment, arena, SafetensorLoadResult.LoadMode.MMAP);

        } catch (UnsupportedOperationException e) {
            // mmap not supported — fall back to copy
            arena.close();
            log.info("mmap not supported for [{}], falling back to COPY mode: {}",
                    resolved, e.getMessage());
            return loadCopy(resolved);
        } catch (IOException e) {
            arena.close();
            throw e;
        }
    }

    // ── COPY load (fallback) ──────────────────────────────────────────────────

    private SafetensorLoadResult loadCopy(Path resolved) throws IOException {
        Arena arena = Arena.ofAuto();
        try (FileChannel channel = FileChannel.open(resolved, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize == 0) {
                arena.close();
                throw new IOException("File is empty: " + resolved);
            }

            MemorySegment nativeBuffer = arena.allocate(fileSize);

            long remaining = fileSize;
            long writeOffset = 0L;
            java.nio.ByteBuffer chunk = java.nio.ByteBuffer.allocate(readChunkBytes);

            while (remaining > 0) {
                chunk.clear();
                int toRead = (int) Math.min(remaining, readChunkBytes);
                chunk.limit(toRead);

                int bytesRead = channel.read(chunk);
                if (bytesRead < 0) {
                    throw new IOException("Unexpected EOF at offset " + writeOffset
                            + " (expected " + fileSize + " bytes)");
                }

                chunk.flip();
                MemorySegment src = MemorySegment.ofBuffer(chunk);
                MemorySegment dest = nativeBuffer.asSlice(writeOffset, bytesRead);
                dest.copyFrom(src.asSlice(0, bytesRead));

                writeOffset += bytesRead;
                remaining -= bytesRead;
            }

            SafetensorHeaderParser parser = SafetensorHeaderParser.create(objectMapper);
            SafetensorHeader header = parser.parse(nativeBuffer, resolved);

            return new SafetensorLoadResult(
                    resolved, header, nativeBuffer, arena, SafetensorLoadResult.LoadMode.COPY);

        } catch (IOException e) {
            arena.close();
            throw e;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void validatePath(Path resolved) throws IOException {
        if (!Files.exists(resolved)) {
            throw new IOException("File not found: " + resolved);
        }
        if (!Files.isRegularFile(resolved)) {
            throw new IOException("Path is not a regular file: " + resolved);
        }
        String name = resolved.getFileName().toString().toLowerCase();
        if (!name.endsWith(".safetensors") && !name.endsWith(".safetensor")) {
            log.warn("File [{}] does not have a .safetensors extension — proceeding anyway", resolved);
        }
    }

    @Override
    public void close() {
        // GPTQSafetensorFileLoader itself doesn't hold long-lived resources.
        // Each GPTQSafetensorShard owns its own Arena and must be closed independently.
        log.debug("GPTQSafetensorFileLoader closed");
    }
}
