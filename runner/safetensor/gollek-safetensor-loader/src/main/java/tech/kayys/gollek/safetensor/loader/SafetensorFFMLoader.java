/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorFFMLoader.java
 * ────────────────────────
 * Core implementation of the SafeTensors loader using the JDK 25 Foreign
 * Function & Memory (FFM) API (JEP 454 — finalized in JDK 22, extended in 25).
 *
 * Loading strategy
 * ════════════════
 *                          ┌──────────────────────────────────────────┐
 *           .safetensors   │  FileChannel.map(…, Arena.ofShared())    │
 *               file  ────▶│  → MemorySegment (mmap, off-heap, lazy) │
 *                          └──────────────────────────────────────────┘
 *                                           │
 *                          ┌────────────────▼──────────────────────────┐
 *                          │  SafetensorHeaderParser.parse(segment)    │
 *                          │  reads 8-byte length + JSON (zero-copy)   │
 *                          └──────────────────────────────────────────-┘
 *                                           │
 *                          ┌────────────────▼──────────────────────────┐
 *                          │  SafetensorLoadResult                     │
 *                          │  ├── header (metadata map)                │
 *                          │  └── tensors (MemorySegment slices)       │
 *                          └───────────────────────────────────────────┘
 *
 * Key FFM APIs used
 * ═════════════════
 *  • Arena.ofShared()          — shared, thread-safe arena for mmap lifetime
 *  • FileChannel.map(…, arena) — produces MemorySegment directly from JDK 22+
 *  • MemorySegment.asSlice()   — zero-copy view of a tensor's byte range
 *  • ValueLayout.*             — typed element access with explicit byte order
 *
 * Fallback (COPY mode)
 * ════════════════════
 * On filesystems that do not support mmap (e.g. NFS, some container volume
 * drivers, Windows network shares), the loader falls back to reading the file
 * into native memory via Arena.allocate() + MemorySegment.copyFrom().
 *
 * Metrics
 * ═══════
 * All load operations are timed and reported via
 * {@link tech.kayys.gollek.inference.safetensor.metrics.SafetensorMetrics}.
 */
package tech.kayys.gollek.safetensor.loader;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.exception.SafetensorException;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.Instant;
import java.util.Objects;

/**
 * FFM-backed SafeTensors loader.
 *
 * <p>
 * Memory-maps the entire file into off-heap memory via a shared
 * {@link Arena}. Header parsing is zero-copy; tensor data is accessed as
 * typed slices of the same mapped region.
 *
 * <p>
 * This bean is the primary CDI entry point for loading SafeTensors files.
 * Inject it where needed:
 *
 * <pre>{@code
 * @Inject
 * SafetensorFFMLoader loader;
 *
 * try (SafetensorLoadResult result = loader.load(path)) {
 *     float[] weights = result.tensor("lm_head.weight").toFloatArray();
 * }
 * }</pre>
 */
@ApplicationScoped
public class SafetensorFFMLoader {

    private static final Logger log = Logger.getLogger(SafetensorFFMLoader.class);

    @Inject
    SafetensorLoaderConfig config;

    @Inject
    ObjectMapper objectMapper;

    @Inject
    SafetensorMetrics metrics;

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Load a SafeTensors file into off-heap memory.
     *
     * <p>
     * The returned {@link SafetensorLoadResult} is an {@link AutoCloseable};
     * callers MUST close it (try-with-resources is strongly recommended) to
     * release the mmap'd region or native allocation.
     *
     * @param filePath path to a {@code .safetensors} or {@code .safetensor} file
     * @return a loaded result ready for tensor access
     * @throws SafetensorException on any loading or parsing failure
     */
    public SafetensorLoadResult load(Path filePath) {
        Objects.requireNonNull(filePath, "filePath must not be null");
        Path resolved = filePath.toAbsolutePath().normalize();
        validatePath(resolved);

        Instant start = Instant.now();
        SafetensorLoadResult result;

        if (config.preferMmap()) {
            result = loadMmap(resolved);
        } else {
            result = loadCopy(resolved);
        }

        Duration elapsed = Duration.between(start, Instant.now());
        metrics.recordLoad(resolved, result.mode(), result.totalDataBytes(), elapsed);

        log.infof("SafeTensors loaded: %s [%d tensors, %d bytes, mode=%s, elapsed=%dms]",
                resolved.getFileName(),
                result.tensorCount(),
                result.totalDataBytes(),
                result.mode(),
                elapsed.toMillis());

        return result;
    }

    /**
     * Load only the header of a SafeTensors file — no tensor data is read or
     * mapped. Useful for model introspection (shape inspection, dtype checks)
     * without the cost of mapping the entire file.
     *
     * @param filePath path to the SafeTensors file
     * @return the parsed header (no associated Arena / tensor data)
     */
    public SafetensorHeader loadHeaderOnly(Path filePath) {
        Objects.requireNonNull(filePath, "filePath must not be null");
        Path resolved = filePath.toAbsolutePath().normalize();
        validatePath(resolved);

        // Map the full file once — the OS only pages in the header region
        // lazily when the parser reads those pages. No need to map twice.
        try (Arena arena = Arena.ofConfined();
                FileChannel channel = FileChannel.open(resolved, StandardOpenOption.READ)) {

            long fileSize = channel.size();
            if (fileSize == 0) {
                throw new SafetensorException.ValidationException("File is empty", resolved);
            }

            MemorySegment seg = channel.map(FileChannel.MapMode.READ_ONLY, 0L, fileSize, arena);

            SafetensorHeaderParser parser = SafetensorHeaderParser.create(objectMapper);
            return parser.parse(seg, resolved);

        } catch (SafetensorException e) {
            throw e;
        } catch (IOException e) {
            throw new SafetensorException.IoException(
                    "Failed to read SafeTensors header", resolved, e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MMAP load
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Memory-map the entire file into a shared Arena.
     *
     * <p>
     * The OS maps file pages lazily — only accessed pages are paged in from
     * disk. For inference workloads that read the entire model, the OS will
     * eventually page in all data, but the initial map call is O(1).
     */
    private SafetensorLoadResult loadMmap(Path resolved) {
        Arena arena = Arena.ofShared(); // arena outlives this method
        try {
            FileChannel channel = FileChannel.open(resolved, StandardOpenOption.READ);
            long fileSize = channel.size();

            if (fileSize == 0) {
                arena.close();
                throw new SafetensorException.ValidationException(
                        "File is empty", resolved);
            }

            // JDK 22+ API: FileChannel.map(mode, offset, size, arena) → MemorySegment
            MemorySegment fileSegment = channel.map(
                    FileChannel.MapMode.READ_ONLY, 0L, fileSize, arena);

            // Channel can be closed after mmap — the segment remains valid
            channel.close();

            SafetensorHeaderParser parser = SafetensorHeaderParser.create(objectMapper);
            SafetensorHeader header = parser.parse(fileSegment, resolved);

            return new SafetensorLoadResult(
                    resolved, header, fileSegment, arena, SafetensorLoadResult.LoadMode.MMAP);

        } catch (SafetensorException e) {
            safeClose(arena);
            throw e;
        } catch (UnsupportedOperationException e) {
            // mmap not supported on this filesystem — fall back to copy
            safeClose(arena);
            log.infof("mmap not supported for [%s], falling back to COPY mode: %s",
                    resolved, e.getMessage());
            return loadCopy(resolved);
        } catch (IOException e) {
            safeClose(arena);
            throw new SafetensorException.IoException(resolved, e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // COPY load (fallback)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Read the file into a native off-heap allocation.
     *
     * <p>
     * Less efficient than mmap for large models (requires a full read on
     * startup) but works on all filesystems.
     */
    private SafetensorLoadResult loadCopy(Path resolved) {
        Arena arena = Arena.ofShared();
        try (FileChannel channel = FileChannel.open(resolved, StandardOpenOption.READ)) {
            long fileSize = channel.size();

            if (fileSize == 0) {
                arena.close();
                throw new SafetensorException.ValidationException("File is empty", resolved);
            }

            // Allocate off-heap native memory
            MemorySegment nativeBuffer = arena.allocate(fileSize);

            // Copy file content into the native buffer
            // Use a heap-side staging ByteBuffer for the read
            long remaining = fileSize;
            long writeOffset = 0L;
            int chunkSize = config.readChunkBytes();

            java.nio.ByteBuffer chunk = java.nio.ByteBuffer.allocate(chunkSize);

            while (remaining > 0) {
                chunk.clear();
                int toRead = (int) Math.min(remaining, chunkSize);
                chunk.limit(toRead);

                int bytesRead = channel.read(chunk);
                if (bytesRead < 0) {
                    throw new SafetensorException.IoException(
                            "Unexpected EOF at offset " + writeOffset + " (expected " + fileSize + " bytes)",
                            resolved, null);
                }

                chunk.flip();
                // Copy chunk bytes into off-heap via MemorySegment
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

        } catch (SafetensorException e) {
            safeClose(arena);
            throw e;
        } catch (IOException e) {
            safeClose(arena);
            throw new SafetensorException.IoException(resolved, e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private void validatePath(Path resolved) {
        if (!Files.exists(resolved)) {
            throw new SafetensorException.IoException(
                    "File not found: " + resolved, resolved, null);
        }
        if (!Files.isRegularFile(resolved)) {
            throw new SafetensorException.ValidationException(
                    "Path is not a regular file: " + resolved, resolved);
        }
        String name = resolved.getFileName().toString().toLowerCase();
        if (!name.endsWith(".safetensors") && !name.endsWith(".safetensor")) {
            log.warnf("File [%s] does not have a .safetensors extension — proceeding anyway",
                    resolved);
        }
    }

    /**
     * Read only the 8-byte header-length prefix, without mapping the full file.
     * Used by {@link #loadHeaderOnly(Path)}.
     */
    private long peekHeaderLength(Path resolved) {
        try (FileChannel channel = FileChannel.open(resolved, StandardOpenOption.READ)) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment prefix = channel.map(
                        FileChannel.MapMode.READ_ONLY, 0L, 8L, arena);
                long length = prefix.get(
                        java.lang.foreign.ValueLayout.JAVA_LONG
                                .withOrder(java.nio.ByteOrder.LITTLE_ENDIAN),
                        0L);
                if (length < 0 || length > SafetensorHeaderParser.MAX_HEADER_BYTES) {
                    throw new SafetensorException.ValidationException(
                            "Invalid header length: " + length, resolved);
                }
                return length;
            }
        } catch (SafetensorException e) {
            throw e;
        } catch (IOException e) {
            throw new SafetensorException.IoException(resolved, e);
        }
    }

    private static void safeClose(Arena arena) {
        try {
            arena.close();
        } catch (Exception ignored) {
        }
    }
}
