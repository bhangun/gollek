/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorLoadResult.java
 * ─────────────────────────
 * AutoCloseable container produced by {@link SafetensorFFMLoader#load}.
 *
 * Lifetime model
 * ══════════════
 * This object is the root owner of the FFM {@link java.lang.foreign.Arena}
 * that backs all tensor memory segments.  Closing this object:
 *   1. Marks all constituent {@link tech.kayys.gollek.inference.safetensor.model.SafetensorTensor}
 *      instances as closed (fast-fail on subsequent access).
 *   2. Closes the Arena, which either:
 *       - unmaps the memory-mapped file segment (MMAP mode), or
 *       - releases the off-heap native allocation (COPY mode).
 *
 * Recommended use pattern:
 * ────────────────────────
 *   try (SafetensorLoadResult result = loader.load(path)) {
 *       SafetensorTensor weights = result.tensor("model.embed_tokens.weight");
 *       // operate on weights...
 *   }  // Arena closed here — tensor segments are unmapped
 *
 * Thread-safety
 * ═════════════
 * Read access to tensors is safe from multiple threads when the underlying
 * Arena is a shared arena (MMAP and COPY modes both use shared arenas).
 * The {@link #close()} method is idempotent and may be called from any thread.
 */
package tech.kayys.gollek.safetensor.loader;

import org.jboss.logging.Logger;

import tech.kayys.gollek.safetensor.exception.SafetensorException;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Holds a fully-loaded SafeTensors file: parsed header + off-heap tensor data.
 *
 * <p>
 * Every {@link SafetensorTensor} vended by this object shares backing memory
 * with the single {@link Arena} managed here. Closing this result invalidates
 * all tensors simultaneously.
 */
public final class SafetensorLoadResult implements AutoCloseable {

    private static final Logger log = Logger.getLogger(SafetensorLoadResult.class);

    // ── Fields ────────────────────────────────────────────────────────────────

    /** The source file (or primary shard) path. */
    private final Path filePath;

    /** Parsed header metadata. */
    private final SafetensorHeader header;

    /**
     * The entire file's memory-mapped segment (or native copy).
     * This is NOT closed here — the arena owns it.
     */
    private final MemorySegment fileSegment;

    /**
     * The Arena whose lifetime controls {@code fileSegment}.
     * Closing the arena unmaps / frees the memory.
     */
    private final Arena arena;

    /** Cached tensor view objects, created lazily on first access. */
    private final Map<String, SafetensorTensor> tensorCache;

    /** Guard against double-close. */
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /** Load mode — for diagnostics. */
    private final LoadMode mode;

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * How the tensor data was loaded into memory.
     */
    public enum LoadMode {
        /**
         * Memory-mapped: the OS maps the file into the process address space.
         * Zero-copy, lazy — only pages actually read cause disk I/O.
         */
        MMAP,
        /**
         * Copied into a native memory allocation.
         * Used when mmap is unavailable (e.g., network filesystems, some OSes).
         */
        COPY
    }

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Public constructor — used by {@link SafetensorFFMLoader} and standalone loaders.
     */
    public SafetensorLoadResult(
            Path filePath,
            SafetensorHeader header,
            MemorySegment fileSegment,
            Arena arena,
            LoadMode mode) {
        this.filePath = Objects.requireNonNull(filePath, "filePath");
        this.header = Objects.requireNonNull(header, "header");
        this.fileSegment = Objects.requireNonNull(fileSegment, "fileSegment");
        this.arena = Objects.requireNonNull(arena, "arena");
        this.mode = Objects.requireNonNull(mode, "mode");
        this.tensorCache = new HashMap<>(header.tensorCount() * 2);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /** The source file path. */
    public Path filePath() {
        return filePath;
    }

    /** The parsed SafeTensors header. */
    public SafetensorHeader header() {
        checkOpen();
        return header;
    }

    /** The load mode (MMAP or COPY). */
    public LoadMode mode() {
        return mode;
    }

    /** {@code true} if this result has been closed. */
    public boolean isClosed() {
        return closed.get();
    }

    // ── TorchTensor access ─────────────────────────────────────────────────────────

    /**
     * Retrieve a zero-copy tensor view by name.
     *
     * <p>
     * The returned {@link SafetensorTensor} shares backing memory with this
     * result's Arena. Do not use the tensor after this result is closed.
     *
     * <p>
     * TorchTensor views are cached — repeated calls with the same name return the
     * same object.
     *
     * @param name tensor key (e.g. {@code "model.embed_tokens.weight"})
     * @return the tensor view
     * @throws SafetensorException.TensorNotFoundException if name is absent
     * @throws IllegalStateException                       if this result has been
     *                                                     closed
     */
    public SafetensorTensor tensor(String name) {
        checkOpen();
        return tensorCache.computeIfAbsent(name, this::buildTensor);
    }

    /**
     * Retrieve a tensor if it exists, returning empty otherwise.
     *
     * @param name tensor key
     * @return an {@link Optional} containing the tensor, or empty
     */
    public Optional<SafetensorTensor> findTensor(String name) {
        checkOpen();
        if (!header.hasTensor(name))
            return Optional.empty();
        return Optional.of(tensor(name));
    }

    /**
     * Return an unmodifiable view of ALL tensors in this file.
     * TorchTensor views are created lazily during iteration.
     *
     * @return ordered map of tensor name → tensor
     */
    public Map<String, SafetensorTensor> allTensors() {
        checkOpen();
        // Ensure all tensors are in the cache
        for (String name : header.tensors().keySet()) {
            tensorCache.computeIfAbsent(name, this::buildTensor);
        }
        return Collections.unmodifiableMap(tensorCache);
    }

    /** Unmodifiable set of all tensor names in this file. */
    public Set<String> tensorNames() {
        checkOpen();
        return header.tensors().keySet();
    }

    /** Number of tensors. */
    public int tensorCount() {
        return header.tensorCount();
    }

    /** Total byte size of all tensor data in this file. */
    public long totalDataBytes() {
        return header.totalDataBytes();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Close this load result:
     * <ol>
     * <li>Mark all cached tensor views as closed (fast-fail guard).
     * <li>Close the Arena (unmaps mmap or frees native memory).
     * </ol>
     *
     * <p>
     * Idempotent — safe to call multiple times.
     */
    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return; // already closed
        }
        // Invalidate all live tensor views
        tensorCache.values().forEach(SafetensorTensor::close);
        tensorCache.clear();

        // Release the off-heap memory / unmap the file
        try {
            arena.close();
            log.debugf("SafetensorLoadResult closed [%s, mode=%s]", filePath, mode);
        } catch (Exception e) {
            log.warnf(e, "Failed to close Arena for SafeTensors file [%s]", filePath);
        }
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    private SafetensorTensor buildTensor(String name) {
        SafetensorTensorInfo info = header.tensors().get(name);
        if (info == null) {
            throw new SafetensorException.TensorNotFoundException(name, filePath);
        }

        // Compute absolute byte range within the file segment
        long absBegin = header.dataBlobOffset() + info.dataBegin();
        long absEnd = header.dataBlobOffset() + info.dataEnd();
        long len = absEnd - absBegin;

        // Slice the file segment to exactly this tensor's bytes (zero-copy)
        MemorySegment tensorSegment = fileSegment.asSlice(absBegin, len);

        return new SafetensorTensor(info, tensorSegment);
    }

    private void checkOpen() {
        if (closed.get()) {
            throw new IllegalStateException(
                    "SafetensorLoadResult for [" + filePath + "] has been closed.");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public String toString() {
        return "SafetensorLoadResult{"
                + "path=" + filePath
                + ", tensors=" + header.tensorCount()
                + ", mode=" + mode
                + ", closed=" + closed.get()
                + '}';
    }
}
