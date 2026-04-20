/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorShardLoader.java
 * ──────────────────────────
 * Unified entry-point that auto-detects whether a model is single-file or
 * multi-shard, and provides a consistent tensor-access API for both cases.
 *
 * Decision logic
 * ══════════════
 *   Path is a file                 → load as single SafeTensors file
 *   Path is a directory + index    → load as multi-shard model
 *   Path is a directory, no index  → scan for a single .safetensors file
 *
 * Multi-shard tensor access
 * ═════════════════════════
 * Shards are opened lazily — only the shard files that contain actually
 * requested tensors are memory-mapped.  This is crucial for large models
 * where you may only need a subset of weights (e.g. for LoRA adapter merging).
 *
 * Each open shard is kept in a cache and reused for subsequent lookups
 * within the same session.  The cache is evicted on {@link #close()}.
 */
package tech.kayys.gollek.safetensor.loader;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.exception.SafetensorException;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Auto-detecting, lazy-loading SafeTensors loader for both single-file and
 * multi-shard model checkpoints.
 *
 * <p>
 * This is the recommended CDI entry point for production use — it handles
 * all file/directory layouts used by HuggingFace Hub-style model repositories.
 *
 * <p>
 * <b>Usage:</b>
 * 
 * <pre>{@code
 * @Inject
 * SafetensorShardLoader loader;
 *
 * try (SafetensorShardSession session = loader.open(Path.of("/models/llama2-70b"))) {
 *     SafetensorTensor q = session.tensor("model.layers.0.self_attn.q_proj.weight");
 *     float[] weights = q.toFloatArray();
 * }
 * }</pre>
 */
@ApplicationScoped
public class SafetensorShardLoader {

    private static final Logger log = Logger.getLogger(SafetensorShardLoader.class);

    @Inject
    SafetensorFFMLoader ffmLoader;

    @Inject
    ObjectMapper objectMapper;

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Open a SafeTensors model checkpoint at the given path.
     *
     * <p>
     * The path may be:
     * <ul>
     * <li>A {@code .safetensors} file — single-file model
     * <li>A directory containing a {@code model.safetensors.index.json}
     * — multi-shard model (shards loaded lazily)
     * <li>A directory without an index — single-file model found by scanning
     * </ul>
     *
     * @param modelPath path to the model file or directory
     * @return an open {@link SafetensorShardSession}
     * @throws SafetensorException on detection or loading errors
     */
    public SafetensorShardSession open(Path modelPath) {
        Objects.requireNonNull(modelPath, "modelPath must not be null");
        Path resolved = modelPath.toAbsolutePath().normalize();

        if (Files.isRegularFile(resolved)) {
            // Direct single-file load
            log.infof("SafetensorShardLoader: opening single-file model [%s]", resolved);
            SafetensorLoadResult result = ffmLoader.load(resolved);
            return SafetensorShardSession.singleFile(result);
        }

        if (Files.isDirectory(resolved)) {
            if (SafetensorShardIndex.isShardedModel(resolved)) {
                // Multi-shard model
                log.infof("SafetensorShardLoader: opening multi-shard model [%s]", resolved);
                Path indexPath = SafetensorShardIndex.resolveIndexPath(resolved);
                SafetensorShardIndex index = SafetensorShardIndex.load(indexPath, objectMapper);
                return SafetensorShardSession.sharded(index, ffmLoader);
            } else {
                // Scan for a single safetensors file in the directory
                Path candidate = findSingleSafetensorFile(resolved);
                log.infof("SafetensorShardLoader: found single file in dir [%s]", candidate);
                SafetensorLoadResult result = ffmLoader.load(candidate);
                return SafetensorShardSession.singleFile(result);
            }
        }

        throw new SafetensorException.IoException(
                "Path does not exist or is not a file/directory: " + resolved, resolved, null);
    }

    /**
     * Inspect the header(s) of a model without loading tensor data.
     * Returns a merged view of all tensor metadata across all shards.
     *
     * @param modelPath path to the model file or directory
     * @return list of all tensor info objects (across all shards)
     */
    public List<SafetensorTensorInfo> inspectTensors(Path modelPath) {
        Objects.requireNonNull(modelPath);
        Path resolved = modelPath.toAbsolutePath().normalize();

        if (Files.isRegularFile(resolved)) {
            SafetensorHeader hdr = ffmLoader.loadHeaderOnly(resolved);
            return new ArrayList<>(hdr.tensors().values());
        }

        if (Files.isDirectory(resolved)) {
            if (SafetensorShardIndex.isShardedModel(resolved)) {
                Path indexPath = SafetensorShardIndex.resolveIndexPath(resolved);
                SafetensorShardIndex index = SafetensorShardIndex.load(indexPath, objectMapper);

                List<SafetensorTensorInfo> all = new ArrayList<>();
                for (String shardName : index.shardFileNames()) {
                    Path shardPath = index.shardPath(shardName);
                    SafetensorHeader hdr = ffmLoader.loadHeaderOnly(shardPath);
                    all.addAll(hdr.tensors().values());
                }
                return Collections.unmodifiableList(all);
            } else {
                Path candidate = findSingleSafetensorFile(resolved);
                SafetensorHeader hdr = ffmLoader.loadHeaderOnly(candidate);
                return new ArrayList<>(hdr.tensors().values());
            }
        }

        throw new SafetensorException.IoException(
                "Path not found: " + resolved, resolved, null);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    private Path findSingleSafetensorFile(Path dir) {
        try (var stream = Files.walk(dir, 2)) {
            List<Path> candidates = stream
                    .filter(Files::isRegularFile)
                    .filter(p -> {
                        String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
                        return n.endsWith(".safetensors") || n.endsWith(".safetensor");
                    })
                    .collect(Collectors.toList());

            if (candidates.isEmpty()) {
                throw new SafetensorException.IoException(
                        "No .safetensors file found in directory: " + dir, dir, null);
            }
            if (candidates.size() > 1) {
                log.warnf("Multiple .safetensors files found in [%s], using first: %s",
                        dir, candidates.get(0));
            }
            return candidates.get(0);
        } catch (SafetensorException e) {
            throw e;
        } catch (IOException e) {
            throw new SafetensorException.IoException(
                    "Error scanning directory for .safetensors files", dir, e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inner class: Session
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * An open model session that provides tensor access.
     *
     * <p>
     * For multi-shard models, shard files are opened (mmap'd) lazily on
     * first tensor access and cached for the session lifetime.
     */
    public static final class SafetensorShardSession implements AutoCloseable {

        /** {@code true} for single-file models. */
        private final boolean singleFile;

        /** Single-file result (non-null only for single-file models). */
        private final SafetensorLoadResult singleResult;

        /** Multi-shard index (non-null only for sharded models). */
        private final SafetensorShardIndex shardIndex;

        /** Lazily-opened shard results, keyed by shard filename. */
        private final Map<String, SafetensorLoadResult> openShards;

        /** FFM loader used to open additional shards on demand. */
        private final SafetensorFFMLoader ffmLoader;

        private volatile boolean closed = false;

        // Single-file constructor
        private SafetensorShardSession(SafetensorLoadResult single) {
            this.singleFile = true;
            this.singleResult = single;
            this.shardIndex = null;
            this.openShards = null;
            this.ffmLoader = null;
        }

        // Sharded constructor
        private SafetensorShardSession(SafetensorShardIndex index, SafetensorFFMLoader loader) {
            this.singleFile = false;
            this.singleResult = null;
            this.shardIndex = index;
            this.openShards = new ConcurrentHashMap<>();
            this.ffmLoader = loader;
        }

        static SafetensorShardSession singleFile(SafetensorLoadResult result) {
            return new SafetensorShardSession(result);
        }

        static SafetensorShardSession sharded(SafetensorShardIndex index, SafetensorFFMLoader loader) {
            return new SafetensorShardSession(index, loader);
        }

        // ── AccelTensor access ─────────────────────────────────────────────────────

        /**
         * Get a tensor by name.
         *
         * @param name tensor key (e.g. {@code "model.layers.31.mlp.gate_proj.weight"})
         * @return the tensor view
         * @throws SafetensorException.TensorNotFoundException if not found
         * @throws IllegalStateException                       if the session is closed
         */
        public SafetensorTensor tensor(String name) {
            checkOpen();
            if (singleFile) {
                return singleResult.tensor(name);
            }
            // Lazy shard open
            Path shardPath = shardIndex.shardPathForTensor(name);
            String shardName = shardPath.getFileName().toString();
            SafetensorLoadResult shard = openShards.computeIfAbsent(
                    shardName, k -> ffmLoader.load(shardPath));
            return shard.tensor(name);
        }

        /** Find a tensor by name, returning empty if absent. */
        public Optional<SafetensorTensor> findTensor(String name) {
            checkOpen();
            if (singleFile) {
                return singleResult.findTensor(name);
            }
            if (!shardIndex.hasTensor(name))
                return Optional.empty();
            return Optional.of(tensor(name));
        }

        /** Whether this session is a sharded model. */
        public boolean isSharded() {
            return !singleFile;
        }

        /** Number of tensors (across all shards). */
        public int tensorCount() {
            return singleFile ? singleResult.tensorCount() : shardIndex.tensorCount();
        }

        /** All tensor names in this session. */
        public Set<String> tensorNames() {
            checkOpen();
            return singleFile
                    ? singleResult.tensorNames()
                    : shardIndex.weightMap().keySet();
        }

        // ── Lifecycle ─────────────────────────────────────────────────────────

        @Override
        public void close() {
            if (closed)
                return;
            closed = true;

            if (singleFile) {
                safeClose(singleResult);
            } else {
                openShards.values().forEach(SafetensorShardSession::safeClose);
                openShards.clear();
            }
        }

        private void checkOpen() {
            if (closed)
                throw new IllegalStateException("SafetensorShardSession is closed.");
        }

        private static void safeClose(AutoCloseable c) {
            try {
                if (c != null)
                    c.close();
            } catch (Exception e) {
                /* best effort */ }
        }

        @Override
        public String toString() {
            return "SafetensorShardSession{"
                    + (singleFile ? "file=" + singleResult.filePath()
                            : "shards=" + shardIndex.shardCount()
                                    + ", tensors=" + shardIndex.tensorCount())
                    + ", closed=" + closed
                    + '}';
        }
    }
}
