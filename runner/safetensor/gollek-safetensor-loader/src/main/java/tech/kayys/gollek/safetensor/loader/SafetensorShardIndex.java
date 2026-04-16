/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorShardIndex.java
 * ─────────────────────────
 * Representation and loader for the multi-shard index format used by large
 * models (LLaMA-2-70B, Mixtral, Falcon-180B, etc.) that split weights across
 * multiple SafeTensors files.
 *
 * Index file format  (model.safetensors.index.json)
 * ══════════════════════════════════════════════════
 *   {
 *     "metadata": {
 *         "total_size": 131072000000
 *     },
 *     "weight_map": {
 *         "model.embed_tokens.weight": "model-00001-of-00015.safetensors",
 *         "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00015.safetensors",
 *         ...
 *         "lm_head.weight": "model-00015-of-00015.safetensors"
 *     }
 *   }
 *
 * The index tells us WHICH shard file contains each tensor.
 * Shard files live in the same directory as the index file.
 *
 * Design
 * ══════
 * This class is intentionally separate from SafetensorFFMLoader to keep
 * single-file loading simple.  The {@link SafetensorShardLoader} wraps both
 * and exposes a unified API.
 */
package tech.kayys.gollek.safetensor.loader;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

import tech.kayys.gollek.safetensor.exception.SafetensorException;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Parsed representation of a {@code model.safetensors.index.json} file.
 *
 * <p>
 * Provides tensor-name → shard-file lookup and shard enumeration.
 * Immutable and thread-safe after construction.
 */
public final class SafetensorShardIndex {

    private static final Logger log = Logger.getLogger(SafetensorShardIndex.class);

    /** Canonical file name that triggers multi-shard detection. */
    public static final String INDEX_FILE_NAME = "model.safetensors.index.json";
    public static final String INDEX_FILE_NAME_ALT = "model.safetensor.index.json";

    // ─────────────────────────────────────────────────────────────────────────

    /** Directory containing both the index file and all shard files. */
    private final Path modelDir;

    /** Path to the index JSON file itself. */
    private final Path indexPath;

    /** Optional total-size metadata entry. */
    private final long totalSize;

    /**
     * Map: tensor name → shard file name (just the filename, not the full path).
     * Insertion-ordered to preserve the logical model order.
     */
    private final Map<String, String> weightMap;

    /**
     * Derived: set of unique shard file names referenced by this index.
     * Sorted for deterministic iteration (shard-00001 before shard-00002, etc.).
     */
    private final List<String> shardFileNames;

    // ─────────────────────────────────────────────────────────────────────────

    private SafetensorShardIndex(
            Path modelDir,
            Path indexPath,
            long totalSize,
            Map<String, String> weightMap) {
        this.modelDir = Objects.requireNonNull(modelDir);
        this.indexPath = Objects.requireNonNull(indexPath);
        this.totalSize = totalSize;
        this.weightMap = Collections.unmodifiableMap(new LinkedHashMap<>(weightMap));

        // Collect unique shard file names and sort them
        SortedSet<String> unique = new TreeSet<>(weightMap.values());
        this.shardFileNames = Collections.unmodifiableList(new ArrayList<>(unique));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Factory / loading
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Parse an index file from a specific path.
     *
     * @param indexPath path to the {@code model.safetensors.index.json} file
     * @param mapper    Jackson mapper
     * @return the parsed index
     * @throws SafetensorException on I/O or parsing errors
     */
    public static SafetensorShardIndex load(Path indexPath, ObjectMapper mapper) {
        Objects.requireNonNull(indexPath, "indexPath must not be null");
        Objects.requireNonNull(mapper, "mapper must not be null");

        Path abs = indexPath.toAbsolutePath().normalize();
        if (!Files.exists(abs)) {
            throw new SafetensorException.IoException("Index file not found: " + abs, abs, null);
        }

        Path modelDir = abs.getParent();
        if (modelDir == null) {
            throw new SafetensorException.ValidationException(
                    "Index file has no parent directory", abs);
        }

        try {
            byte[] json = Files.readAllBytes(abs);
            IndexDto dto = mapper.readValue(json, IndexDto.class);

            if (dto.weightMap == null || dto.weightMap.isEmpty()) {
                throw new SafetensorException.ValidationException(
                        "Index file has empty or missing 'weight_map'", abs);
            }

            long totalSize = dto.metadata != null && dto.metadata.totalSize != null
                    ? dto.metadata.totalSize
                    : -1L;

            // Validate that all referenced shard files actually exist
            Set<String> missingShards = new LinkedHashSet<>();
            for (String shardName : new TreeSet<>(dto.weightMap.values())) {
                if (!Files.exists(modelDir.resolve(shardName))) {
                    missingShards.add(shardName);
                }
            }
            if (!missingShards.isEmpty()) {
                throw new SafetensorException.ShardException(
                        "Index references " + missingShards.size() + " missing shard file(s): "
                                + missingShards,
                        abs, null);
            }

            log.infof("Loaded SafeTensors index: %d tensors across %d shards [%s]",
                    dto.weightMap.size(),
                    new TreeSet<>(dto.weightMap.values()).size(),
                    abs.getFileName());

            return new SafetensorShardIndex(modelDir, abs, totalSize, dto.weightMap);

        } catch (SafetensorException e) {
            throw e;
        } catch (IOException e) {
            throw new SafetensorException.IoException(
                    "Failed to parse shard index JSON", abs, e);
        }
    }

    /**
     * Check whether a given directory contains a multi-shard index file.
     *
     * @param modelDir the directory to check
     * @return {@code true} if an index file is present
     */
    public static boolean isShardedModel(Path modelDir) {
        if (modelDir == null || !Files.isDirectory(modelDir))
            return false;
        return Files.exists(modelDir.resolve(INDEX_FILE_NAME))
                || Files.exists(modelDir.resolve(INDEX_FILE_NAME_ALT));
    }

    /**
     * Resolve the index file path within a model directory, or throw if absent.
     */
    public static Path resolveIndexPath(Path modelDir) {
        Path primary = modelDir.resolve(INDEX_FILE_NAME);
        if (Files.exists(primary))
            return primary;
        Path alt = modelDir.resolve(INDEX_FILE_NAME_ALT);
        if (Files.exists(alt))
            return alt;
        throw new SafetensorException.IoException(
                "No shard index file found in directory: " + modelDir, modelDir, null);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────────────

    /** Directory containing all shard files and the index. */
    public Path modelDir() {
        return modelDir;
    }

    /** Path to the index JSON file. */
    public Path indexPath() {
        return indexPath;
    }

    /**
     * Total model byte size as declared in the index metadata, or {@code -1} if
     * absent.
     */
    public long totalSize() {
        return totalSize;
    }

    /**
     * Full tensor name → shard filename mapping.
     * The shard filename is a bare name (e.g.
     * {@code "model-00001-of-00015.safetensors"}),
     * not an absolute path.
     */
    public Map<String, String> weightMap() {
        return weightMap;
    }

    /**
     * Ordered list of unique shard file names (sorted lexicographically).
     * Shard files are always in the same directory as the index.
     */
    public List<String> shardFileNames() {
        return shardFileNames;
    }

    /** Number of unique shard files. */
    public int shardCount() {
        return shardFileNames.size();
    }

    /** Total number of tensors across all shards. */
    public int tensorCount() {
        return weightMap.size();
    }

    /**
     * Resolve the shard file path for a given tensor name.
     *
     * @param tensorName the tensor key
     * @return absolute path to the shard file containing this tensor
     * @throws SafetensorException.TensorNotFoundException if not in the index
     */
    public Path shardPathForTensor(String tensorName) {
        String shardName = weightMap.get(tensorName);
        if (shardName == null) {
            throw new SafetensorException.TensorNotFoundException(tensorName, indexPath);
        }
        return modelDir.resolve(shardName);
    }

    /**
     * Absolute path to a shard file by its bare filename.
     *
     * @param shardFileName bare shard filename (from {@link #shardFileNames()})
     * @return absolute shard path
     */
    public Path shardPath(String shardFileName) {
        return modelDir.resolve(shardFileName);
    }

    /**
     * Return all tensor names that live in a specific shard.
     *
     * @param shardFileName bare shard filename
     * @return ordered list of tensor names in that shard
     */
    public List<String> tensorsInShard(String shardFileName) {
        List<String> result = new ArrayList<>();
        weightMap.forEach((tensorName, shard) -> {
            if (shard.equals(shardFileName))
                result.add(tensorName);
        });
        return Collections.unmodifiableList(result);
    }

    /** Whether the index contains a specific tensor. */
    public boolean hasTensor(String tensorName) {
        return weightMap.containsKey(tensorName);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Jackson DTOs (package-private)
    // ─────────────────────────────────────────────────────────────────────────

    @JsonIgnoreProperties(ignoreUnknown = true)
    static final class IndexDto {
        @JsonProperty("metadata")
        MetadataDto metadata;

        @JsonProperty("weight_map")
        Map<String, String> weightMap;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static final class MetadataDto {
        @JsonProperty("total_size")
        Long totalSize;
    }

    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public String toString() {
        return "SafetensorShardIndex{"
                + "shards=" + shardCount()
                + ", tensors=" + tensorCount()
                + ", totalSize=" + totalSize
                + ", dir=" + modelDir
                + '}';
    }
}
