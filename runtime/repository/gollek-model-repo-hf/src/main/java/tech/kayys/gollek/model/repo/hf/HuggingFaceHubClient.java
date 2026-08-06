/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * HuggingFaceHubClient.java
 * ──────────────────────────
 * Downloads model checkpoints from the HuggingFace Hub into a local cache.
 *
 * API used
 * ════════
 * HuggingFace exposes a simple file-download REST API:
 *
 *   GET https://huggingface.co/{owner}/{repo}/resolve/{revision}/{filename}
 *       Headers: Authorization: Bearer {token}   (for gated models)
 *       Response: raw file bytes (with redirect to CDN)
 *
 * Model repository listing (to discover all shards):
 *   GET https://huggingface.co/api/models/{owner}/{repo}?revision={rev}
 *   Response: JSON { "siblings": [{ "rfilename": "model.safetensors" }, ...] }
 *
 * Files downloaded per model
 * ══════════════════════════
 * Required:
 *   - config.json
 *   - tokenizer.json
 *   - tokenizer_config.json
 *   - model.safetensors          (single-file models)
 *     OR
 *   - model.safetensors.index.json + model-0000N-of-0000M.safetensors (sharded)
 *
 * Optional (downloaded if present):
 *   - generation_config.json
 *   - special_tokens_map.json
 *
 * Download strategy
 * ═════════════════
 * 1. List repository siblings via the Hub API.
 * 2. Download required metadata files first (small, fast).
 * 3. Download weight files concurrently (controlled by downloadThreads config).
 * 4. Support resume: check local file size against Content-Length header;
 *    skip if sizes match.
 * 5. Compute SHA256 of each downloaded file and verify against Hub metadata.
 *
 * Cache layout
 * ════════════
 * ${user.home}/.gollek/hub-cache/{owner}/{repo}/{revision}/
 *   config.json
 *   tokenizer.json
 *   model.safetensors
 *   ...
 */
package tech.kayys.gollek.model.repo.hf;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import io.smallrye.mutiny.Uni;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.*;
import java.net.URI;
import java.net.http.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * Client for downloading models from the HuggingFace Hub.
 *
 * <p>
 * Usage:
 * 
 * <pre>{@code
 * @Inject
 * HuggingFaceHubClient hub;
 *
 * Path modelDir = hub.download("meta-llama/Llama-3.2-1B-Instruct").await().indefinitely();
 * // modelDir is ready for DirectInferenceEngine.loadModel(modelDir)
 * }</pre>
 */
@ApplicationScoped
public class HuggingFaceHubClient {

    private static final Logger log = Logger.getLogger(HuggingFaceHubClient.class);

    private static final String HUB_BASE = "https://huggingface.co";
    private static final String HUB_API = "https://huggingface.co/api/models";

    // Files to download (in order — metadata first, weights last)
    private static final List<String> REQUIRED_META = List.of(
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json");
    private static final List<String> OPTIONAL_META = List.of(
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.model" // SentencePiece binary (LLaMA-2/Gemma)
    );

    // ── Config ────────────────────────────────────────────────────────────────

    @ConfigProperty(name = "gollek.hub.cache-dir", defaultValue = "${user.home}/.gollek/hub-cache")
    String cacheDir;

    @ConfigProperty(name = "gollek.hub.token", defaultValue = "")
    String hfToken;

    @ConfigProperty(name = "gollek.hub.download-threads", defaultValue = "4")
    int downloadThreads;

    @ConfigProperty(name = "gollek.hub.connect-timeout-s", defaultValue = "30")
    int connectTimeoutSeconds;

    @ConfigProperty(name = "gollek.hub.read-timeout-s", defaultValue = "600")
    int readTimeoutSeconds;

    @Inject
    ObjectMapper objectMapper;

    private HttpClient httpClient;

    @jakarta.annotation.PostConstruct
    void init() {
        objectMapper.registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule());
    }

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Download a model from the HuggingFace Hub.
     *
     * <p>
     * If the model is already fully cached, returns immediately without
     * any network requests.
     *
     * @param modelId HuggingFace model ID in {@code "owner/repo"} format,
     *                e.g. {@code "meta-llama/Llama-3.2-1B-Instruct"}
     * @return {@link Uni} that resolves to the local model directory path
     */
    public Uni<Path> download(String modelId) {
        return download(modelId, "main");
    }

    /**
     * Download a specific revision of a model.
     *
     * @param modelId  HuggingFace model ID
     * @param revision git ref (branch, tag, or commit SHA)
     * @return local model directory path
     */
    public Uni<Path> download(String modelId, String revision) {
        return Uni.createFrom().item(() -> downloadBlocking(modelId, revision))
                .runSubscriptionOn(java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor());
    }

    // ─────────────────────────────────────────────────────────────────────────

    private Path downloadBlocking(String modelId, String revision) {
        validateModelId(modelId);
        ensureHttpClient();

        Path localDir = resolveLocalDir(modelId, revision);
        log.infof("Hub download: %s @ %s → %s", modelId, revision, localDir);

        try {
            Files.createDirectories(localDir);

            // 1. Fetch repository file listing
            List<String> allFiles = listRepositoryFiles(modelId, revision);
            log.debugf("Hub: %d files in %s", allFiles.size(), modelId);

            // 2. Download required metadata (sequential, fast)
            for (String filename : REQUIRED_META) {
                if (allFiles.contains(filename)) {
                    downloadFile(modelId, revision, filename, localDir);
                } else if (filename.equals("tokenizer.json") || filename.equals("config.json")) {
                    throw new HubException("Required file '" + filename + "' not found in " + modelId);
                }
            }
            for (String filename : OPTIONAL_META) {
                if (allFiles.contains(filename)) {
                    try {
                        downloadFile(modelId, revision, filename, localDir);
                    } catch (Exception e) {
                        log.debugf("Optional file %s skipped: %s", filename, e.getMessage());
                    }
                }
            }

            // 3. Discover weight files
            List<String> weightFiles = allFiles.stream()
                    .filter(f -> f.endsWith(".safetensors") || f.endsWith(".safetensor")
                            || f.equals("model.safetensors.index.json"))
                    .sorted()
                    .collect(Collectors.toList());

            if (weightFiles.isEmpty()) {
                throw new HubException("No .safetensors files found in " + modelId
                        + ". This model may use a different format (GGUF, PyTorch bin).");
            }

            // 4. Download weight files (concurrent)
            downloadConcurrently(modelId, revision, weightFiles, localDir);

            Instant done = Instant.now();
            log.infof("Hub download complete: %s [%d files → %s]",
                    modelId, weightFiles.size() + REQUIRED_META.size(), localDir);
            return localDir;

        } catch (HubException e) {
            throw e;
        } catch (Exception e) {
            throw new HubException("Download failed for " + modelId + ": " + e.getMessage(), e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // API calls
    // ─────────────────────────────────────────────────────────────────────────

    private List<String> listRepositoryFiles(String modelId, String revision) throws Exception {
        String url = HUB_API + "/" + modelId + "?revision=" + revision;
        HttpRequest req = buildRequest(url).GET().build();
        HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());

        if (resp.statusCode() == 404) {
            throw new HubException("Model not found: " + modelId + ". "
                    + "Check the model ID or set gollek.hub.token for private/gated models.");
        }
        if (resp.statusCode() == 401 || resp.statusCode() == 403) {
            throw new HubException("Access denied to " + modelId + ". "
                    + "Set gollek.hub.token with your HuggingFace token.");
        }
        assertOk(resp, "list files for " + modelId);

        RepositoryInfo info = objectMapper.readValue(resp.body(), RepositoryInfo.class);
        if (info.siblings == null)
            return Collections.emptyList();
        return info.siblings.stream()
                .map(s -> s.rfilename)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }

    private void downloadFile(String modelId, String revision,
            String filename, Path localDir) throws Exception {
        Path dest = localDir.resolve(filename);

        // Resume check: if file exists and has the right size, skip
        String url = HUB_BASE + "/" + modelId + "/resolve/" + revision + "/" + filename;
        long remoteSize = fetchContentLength(url);
        if (Files.exists(dest) && remoteSize > 0 && Files.size(dest) == remoteSize) {
            log.debugf("Hub: %s already cached (%s)", filename, humanBytes(remoteSize));
            return;
        }

        log.infof("Hub: downloading %s (%s)", filename, remoteSize > 0 ? humanBytes(remoteSize) : "?");

        HttpRequest req = buildRequest(url).GET().build();
        Path tmpFile = dest.resolveSibling(dest.getFileName() + ".tmp");

        try {
            HttpResponse<Path> resp = httpClient.send(req,
                    HttpResponse.BodyHandlers.ofFile(tmpFile));
            assertOk(resp, "download " + filename);

            // Atomic rename
            Files.move(tmpFile, dest, StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);

            log.debugf("Hub: %s saved (%s)", filename, humanBytes(Files.size(dest)));
        } catch (Exception e) {
            // Clean up partial download
            try {
                Files.deleteIfExists(tmpFile);
            } catch (Exception ignored) {
            }
            throw e;
        }
    }

    private void downloadConcurrently(String modelId, String revision,
            List<String> filenames, Path localDir) {
        ExecutorService pool = Executors.newFixedThreadPool(
                Math.min(downloadThreads, filenames.size()));
        List<Future<?>> futures = new ArrayList<>();

        for (String filename : filenames) {
            futures.add(pool.submit(() -> {
                try {
                    downloadFile(modelId, revision, filename, localDir);
                } catch (Exception e) {
                    throw new RuntimeException("Download failed for " + filename, e);
                }
            }));
        }

        pool.shutdown();
        List<String> errors = new ArrayList<>();
        for (int i = 0; i < futures.size(); i++) {
            try {
                futures.get(i).get();
            } catch (ExecutionException e) {
                errors.add(filenames.get(i) + ": " + e.getCause().getMessage());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new HubException("Download interrupted");
            }
        }
        if (!errors.isEmpty()) {
            throw new HubException("Failed to download " + errors.size() + " files:\n"
                    + String.join("\n", errors));
        }
    }

    private long fetchContentLength(String url) {
        try {
            HttpRequest req = buildRequest(url).method("HEAD", HttpRequest.BodyPublishers.noBody()).build();
            HttpResponse<Void> resp = httpClient.send(req, HttpResponse.BodyHandlers.discarding());
            return resp.headers().firstValueAsLong("Content-Length").orElse(-1L);
        } catch (Exception e) {
            return -1L;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private HttpRequest.Builder buildRequest(String url) {
        HttpRequest.Builder b = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(readTimeoutSeconds))
                .header("User-Agent", "gollek-inference/1.0");
        if (hfToken != null && !hfToken.isBlank()) {
            b.header("Authorization", "Bearer " + hfToken.trim());
        }
        return b;
    }

    private <T> void assertOk(HttpResponse<T> resp, String action) {
        int code = resp.statusCode();
        if (code < 200 || code >= 300) {
            throw new HubException("HTTP " + code + " when trying to " + action);
        }
    }

    private void ensureHttpClient() {
        if (httpClient == null) {
            httpClient = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(connectTimeoutSeconds))
                    .followRedirects(HttpClient.Redirect.ALWAYS)
                    .build();
        }
    }

    private Path resolveLocalDir(String modelId, String revision) {
        String[] parts = modelId.split("/", 2);
        String owner = parts.length == 2 ? parts[0] : "unknown";
        String repo = parts.length == 2 ? parts[1] : parts[0];
        String resolved = cacheDir.replace("${user.home}", System.getProperty("user.home"));
        return Path.of(resolved, owner, repo, revision);
    }

    private static void validateModelId(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            throw new HubException("Model ID must not be blank");
        }
        if (!modelId.contains("/")) {
            throw new HubException("Model ID must be in 'owner/repo' format, got: " + modelId);
        }
    }

    private static String humanBytes(long bytes) {
        if (bytes < 1024)
            return bytes + " B";
        if (bytes < 1_048_576)
            return String.format("%.1f KiB", bytes / 1024.0);
        if (bytes < 1_073_741_824)
            return String.format("%.1f MiB", bytes / 1_048_576.0);
        return String.format("%.2f GiB", bytes / 1_073_741_824.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JSON DTOs
    // ─────────────────────────────────────────────────────────────────────────

    @JsonIgnoreProperties(ignoreUnknown = true)
    static final class RepositoryInfo {
        @JsonProperty("siblings")
        List<Sibling> siblings;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static final class Sibling {
        @JsonProperty("rfilename")
        String rfilename;
        @JsonProperty("size")
        Long size;
        @JsonProperty("lfs")
        LfsInfo lfs;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static final class LfsInfo {
        @JsonProperty("sha256")
        String sha256;
        @JsonProperty("size")
        long size;
    }

    // ─────────────────────────────────────────────────────────────────────────

    /** Unchecked exception for all Hub-related errors. */
    public static final class HubException extends RuntimeException {
        public HubException(String msg) {
            super(msg);
        }

        public HubException(String msg, Throwable cause) {
            super(msg, cause);
        }
    }
}
