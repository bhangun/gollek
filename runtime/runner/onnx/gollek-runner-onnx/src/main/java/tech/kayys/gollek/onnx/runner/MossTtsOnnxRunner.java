package tech.kayys.gollek.onnx.runner;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxTensorLike;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.suling.audio.AudioEncodeOptions;
import tech.kayys.suling.audio.AudioProcessingOptions;
import tech.kayys.suling.audio.EncodedMedia;
import tech.kayys.suling.audio.PcmAudio;
import tech.kayys.suling.audio.Suling;

import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HexFormat;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.locks.ReentrantLock;

@ApplicationScoped
public class MossTtsOnnxRunner {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String ORT_JAVA_VERSIONED_MAC_DYLIB = "libonnxruntime.1.19.2.dylib";
    private static final String OPTIMIZED_CACHE_ENABLED_PROPERTY = "gollek.onnx.optimized-model-cache.enabled";
    private static final String OPTIMIZED_CACHE_ENABLED_ENV = "GOLLEK_ONNX_OPTIMIZED_MODEL_CACHE";
    private static final String OPTIMIZED_CACHE_DIR_PROPERTY = "gollek.onnx.optimized-model-cache.dir";
    private static final String OPTIMIZED_CACHE_DIR_ENV = "GOLLEK_ONNX_OPTIMIZED_MODEL_CACHE_DIR";
    private static final String SESSION_CACHE_ENABLED_PROPERTY = "gollek.onnx.moss-tts.session-cache.enabled";
    private static final String SESSION_CACHE_ENABLED_ENV = "GOLLEK_ONNX_MOSS_TTS_SESSION_CACHE";
    private static final String SESSION_CACHE_MAX_ENTRIES_PROPERTY = "gollek.onnx.moss-tts.session-cache.max-entries";
    private static final String SESSION_CACHE_MAX_ENTRIES_ENV = "GOLLEK_ONNX_MOSS_TTS_SESSION_CACHE_MAX_ENTRIES";
    private static final String ASSET_CACHE_ENABLED_PROPERTY = "gollek.onnx.moss-tts.asset-cache.enabled";
    private static final String ASSET_CACHE_ENABLED_ENV = "GOLLEK_ONNX_MOSS_TTS_ASSET_CACHE";
    private static final String ASSET_CACHE_MAX_ENTRIES_PROPERTY = "gollek.onnx.moss-tts.asset-cache.max-entries";
    private static final String ASSET_CACHE_MAX_ENTRIES_ENV = "GOLLEK_ONNX_MOSS_TTS_ASSET_CACHE_MAX_ENTRIES";
    private static final String ORT_LOG_LEVEL_PROPERTY = "gollek.onnx.log-level";
    private static final String ORT_LOG_LEVEL_ENV = "GOLLEK_ONNX_LOG_LEVEL";
    private static final MossTtsProgressListener NOOP_PROGRESS_LISTENER = progress -> {
    };
    private static final MossTtsPcmChunkListener NOOP_PCM_CHUNK_LISTENER = chunk -> {
    };
    private final Map<String, CachedMossSessions> sessionCache = new LinkedHashMap<>(16, 0.75f, true);
    private final Map<String, MossModelAssets> assetCache = new LinkedHashMap<>(16, 0.75f, true);

    public MossTtsAudio synthesize(Path ttsDir, Path codecDir, InferenceRequest request, int threads) throws Exception {
        return synthesize(ttsDir, codecDir, request, threads, NOOP_PROGRESS_LISTENER);
    }

    public MossTtsAudio synthesize(
            Path ttsDir,
            Path codecDir,
            InferenceRequest request,
            int threads,
            MossTtsProgressListener progressListener) throws Exception {
        return synthesize(ttsDir, codecDir, request, threads, progressListener, NOOP_PCM_CHUNK_LISTENER);
    }

    public MossTtsAudio synthesize(
            Path ttsDir,
            Path codecDir,
            InferenceRequest request,
            int threads,
            MossTtsProgressListener progressListener,
            MossTtsPcmChunkListener pcmChunkListener) throws Exception {
        Objects.requireNonNull(ttsDir, "ttsDir");
        Objects.requireNonNull(codecDir, "codecDir");
        MossTtsProgressListener progress = progressListener == null ? NOOP_PROGRESS_LISTENER : progressListener;
        MossTtsPcmChunkListener pcmChunks = pcmChunkListener == null ? NOOP_PCM_CHUNK_LISTENER : pcmChunkListener;
        Instant started = Instant.now();

        MossModelAssets assets = loadAssets(ttsDir, codecDir);
        JsonNode manifest = assets.manifest();
        JsonNode ttsMeta = assets.ttsMeta();
        JsonNode codecMeta = assets.codecMeta();
        MossSentencePieceTokenizer tokenizer = assets.tokenizer();

        String prompt = normalizePrompt(request.getPrompt());
        int[] textTokenIds = tokenizer.encode(prompt);
        VoicePrompt voicePrompt = resolveVoicePrompt(manifest, request, prompt);
        int[][] rows = buildVoiceCloneRows(manifest, voicePrompt.promptAudioCodes(), textTokenIds);
        FrameLimit frameLimit = resolveFrameLimit(manifest, codecMeta, request);
        int maxFrames = frameLimit.frames();
        ResolvedSeed seed = resolvedSeed(request);
        Random random = new Random(seed.value());
        double frameRate = codecFrameRate(codecMeta);
        int progressEveryFrames = requestedStreamProgressFrames(request, maxFrames);

        ensureOrtJavaNativeDependency();
        OrtEnvironment env = OrtEnvironment.getEnvironment(ortLoggingLevel(), "gollek-moss-tts");
        try (MossSessionLease lease = acquireSessions(env, ttsDir, codecDir, ttsMeta, codecMeta, threads)) {
            MossSessions sessions = lease.sessions();
            progress.onProgress(new MossTtsProgress("prefill", 0, maxFrames, "TTS prefill"));
            GeneratedFrames generated = generateFrames(
                    env,
                    sessions,
                    manifest,
                    ttsMeta,
                    codecMeta,
                    rows,
                    frameLimit,
                    random,
                    request,
                    progress,
                    progressEveryFrames,
                    pcmChunks);
            List<int[]> generatedFrames = generated.frames();
            if (generatedFrames.isEmpty()) {
                throw new IllegalStateException("MOSS TTS generated no audio frames.");
            }
            progress.onProgress(new MossTtsProgress("decode", generatedFrames.size(), maxFrames, "TTS decode"));
            DecodedAudio decoded = normalizeDecodedAudioChannels(
                    decodeAudio(
                            env,
                            sessions,
                            codecMeta,
                            generatedFrames,
                            request,
                            generated.streamedPcmChunks() > 0 ? NOOP_PCM_CHUNK_LISTENER : pcmChunks),
                    request);
            Map<String, String> audioInfo = wavInfoMetadata(
                    ttsDir, prompt, voicePrompt, seed, generated.stopReason(), started);
            PcmAudio pcm = PcmAudio.fromChannelMajorFloat(
                    decoded.channelMajorAudio(), decoded.channels(), decoded.samples(), decoded.sampleRate(), audioInfo);
            progress.onProgress(new MossTtsProgress("polish", generatedFrames.size(), maxFrames, "TTS polish"));
            pcm = Suling.process(pcm, audioProcessingOptions(request));
            AudioEncodeOptions encodeOptions = audioEncodeOptions(request, audioInfo);
            progress.onProgress(new MossTtsProgress(
                    "encode",
                    generatedFrames.size(),
                    maxFrames,
                    "TTS encode " + encodeOptions.format().toUpperCase(Locale.ROOT)));
            EncodedMedia encoded = Suling.encode(pcm, encodeOptions);
            long durationMs = Duration.between(started, Instant.now()).toMillis();
            double generationSeconds = Math.max(durationMs / 1000.0, 0.001);
            double audioDurationSeconds = decoded.samples() / (double) decoded.sampleRate();
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("runner", "moss-tts-onnx");
            metadata.put("effective_provider", "onnx");
            metadata.put("execution_backend", "onnxruntime-java");
            metadata.put("onnx_log_level", ortLoggingLevelName());
            sessions.cacheStats().putMetadata(metadata);
            assets.putMetadata(metadata);
            lease.putMetadata(metadata);
            metadata.put("audio_format", encoded.format());
            metadata.put("audio_mime", encoded.mimeType());
            metadata.put("audio_encoder", "suling");
            metadata.putAll(encoded.metadata());
            copyAudioProcessingMetadata(pcm, metadata);
            metadata.put("audio_sample_rate", decoded.sampleRate());
            metadata.put("audio_channels", decoded.channels());
            metadata.put("audio_source_channels", decoded.sourceChannels());
            metadata.put("audio_codec_decode_mode", decoded.codecDecodeMode());
            metadata.put("audio_channel_mode", decoded.channelMode());
            metadata.put("audio_samples", decoded.samples());
            metadata.put("audio_bytes", encoded.bytes().length);
            metadata.put("audio_duration_seconds", audioDurationSeconds);
            metadata.put("audio_generation_duration_seconds", generationSeconds);
            metadata.put("audio_realtime_speed", audioDurationSeconds / generationSeconds);
            metadata.put("audio_realtime_factor", generationSeconds / Math.max(audioDurationSeconds, 0.001));
            metadata.put("tts_frames", generatedFrames.size());
            metadata.put("tts_frames_per_second", generatedFrames.size() / generationSeconds);
            metadata.put("tts_max_frames", maxFrames);
            metadata.put("tts_streamed_pcm_chunks", generated.streamedPcmChunks());
            metadata.put("tts_low_latency_pcm", generated.streamedPcmChunks() > 0);
            metadata.put("tts_limit_source", frameLimit.source());
            metadata.put("tts_stop_reason", generated.stopReason());
            metadata.put("tts_truncated", !"stop_token".equals(generated.stopReason()));
            metadata.put("tts_frame_rate", frameRate);
            metadata.put("tts_frame_duration_ms", 1000.0 / frameRate);
            metadata.put("tts_voice", voicePrompt.label());
            metadata.put("tts_voice_id", voicePrompt.id());
            metadata.put("tts_voice_mode", voicePrompt.mode());
            if (!voicePrompt.language().isBlank()) {
                metadata.put("tts_voice_language", voicePrompt.language());
            }
            metadata.put("tts_seed", seed.value());
            metadata.put("tts_seed_source", seed.source());
            metadata.put("tokens.input", textTokenIds.length);
            metadata.put("bench.latency_ms", durationMs);
            progress.onProgress(new MossTtsProgress("complete", generatedFrames.size(), maxFrames, "TTS complete"));
            return new MossTtsAudio(encoded.bytes(), encoded.format(), encoded.mimeType(), decoded.sampleRate(),
                    decoded.channels(), decoded.samples(), generatedFrames.size(), prompt, metadata);
        }
    }

    public String encodeAudioBase64(MossTtsAudio audio) {
        return Base64.getEncoder().encodeToString(audio.bytes());
    }

    public MossTtsWarmup warmup(Path ttsDir, Path codecDir, int threads) throws Exception {
        Objects.requireNonNull(ttsDir, "ttsDir");
        Objects.requireNonNull(codecDir, "codecDir");
        Instant started = Instant.now();
        MossModelAssets assets = loadAssets(ttsDir, codecDir);
        ensureOrtJavaNativeDependency();
        OrtEnvironment env = OrtEnvironment.getEnvironment(ortLoggingLevel(), "gollek-moss-tts");
        try (MossSessionLease lease = acquireSessions(env, ttsDir, codecDir, assets.ttsMeta(), assets.codecMeta(), threads)) {
            MossSessions sessions = lease.sessions();
            sessions.codecDecode();
            sessions.codecDecodeStep();
            long durationMs = Duration.between(started, Instant.now()).toMillis();
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("runner", "moss-tts-onnx");
            metadata.put("effective_provider", "onnx");
            metadata.put("execution_backend", "onnxruntime-java");
            metadata.put("onnx_log_level", ortLoggingLevelName());
            metadata.put("onnx_warmup", true);
            metadata.put("onnx_warmup_duration_ms", durationMs);
            metadata.put("onnx_warmup_threads", Math.max(1, threads));
            metadata.put("onnx_warmup_sessions_requested", 5);
            metadata.put("onnx_warmup_assets", true);
            assets.putMetadata(metadata);
            sessions.cacheStats().putMetadata(metadata);
            lease.putMetadata(metadata);
            return new MossTtsWarmup(durationMs, metadata);
        }
    }

    private MossModelAssets loadAssets(Path ttsDir, Path codecDir) throws Exception {
        boolean enabled = mossAssetCacheEnabled();
        int maxEntries = mossAssetCacheMaxEntries();
        String key = MossModelAssets.cacheKey(ttsDir, codecDir);
        if (!enabled || maxEntries <= 0) {
            return MossModelAssets.load(ttsDir, codecDir, key, "disabled", enabled, maxEntries, 0);
        }
        synchronized (assetCache) {
            MossModelAssets cached = assetCache.get(key);
            if (cached != null) {
                cached = cached.withCacheState("hit", assetCache.size(), maxEntries);
                assetCache.put(key, cached);
                return cached;
            }
            MossModelAssets loaded = MossModelAssets.load(ttsDir, codecDir, key, "miss", true, maxEntries, assetCache.size() + 1);
            assetCache.put(key, loaded);
            evictAssetsLocked(maxEntries);
            return loaded.withCacheState("miss", assetCache.size(), maxEntries);
        }
    }

    private void evictAssetsLocked(int maxEntries) {
        Iterator<Map.Entry<String, MossModelAssets>> iterator = assetCache.entrySet().iterator();
        while (assetCache.size() > maxEntries && iterator.hasNext()) {
            iterator.next();
            iterator.remove();
        }
    }

    private MossSessionLease acquireSessions(
            OrtEnvironment env,
            Path ttsDir,
            Path codecDir,
            JsonNode ttsMeta,
            JsonNode codecMeta,
            int threads) throws OrtException {
        boolean enabled = mossSessionCacheEnabled();
        int maxEntries = mossSessionCacheMaxEntries();
        if (!enabled || maxEntries <= 0) {
            return MossSessionLease.ephemeral(MossSessions.open(env, ttsDir, codecDir, ttsMeta, codecMeta, threads),
                    enabled, maxEntries);
        }

        String key = MossSessions.sessionCacheKey(ttsDir, codecDir, ttsMeta, codecMeta, threads);
        CachedMossSessions cached;
        boolean hit;
        synchronized (sessionCache) {
            cached = sessionCache.get(key);
            hit = cached != null;
            if (cached == null) {
                MossSessions sessions = MossSessions.open(env, ttsDir, codecDir, ttsMeta, codecMeta, threads);
                cached = new CachedMossSessions(key, sessions);
                sessionCache.put(key, cached);
            }
            cached.active++;
            evictIdleSessionsLocked(maxEntries);
        }
        cached.lock.lock();
        return MossSessionLease.cached(this, cached, hit, maxEntries);
    }

    private void releaseCachedSessions(CachedMossSessions cached, int maxEntries) {
        synchronized (sessionCache) {
            cached.active = Math.max(0, cached.active - 1);
            evictIdleSessionsLocked(maxEntries);
        }
    }

    private void evictIdleSessionsLocked(int maxEntries) {
        if (maxEntries <= 0) {
            return;
        }
        Iterator<Map.Entry<String, CachedMossSessions>> iterator = sessionCache.entrySet().iterator();
        while (sessionCache.size() > maxEntries && iterator.hasNext()) {
            Map.Entry<String, CachedMossSessions> entry = iterator.next();
            CachedMossSessions cached = entry.getValue();
            if (cached.active > 0) {
                continue;
            }
            iterator.remove();
            closeQuietly(cached.sessions);
        }
    }

    private int sessionCacheSize() {
        synchronized (sessionCache) {
            return sessionCache.size();
        }
    }

    private static boolean mossSessionCacheEnabled() {
        String raw = firstNonBlank(
                System.getProperty(SESSION_CACHE_ENABLED_PROPERTY),
                System.getenv(SESSION_CACHE_ENABLED_ENV),
                "true");
        return !Set.of("0", "false", "no", "off", "disable", "disabled")
                .contains(raw.trim().toLowerCase(Locale.ROOT));
    }

    private static int mossSessionCacheMaxEntries() {
        String raw = firstNonBlank(
                System.getProperty(SESSION_CACHE_MAX_ENTRIES_PROPERTY),
                System.getenv(SESSION_CACHE_MAX_ENTRIES_ENV),
                "2");
        try {
            return Math.max(0, Integer.parseInt(raw.trim()));
        } catch (NumberFormatException e) {
            return 2;
        }
    }

    private static boolean mossAssetCacheEnabled() {
        String raw = firstNonBlank(
                System.getProperty(ASSET_CACHE_ENABLED_PROPERTY),
                System.getenv(ASSET_CACHE_ENABLED_ENV),
                "true");
        return !Set.of("0", "false", "no", "off", "disable", "disabled")
                .contains(raw.trim().toLowerCase(Locale.ROOT));
    }

    private static int mossAssetCacheMaxEntries() {
        String raw = firstNonBlank(
                System.getProperty(ASSET_CACHE_MAX_ENTRIES_PROPERTY),
                System.getenv(ASSET_CACHE_MAX_ENTRIES_ENV),
                "4");
        try {
            return Math.max(0, Integer.parseInt(raw.trim()));
        } catch (NumberFormatException e) {
            return 4;
        }
    }

    private static String firstNonBlank(String... values) {
        if (values == null) {
            return "";
        }
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value.trim();
            }
        }
        return "";
    }

    @PreDestroy
    public void close() {
        synchronized (sessionCache) {
            for (CachedMossSessions cached : sessionCache.values()) {
                closeQuietly(cached.sessions);
            }
            sessionCache.clear();
        }
        synchronized (assetCache) {
            assetCache.clear();
        }
    }

    private static final class CachedMossSessions {
        private final String key;
        private final MossSessions sessions;
        private final ReentrantLock lock = new ReentrantLock();
        private int active;

        private CachedMossSessions(String key, MossSessions sessions) {
            this.key = key;
            this.sessions = sessions;
        }
    }

    private static final class MossSessionLease implements AutoCloseable {
        private final MossTtsOnnxRunner owner;
        private final CachedMossSessions cached;
        private final MossSessions sessions;
        private final boolean cacheEnabled;
        private final boolean cacheHit;
        private final int maxEntries;
        private boolean closed;

        private MossSessionLease(
                MossTtsOnnxRunner owner,
                CachedMossSessions cached,
                MossSessions sessions,
                boolean cacheEnabled,
                boolean cacheHit,
                int maxEntries) {
            this.owner = owner;
            this.cached = cached;
            this.sessions = sessions;
            this.cacheEnabled = cacheEnabled;
            this.cacheHit = cacheHit;
            this.maxEntries = maxEntries;
        }

        private static MossSessionLease cached(
                MossTtsOnnxRunner owner,
                CachedMossSessions cached,
                boolean cacheHit,
                int maxEntries) {
            return new MossSessionLease(owner, cached, cached.sessions, true, cacheHit, maxEntries);
        }

        private static MossSessionLease ephemeral(MossSessions sessions, boolean cacheEnabled, int maxEntries) {
            return new MossSessionLease(null, null, sessions, cacheEnabled, false, maxEntries);
        }

        private MossSessions sessions() {
            return sessions;
        }

        private void putMetadata(Map<String, Object> metadata) {
            metadata.put("onnx_session_cache_enabled", cacheEnabled && maxEntries > 0);
            metadata.put("onnx_session_cache_hit", cacheHit);
            metadata.put("onnx_session_cache_state", cached == null
                    ? "disabled"
                    : (cacheHit ? "hit" : "miss"));
            metadata.put("onnx_session_cache_max_entries", Math.max(0, maxEntries));
            if (cached != null) {
                metadata.put("onnx_session_cache_key", cached.key);
                metadata.put("onnx_session_cache_entries", owner == null ? 0 : owner.sessionCacheSize());
            }
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            if (cached == null) {
                closeQuietly(sessions);
                return;
            }
            try {
                cached.lock.unlock();
            } finally {
                if (owner != null) {
                    owner.releaseCachedSessions(cached, maxEntries);
                }
            }
        }
    }

    private record MossModelAssets(
            JsonNode manifest,
            JsonNode ttsMeta,
            JsonNode codecMeta,
            MossSentencePieceTokenizer tokenizer,
            String cacheKey,
            String cacheState,
            boolean cacheEnabled,
            int maxEntries,
            int entries) {
        private static MossModelAssets load(
                Path ttsDir,
                Path codecDir,
                String cacheKey,
                String cacheState,
                boolean cacheEnabled,
                int maxEntries,
                int entries) throws Exception {
            JsonNode manifest = readJson(ttsDir.resolve("browser_poc_manifest.json"));
            JsonNode ttsMeta = readJson(ttsDir.resolve("tts_browser_onnx_meta.json"));
            JsonNode codecMeta = readJson(codecDir.resolve("codec_browser_onnx_meta.json"));
            MossSentencePieceTokenizer tokenizer = MossSentencePieceTokenizer.load(
                    ttsDir.resolve(manifest.path("model_files").path("tokenizer_model").asText("tokenizer.model")));
            return new MossModelAssets(
                    manifest,
                    ttsMeta,
                    codecMeta,
                    tokenizer,
                    cacheKey,
                    cacheState,
                    cacheEnabled,
                    maxEntries,
                    entries);
        }

        private static String cacheKey(Path ttsDir, Path codecDir) throws Exception {
            Path manifestPath = ttsDir.resolve("browser_poc_manifest.json");
            JsonNode manifest = readJson(manifestPath);
            List<Path> paths = List.of(
                    manifestPath,
                    ttsDir.resolve("tts_browser_onnx_meta.json"),
                    codecDir.resolve("codec_browser_onnx_meta.json"),
                    ttsDir.resolve(manifest.path("model_files").path("tokenizer_model").asText("tokenizer.model")));
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (Path path : paths) {
                Path modelPath = path.toAbsolutePath().normalize();
                String text = String.join("\t",
                        modelPath.toRealPath().toString(),
                        String.valueOf(Files.size(modelPath)),
                        String.valueOf(Files.getLastModifiedTime(modelPath).toMillis()));
                digest.update(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
                digest.update((byte) '\n');
            }
            return HexFormat.of().formatHex(digest.digest()).substring(0, 24);
        }

        private MossModelAssets withCacheState(String state, int entries, int maxEntries) {
            return new MossModelAssets(
                    manifest,
                    ttsMeta,
                    codecMeta,
                    tokenizer,
                    cacheKey,
                    state,
                    cacheEnabled,
                    maxEntries,
                    entries);
        }

        private void putMetadata(Map<String, Object> metadata) {
            metadata.put("onnx_asset_cache_enabled", cacheEnabled && maxEntries > 0);
            metadata.put("onnx_asset_cache_state", cacheState);
            metadata.put("onnx_asset_cache_key", cacheKey);
            metadata.put("onnx_asset_cache_entries", entries);
            metadata.put("onnx_asset_cache_max_entries", Math.max(0, maxEntries));
        }
    }

    private static GeneratedFrames generateFrames(
            OrtEnvironment env,
            MossSessions sessions,
            JsonNode manifest,
            JsonNode ttsMeta,
            JsonNode codecMeta,
            int[][] rows,
            FrameLimit frameLimit,
            Random random,
            InferenceRequest request,
            MossTtsProgressListener progress,
            int progressEveryFrames,
            MossTtsPcmChunkListener pcmChunkListener) throws OrtException {
        int maxFrames = frameLimit.frames();
        TtsConfig config = TtsConfig.from(manifest, ttsMeta);
        List<String> decodePastInputs = stringList(ttsMeta.path("onnx").path("decode_input_names"), 2);
        List<String> decodePresentOutputs = stringList(ttsMeta.path("onnx").path("decode_output_names"), 1);
        if (decodePastInputs.size() != decodePresentOutputs.size()) {
            throw new IllegalStateException("MOSS TTS decode past/output metadata is inconsistent.");
        }

        OrtSession.Result currentPast = null;
        OnnxTensor globalHidden = null;
        CodecStreamingDecodeState livePcmState = null;
        int streamedPcmChunks = 0;
        try {
            if (shouldStreamPcmDuringGeneration(request, pcmChunkListener)) {
                try {
                    livePcmState = CodecStreamingDecodeState.create(env, sessions.codecDecodeStep(), codecMeta);
                    progress.onProgress(new MossTtsProgress("live_audio", 0, maxFrames, "TTS live audio ready"));
                } catch (Exception e) {
                    livePcmState = null;
                    progress.onProgress(new MossTtsProgress("live_audio_disabled", 0, maxFrames,
                            "TTS live audio preview unavailable"));
                }
            }
            currentPast = runPrefill(env, sessions.prefill, rows);
            globalHidden = extractLastHidden(env, tensor(currentPast, "global_hidden"));
            int pastValidLength = rows.length;

            List<int[]> frames = new ArrayList<>();
            List<Set<Integer>> previousTokens = new ArrayList<>();
            for (int i = 0; i < config.nVq(); i++) {
                previousTokens.add(new HashSet<>());
            }

            for (int step = 0; step < maxFrames; step++) {
                LocalFrame frame = runLocalFixedSampledFrame(
                        env, sessions.localFixedSampledFrame, globalHidden, previousTokens, config, random);
                if (!frame.shouldContinue()) {
                    progress.onProgress(new MossTtsProgress(
                            "generate",
                            frames.size(),
                            maxFrames,
                            "TTS generated " + frames.size() + "/" + maxFrames + " frame(s)"));
                    return new GeneratedFrames(frames, "stop_token", streamedPcmChunks);
                }
                int[] row = frame.tokenIds();
                frames.add(row);
                if (livePcmState != null) {
                    try {
                        DecodedAudio chunk = livePcmState.runFrames(List.of(row));
                        if (chunk.samples() > 0) {
                            emitPcmStreamChunk(
                                    chunk,
                                    request,
                                    pcmChunkListener,
                                    frames.size() - 1,
                                    maxFrames,
                                    "generation_streaming_decode_step");
                            streamedPcmChunks++;
                        }
                    } catch (Exception e) {
                        closeQuietly(livePcmState);
                        livePcmState = null;
                        progress.onProgress(new MossTtsProgress(
                                "live_audio_disabled",
                                frames.size(),
                                maxFrames,
                                "TTS live audio preview stopped"));
                    }
                }
                if (shouldEmitFrameProgress(frames.size(), progressEveryFrames, maxFrames)) {
                    progress.onProgress(new MossTtsProgress(
                            "generate",
                            frames.size(),
                            maxFrames,
                            "TTS generated " + frames.size() + "/" + maxFrames + " frame(s)"));
                }
                for (int channel = 0; channel < Math.min(row.length, previousTokens.size()); channel++) {
                    previousTokens.get(channel).add(row[channel]);
                }

                OrtSession.Result nextPast = runDecode(
                        env, sessions.decode, currentPast, decodePastInputs, decodePresentOutputs,
                        row, pastValidLength, config);
                closeQuietly(currentPast);
                currentPast = nextPast;
                closeQuietly(globalHidden);
                globalHidden = extractLastHidden(env, tensor(currentPast, "global_hidden"));
                pastValidLength++;
            }
            return new GeneratedFrames(frames, frameLimit.stopReason(), streamedPcmChunks);
        } finally {
            closeQuietly(livePcmState);
            closeQuietly(globalHidden);
            closeQuietly(currentPast);
        }
    }

    private static boolean shouldStreamPcmDuringGeneration(
            InferenceRequest request,
            MossTtsPcmChunkListener pcmChunkListener) {
        return pcmChunkListener != null
                && pcmChunkListener != NOOP_PCM_CHUNK_LISTENER
                && !falseParameter(
                        request,
                        "tts_low_latency_audio",
                        "tts_low_latency_pcm",
                        "audio_low_latency",
                        "audio_low_latency_pcm",
                        "tts_stream_pcm_generation",
                        "audio_stream_pcm_generation");
    }

    private static boolean shouldEmitFrameProgress(int frameCount, int everyFrames, int maxFrames) {
        return frameCount == 1 || frameCount >= maxFrames || frameCount % Math.max(1, everyFrames) == 0;
    }

    private static OrtSession.Result runPrefill(OrtEnvironment env, OrtSession session, int[][] rows)
            throws OrtException {
        int[] mask = new int[rows.length];
        java.util.Arrays.fill(mask, 1);
        try (OnnxTensor inputIds = OnnxTensor.createTensor(env, new int[][][] { rows });
             OnnxTensor attentionMask = OnnxTensor.createTensor(env, new int[][] { mask })) {
            return session.run(Map.of(
                    "input_ids", inputIds,
                    "attention_mask", attentionMask));
        }
    }

    private static OrtSession.Result runDecode(
            OrtEnvironment env,
            OrtSession session,
            OrtSession.Result previous,
            List<String> pastInputNames,
            List<String> presentOutputNames,
            int[] frame,
            int pastValidLength,
            TtsConfig config) throws OrtException {
        int[] row = new int[config.rowWidth()];
        java.util.Arrays.fill(row, config.audioPadTokenId());
        row[0] = config.audioAssistantSlotTokenId();
        for (int i = 0; i < Math.min(frame.length, config.nVq()); i++) {
            row[i + 1] = frame[i];
        }

        Map<String, OnnxTensorLike> feeds = new LinkedHashMap<>();
        try (OnnxTensor inputIds = OnnxTensor.createTensor(env, new int[][][] { new int[][] { row } });
             OnnxTensor pastLengths = OnnxTensor.createTensor(env, new int[] { pastValidLength })) {
            feeds.put("input_ids", inputIds);
            feeds.put("past_valid_lengths", pastLengths);
            for (int i = 0; i < pastInputNames.size(); i++) {
                feeds.put(pastInputNames.get(i), tensor(previous, presentOutputNames.get(i)));
            }
            return session.run(feeds);
        }
    }

    private static LocalFrame runLocalFixedSampledFrame(
            OrtEnvironment env,
            OrtSession session,
            OnnxTensor globalHidden,
            List<Set<Integer>> previousTokens,
            TtsConfig config,
            Random random) throws OrtException {
        int[][][] repetitionMask = new int[1][config.nVq()][config.audioCodebookSize()];
        for (int channel = 0; channel < Math.min(previousTokens.size(), config.nVq()); channel++) {
            for (int tokenId : previousTokens.get(channel)) {
                if (tokenId >= 0 && tokenId < config.audioCodebookSize()) {
                    repetitionMask[0][channel][tokenId] = 1;
                }
            }
        }
        float[] assistantRandom = new float[] { drawClampedRandomU(random) };
        float[][] audioRandom = new float[1][config.nVq()];
        for (int i = 0; i < config.nVq(); i++) {
            audioRandom[0][i] = drawClampedRandomU(random);
        }

        try (OnnxTensor repetitionTensor = OnnxTensor.createTensor(env, repetitionMask);
             OnnxTensor assistantTensor = OnnxTensor.createTensor(env, assistantRandom);
             OnnxTensor audioTensor = OnnxTensor.createTensor(env, audioRandom)) {
            Map<String, OnnxTensorLike> feeds = new LinkedHashMap<>();
            feeds.put("global_hidden", globalHidden);
            feeds.put("repetition_seen_mask", repetitionTensor);
            feeds.put("assistant_random_u", assistantTensor);
            feeds.put("audio_random_u", audioTensor);
            try (OrtSession.Result result = session.run(feeds)) {
                boolean shouldContinue = firstScalarAsInt(tensor(result, "should_continue").getValue()) > 0;
                int[] frame = readIntVector(tensor(result, "frame_token_ids"), config.nVq());
                return new LocalFrame(shouldContinue, frame);
            }
        }
    }

    private static DecodedAudio decodeFullAudio(
            OrtEnvironment env,
            OrtSession session,
            JsonNode codecMeta,
            List<int[]> frames) throws OrtException {
        int nVq = codecMeta.path("codec_config").path("num_quantizers").asInt(16);
        int[][][] codes = new int[1][frames.size()][nVq];
        for (int frame = 0; frame < frames.size(); frame++) {
            int[] src = frames.get(frame);
            for (int channel = 0; channel < Math.min(nVq, src.length); channel++) {
                codes[0][frame][channel] = src[channel];
            }
        }
        try (OnnxTensor audioCodes = OnnxTensor.createTensor(env, codes);
             OnnxTensor audioCodeLengths = OnnxTensor.createTensor(env, new int[] { frames.size() });
             OrtSession.Result result = session.run(Map.of(
                     "audio_codes", audioCodes,
                     "audio_code_lengths", audioCodeLengths))) {
            OnnxTensor audio = tensor(result, "audio");
            OnnxTensor audioLengths = tensor(result, "audio_lengths");
            long[] shape = audio.getInfo().getShape();
            if (shape.length != 3 || shape[0] != 1) {
                throw new IllegalStateException("Unexpected MOSS codec audio shape: "
                        + java.util.Arrays.toString(shape));
            }
            int channels = Math.toIntExact(shape[1]);
            int totalSamples = Math.toIntExact(shape[2]);
            int samples = Math.min(totalSamples, firstScalarAsInt(audioLengths.getValue()));
            float[] flat = readFloatVector(audio);
            if (samples < totalSamples) {
                float[] compact = new float[channels * samples];
                for (int channel = 0; channel < channels; channel++) {
                    System.arraycopy(flat, channel * totalSamples, compact, channel * samples, samples);
                }
                flat = compact;
            }
            int sampleRate = codecMeta.path("codec_config").path("sample_rate").asInt(48_000);
            return new DecodedAudio(flat, channels, samples, sampleRate, channels, "full_decode", "native");
        }
    }

    private static DecodedAudio decodeAudio(
            OrtEnvironment env,
            MossSessions sessions,
            JsonNode codecMeta,
            List<int[]> frames,
            InferenceRequest request) throws Exception {
        return decodeAudio(env, sessions, codecMeta, frames, request, NOOP_PCM_CHUNK_LISTENER);
    }

    private static DecodedAudio decodeAudio(
            OrtEnvironment env,
            MossSessions sessions,
            JsonNode codecMeta,
            List<int[]> frames,
            InferenceRequest request,
            MossTtsPcmChunkListener pcmChunkListener) throws Exception {
        String mode = requestedCodecDecodeMode(request);
        if ("full".equals(mode)) {
            DecodedAudio audio = decodeFullAudio(env, sessions.codecDecode(), codecMeta, frames);
            emitPcmStreamChunk(audio, request, pcmChunkListener, 0, frames.size(), "full_decode");
            return audio;
        }
        try {
            return decodeStreamingAudio(env, sessions.codecDecodeStep(), codecMeta, frames, request, pcmChunkListener);
        } catch (Exception e) {
            if ("streaming".equals(mode)) {
                throw e;
            }
            DecodedAudio audio = decodeFullAudio(env, sessions.codecDecode(), codecMeta, frames);
            emitPcmStreamChunk(audio, request, pcmChunkListener, 0, frames.size(), "full_decode_fallback");
            return audio;
        }
    }

    private static String requestedCodecDecodeMode(InferenceRequest request) {
        String value = stringParameter(request, "tts_codec_decode", "codec_decode", "audio_decode");
        if (value == null || value.isBlank()) {
            return "auto";
        }
        return switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "full", "full_decode", "decode_full" -> "full";
            case "stream", "streaming", "step", "decode_step" -> "streaming";
            default -> "auto";
        };
    }

    private static DecodedAudio decodeStreamingAudio(
            OrtEnvironment env,
            OrtSession session,
            JsonNode codecMeta,
            List<int[]> frames) throws OrtException {
        return decodeStreamingAudio(env, session, codecMeta, frames, null, NOOP_PCM_CHUNK_LISTENER);
    }

    private static DecodedAudio decodeStreamingAudio(
            OrtEnvironment env,
            OrtSession session,
            JsonNode codecMeta,
            List<int[]> frames,
            InferenceRequest request,
            MossTtsPcmChunkListener pcmChunkListener) throws OrtException {
        int channels = codecMeta.path("codec_config").path("channels").asInt(2);
        int sampleRate = codecMeta.path("codec_config").path("sample_rate").asInt(48_000);
        List<List<float[]>> chunksByChannel = new ArrayList<>();
        for (int channel = 0; channel < channels; channel++) {
            chunksByChannel.add(new ArrayList<>());
        }
        int totalSamples = 0;

        try (CodecStreamingDecodeState state = CodecStreamingDecodeState.create(env, session, codecMeta)) {
            for (int frameIndex = 0; frameIndex < frames.size(); frameIndex++) {
                DecodedAudio chunk = state.runFrames(List.of(frames.get(frameIndex)));
                if (chunk.samples() <= 0) {
                    continue;
                }
                int chunkChannels = Math.min(channels, chunk.channels());
                emitPcmStreamChunk(
                        chunk,
                        request,
                        pcmChunkListener,
                        frameIndex,
                        frames.size(),
                        "streaming_decode_step");
                for (int channel = 0; channel < chunkChannels; channel++) {
                    float[] copy = new float[chunk.samples()];
                    System.arraycopy(chunk.channelMajorAudio(), channel * chunk.samples(), copy, 0, chunk.samples());
                    chunksByChannel.get(channel).add(copy);
                }
                totalSamples += chunk.samples();
            }
        }

        if (totalSamples <= 0) {
            throw new IllegalStateException("MOSS codec streaming decode produced no audio samples.");
        }

        float[] merged = new float[channels * totalSamples];
        for (int channel = 0; channel < channels; channel++) {
            int offset = channel * totalSamples;
            int cursor = 0;
            for (float[] chunk : chunksByChannel.get(channel)) {
                System.arraycopy(chunk, 0, merged, offset + cursor, chunk.length);
                cursor += chunk.length;
            }
        }
        return new DecodedAudio(merged, channels, totalSamples, sampleRate, channels, "streaming_decode_step", "native");
    }

    private static void emitPcmStreamChunk(
            DecodedAudio chunk,
            InferenceRequest request,
            MossTtsPcmChunkListener listener,
            int frameIndex,
            int totalFrames,
            String decodeMode) {
        if (listener == null || listener == NOOP_PCM_CHUNK_LISTENER || chunk == null || chunk.samples() <= 0) {
            return;
        }
        DecodedAudio normalized = request == null ? chunk : normalizeDecodedAudioChannels(chunk, request);
        PcmAudio pcm = PcmAudio.fromChannelMajorFloat(
                normalized.channelMajorAudio(),
                normalized.channels(),
                normalized.samples(),
                normalized.sampleRate(),
                Map.of());
        listener.onPcmChunk(new MossTtsPcmChunk(
                pcm.data(),
                normalized.sampleRate(),
                normalized.channels(),
                normalized.samples(),
                normalized.sourceChannels(),
                frameIndex,
                totalFrames,
                decodeMode,
                normalized.channelMode()));
    }

    private static DecodedAudio normalizeDecodedAudioChannels(DecodedAudio audio, InferenceRequest request) {
        if (audio.channels() <= 1) {
            return audio;
        }

        String requested = requestedAudioChannelMode(request);
        if ("native".equals(requested)) {
            return audio;
        }
        boolean nearDuplicateChannels = channelsAreNearDuplicates(audio);
        if ("stereo".equals(requested)) {
            return nearDuplicateChannels ? downmixToDualMono(audio, "stereo_dual_mono") : audio;
        }
        if ("mono".equals(requested) || nearDuplicateChannels) {
            return downmixToMono(audio, "mono".equals(requested) ? "mono_requested" : "mono_collapsed_duplicate");
        }
        return audio;
    }

    private static String requestedAudioChannelMode(InferenceRequest request) {
        String value = stringParameter(request, "audio_channels", "tts_audio_channels", "channel_mode");
        if (value == null || value.isBlank()) {
            return "auto";
        }
        return switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "1", "mono", "single" -> "mono";
            case "2", "stereo", "dual-mono", "dual_mono" -> "stereo";
            case "native", "source", "raw" -> "native";
            default -> "auto";
        };
    }

    private static boolean channelsAreNearDuplicates(DecodedAudio audio) {
        int samples = audio.samples();
        int channels = audio.channels();
        if (samples <= 0 || channels <= 1) {
            return false;
        }
        float[] data = audio.channelMajorAudio();
        double referenceEnergy = 0.0;
        double diffEnergy = 0.0;
        for (int sample = 0; sample < samples; sample++) {
            double reference = data[sample];
            referenceEnergy += reference * reference;
            for (int channel = 1; channel < channels; channel++) {
                double diff = reference - data[channel * samples + sample];
                diffEnergy += diff * diff;
            }
        }
        double referenceRms = Math.sqrt(referenceEnergy / samples);
        double diffRms = Math.sqrt(diffEnergy / Math.max(1, samples * (channels - 1)));
        return referenceRms > 1.0e-6 && diffRms / referenceRms < 0.01;
    }

    private static DecodedAudio downmixToMono(DecodedAudio audio, String mode) {
        int samples = audio.samples();
        int channels = audio.channels();
        float[] source = audio.channelMajorAudio();
        float[] mono = new float[samples];
        for (int sample = 0; sample < samples; sample++) {
            double sum = 0.0;
            for (int channel = 0; channel < channels; channel++) {
                sum += source[channel * samples + sample];
            }
            mono[sample] = (float) (sum / channels);
        }
        return new DecodedAudio(mono, 1, samples, audio.sampleRate(), audio.sourceChannels(),
                audio.codecDecodeMode(), mode);
    }

    private static DecodedAudio downmixToDualMono(DecodedAudio audio, String mode) {
        DecodedAudio mono = downmixToMono(audio, mode);
        int samples = mono.samples();
        float[] source = mono.channelMajorAudio();
        float[] stereo = new float[samples * 2];
        System.arraycopy(source, 0, stereo, 0, samples);
        System.arraycopy(source, 0, stereo, samples, samples);
        return new DecodedAudio(stereo, 2, samples, mono.sampleRate(), audio.sourceChannels(),
                audio.codecDecodeMode(), mode);
    }

    private static OnnxTensor extractLastHidden(OrtEnvironment env, OnnxTensor hiddenStatesTensor) throws OrtException {
        long[] dims = hiddenStatesTensor.getInfo().getShape();
        float[] flat = readFloatVector(hiddenStatesTensor);
        if (dims.length == 2) {
            int hiddenSize = Math.toIntExact(dims[1]);
            float[][] hidden = new float[1][hiddenSize];
            System.arraycopy(flat, 0, hidden[0], 0, hiddenSize);
            return OnnxTensor.createTensor(env, hidden);
        }
        if (dims.length != 3 || dims[0] != 1) {
            throw new IllegalStateException("Unexpected global_hidden shape: "
                    + java.util.Arrays.toString(dims));
        }
        int seqLen = Math.toIntExact(dims[1]);
        int hiddenSize = Math.toIntExact(dims[2]);
        int start = (seqLen - 1) * hiddenSize;
        float[][] hidden = new float[1][hiddenSize];
        System.arraycopy(flat, start, hidden[0], 0, hiddenSize);
        return OnnxTensor.createTensor(env, hidden);
    }

    private static int[][] buildVoiceCloneRows(JsonNode manifest, int[][] promptAudioCodes, int[] textTokenIds) {
        TtsConfig config = TtsConfig.from(manifest, null);
        List<int[]> rows = new ArrayList<>();

        List<Integer> prefix = new ArrayList<>(intList(manifest.path("prompt_templates")
                .path("user_prompt_prefix_token_ids")));
        prefix.add(config.audioStartTokenId());
        for (int tokenId : prefix) {
            rows.add(textRow(tokenId, config));
        }

        for (int[] codeRow : promptAudioCodes) {
            int[] row = textRow(config.audioUserSlotTokenId(), config);
            for (int i = 0; i < Math.min(codeRow.length, config.nVq()); i++) {
                row[i + 1] = codeRow[i];
            }
            rows.add(row);
        }

        rows.add(textRow(config.audioEndTokenId(), config));
        for (int tokenId : intList(manifest.path("prompt_templates").path("user_prompt_after_reference_token_ids"))) {
            rows.add(textRow(tokenId, config));
        }
        for (int tokenId : textTokenIds) {
            rows.add(textRow(tokenId, config));
        }
        for (int tokenId : intList(manifest.path("prompt_templates").path("assistant_prompt_prefix_token_ids"))) {
            rows.add(textRow(tokenId, config));
        }
        rows.add(textRow(config.audioStartTokenId(), config));
        return rows.toArray(int[][]::new);
    }

    private static int[] textRow(int tokenId, TtsConfig config) {
        int[] row = new int[config.rowWidth()];
        java.util.Arrays.fill(row, config.audioPadTokenId());
        row[0] = tokenId;
        return row;
    }

    private static VoicePrompt resolveVoicePrompt(JsonNode manifest, InferenceRequest request, String prompt) {
        String requestedVoice = stringParameter(request, "voice");
        JsonNode voices = manifest.path("builtin_voices");
        JsonNode selected = null;
        String mode = "default";
        String language = "";
        List<String> ambiguousLabels = new ArrayList<>();
        if (voices.isArray()) {
            VoiceSelector selector = voiceSelector(requestedVoice);
            if (requestedVoice == null || isAutoVoiceRequest(requestedVoice)) {
                language = inferPromptLanguage(prompt);
                selected = chooseAutoVoice(voices, language);
                if (selected == null) {
                    selected = voices.isEmpty() ? null : voices.get(0);
                    mode = "default";
                } else {
                    mode = "auto";
                }
            } else if (isDefaultVoiceRequest(requestedVoice)) {
                selected = voices.isEmpty() ? null : voices.get(0);
                mode = "default";
            } else if (selector != null && selector.isLanguageOnly()) {
                language = selector.language();
                selected = chooseAutoVoice(voices, language);
                mode = "language";
            } else if (selector != null) {
                language = selector.language();
                selected = chooseVoiceBySelector(voices, selector);
                mode = "selector";
                if (selected == null) {
                    throw new IllegalArgumentException(unavailableVoiceSelectorMessage(requestedVoice, voices, selector));
                }
            } else {
                mode = "explicit";
                VoiceMatch best = null;
                for (JsonNode voice : voices) {
                    VoiceMatch match = voiceMatch(voice, requestedVoice);
                    if (!match.matches()) {
                        continue;
                    }
                    if (best == null || match.score() > best.score()) {
                        best = match;
                        selected = voice;
                        ambiguousLabels.clear();
                        ambiguousLabels.add(voiceChoiceLabel(voice));
                    } else if (match.score() == best.score()) {
                        ambiguousLabels.add(voiceChoiceLabel(voice));
                    }
                }
                if (ambiguousLabels.size() > 1) {
                    throw new IllegalArgumentException("MOSS TTS voice preset is ambiguous: " + requestedVoice
                            + ". Matches: " + String.join(", ", ambiguousLabels));
                }
            }
        }
        if (requestedVoice != null && selected == null) {
            throw new IllegalArgumentException(unavailableVoiceMessage(requestedVoice, voices));
        }
        if (selected == null || !selected.path("prompt_audio_codes").isArray()) {
            throw new IllegalStateException("MOSS TTS manifest does not include a usable builtin voice.");
        }
        List<int[]> rows = new ArrayList<>();
        for (JsonNode frame : selected.path("prompt_audio_codes")) {
            rows.add(intList(frame).stream().mapToInt(Integer::intValue).toArray());
        }
        if (rows.isEmpty()) {
            throw new IllegalStateException("Selected MOSS TTS voice has no prompt audio codes.");
        }
        return new VoicePrompt(voiceLabel(selected), voiceId(selected), mode, language, rows.toArray(int[][]::new));
    }

    private static boolean isAutoVoiceRequest(String requestedVoice) {
        return normalizeVoiceText(requestedVoice).equals("auto");
    }

    private static boolean isDefaultVoiceRequest(String requestedVoice) {
        String normalized = normalizeVoiceText(requestedVoice);
        return normalized.equals("default") || normalized.equals("first");
    }

    private static VoiceSelector voiceSelector(String requestedVoice) {
        VoiceSelectorParse parse = voiceSelectorParse(requestedVoice);
        if (parse.language() == null || !parse.unknownTerms().isEmpty()) {
            return null;
        }
        return new VoiceSelector(parse.language(), parse.gender());
    }

    private static VoiceSelectorParse voiceSelectorParse(String requestedVoice) {
        String normalized = normalizeVoiceText(requestedVoice);
        if (normalized.isBlank()) {
            return new VoiceSelectorParse(null, null, List.of());
        }

        String language = null;
        String gender = null;
        List<String> unknownTerms = new ArrayList<>();
        for (String token : normalized.split(" ")) {
            String tokenLanguage = languageAlias(token);
            if (tokenLanguage != null) {
                language = tokenLanguage;
                continue;
            }
            String tokenGender = genderAlias(token);
            if (tokenGender != null) {
                gender = tokenGender;
                continue;
            }
            unknownTerms.add(token);
        }

        return new VoiceSelectorParse(language, gender, List.copyOf(unknownTerms));
    }

    private static String languageAlias(String value) {
        return switch (value) {
            case "en", "eng", "english" -> "english";
            case "jp", "ja", "jpn", "japanese" -> "japanese";
            case "cn", "zh", "zho", "chi", "chinese", "mandarin" -> "chinese";
            default -> null;
        };
    }

    private static String genderAlias(String value) {
        return switch (value) {
            case "f", "female", "woman", "women", "girl" -> "female";
            case "m", "male", "man", "men", "boy" -> "male";
            default -> null;
        };
    }

    private static VoiceMatch voiceMatch(JsonNode voice, String requestedVoice) {
        String needle = normalizeVoiceText(requestedVoice);
        if (needle.isBlank()) {
            return VoiceMatch.none();
        }

        int best = 0;
        for (String field : List.of("voice", "id", "display_name", "name", "group")) {
            String raw = voice.path(field).asText("");
            if (raw.isBlank()) {
                continue;
            }
            String value = normalizeVoiceText(raw);
            if (value.equals(needle)) {
                best = Math.max(best, field.equals("group") ? 80 : 100);
            } else if (value.startsWith(needle)) {
                best = Math.max(best, field.equals("group") ? 45 : 60);
            } else if (value.contains(needle)) {
                best = Math.max(best, field.equals("group") ? 35 : 50);
            }
        }
        return best > 0 ? new VoiceMatch(best) : VoiceMatch.none();
    }

    private static String normalizeVoiceText(String value) {
        if (value == null) {
            return "";
        }
        return value.trim()
                .toLowerCase(Locale.ROOT)
                .replaceAll("[^\\p{L}\\p{N}]+", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static JsonNode chooseAutoVoice(JsonNode voices, String language) {
        if (voices == null || !voices.isArray() || voices.isEmpty() || language == null || language.isBlank()) {
            return null;
        }

        JsonNode selected = null;
        int bestScore = 0;
        for (JsonNode voice : voices) {
            int score = autoVoiceScore(voice, language);
            if (score > bestScore) {
                bestScore = score;
                selected = voice;
            }
        }
        return bestScore > 0 ? selected : null;
    }

    private static JsonNode chooseVoiceBySelector(JsonNode voices, VoiceSelector selector) {
        if (voices == null || !voices.isArray() || voices.isEmpty() || selector == null) {
            return null;
        }

        for (JsonNode voice : voices) {
            String group = normalizeVoiceText(voice.path("group").asText(""));
            String display = normalizeVoiceText(voice.path("display_name").asText(""));
            if (!voiceMatchesLanguage(voice, selector.language(), group, display)) {
                continue;
            }
            if (selector.gender() == null || normalizedTextHasToken(group, selector.gender())) {
                return voice;
            }
        }

        return null;
    }

    private static boolean normalizedTextHasToken(String normalizedText, String token) {
        if (normalizedText == null || normalizedText.isBlank() || token == null || token.isBlank()) {
            return false;
        }
        for (String part : normalizedText.split(" ")) {
            if (part.equals(token)) {
                return true;
            }
        }
        return false;
    }

    private static boolean voiceMatchesLanguage(JsonNode voice, String language, String group, String display) {
        String id = normalizeVoiceText(voiceId(voice));
        String prefix = switch (language) {
            case "english" -> "en";
            case "japanese" -> "jp";
            case "chinese" -> "cn";
            default -> "";
        };
        return group.contains(language)
                || id.contains(language)
                || display.contains(language)
                || (!prefix.isBlank() && (display.equals(prefix) || display.startsWith(prefix + " ")));
    }

    private static int autoVoiceScore(JsonNode voice, String language) {
        String group = normalizeVoiceText(voice.path("group").asText(""));
        String display = normalizeVoiceText(voice.path("display_name").asText(""));
        String id = normalizeVoiceText(voiceId(voice));
        String prefix = switch (language) {
            case "english" -> "en";
            case "japanese" -> "jp";
            case "chinese" -> "cn";
            default -> "";
        };

        int score = 0;
        if (group.contains(language)) {
            score = Math.max(score, 100);
        }
        if (!prefix.isBlank() && (display.equals(prefix) || display.startsWith(prefix + " "))) {
            score = Math.max(score, 90);
        }
        if (display.contains(language) || id.contains(language)) {
            score = Math.max(score, 80);
        }
        if ("english".equals(language) && group.contains("female")) {
            score += 10;
        }
        return score;
    }

    private static String inferPromptLanguage(String prompt) {
        if (prompt == null || prompt.isBlank()) {
            return "";
        }

        int latin = 0;
        int han = 0;
        int kana = 0;
        for (int i = 0; i < prompt.length(); ) {
            int cp = prompt.codePointAt(i);
            i += Character.charCount(cp);
            Character.UnicodeScript script = Character.UnicodeScript.of(cp);
            if (script == Character.UnicodeScript.HIRAGANA || script == Character.UnicodeScript.KATAKANA) {
                kana++;
            } else if (script == Character.UnicodeScript.HAN) {
                han++;
            } else if (script == Character.UnicodeScript.LATIN && Character.isLetter(cp)) {
                latin++;
            }
        }

        if (kana > 0) {
            return "japanese";
        }
        if (han > 0 && han >= latin) {
            return "chinese";
        }
        if (latin > 0) {
            return "english";
        }
        return "";
    }

    private static String voiceLabel(JsonNode voice) {
        for (String field : List.of("display_name", "voice", "id", "name")) {
            String value = voice.path(field).asText("");
            if (!value.isBlank()) {
                return value;
            }
        }
        return "default";
    }

    private static String voiceId(JsonNode voice) {
        for (String field : List.of("voice", "id", "name", "display_name")) {
            String value = voice.path(field).asText("");
            if (!value.isBlank()) {
                return value;
            }
        }
        return "default";
    }

    private static String voiceChoiceLabel(JsonNode voice) {
        String id = voiceId(voice);
        String label = voiceLabel(voice);
        return id.equals(label) ? id : id + " (" + label + ")";
    }

    private static String availableVoiceLabels(JsonNode voices) {
        if (voices == null || !voices.isArray() || voices.isEmpty()) {
            return "(none)";
        }
        List<String> labels = new ArrayList<>();
        for (JsonNode voice : voices) {
            labels.add(voiceChoiceLabel(voice));
        }
        return String.join(", ", labels);
    }

    private static String unavailableVoiceMessage(String requestedVoice, JsonNode voices) {
        VoiceSelectorParse parse = voiceSelectorParse(requestedVoice);
        if (parse.language() != null && !parse.unknownTerms().isEmpty()) {
            List<String> languageVoices = availableVoiceLabelsForLanguage(voices, parse.language());
            String unknown = String.join(", ", parse.unknownTerms());
            String suggestion = selectorFallbackSuggestion(voices, new VoiceSelector(parse.language(), null));
            return "MOSS TTS voice preset not found: " + requestedVoice
                    + ". Unknown selector term" + (parse.unknownTerms().size() == 1 ? "" : "s")
                    + " for " + parse.language() + ": " + unknown
                    + ". Available " + parse.language() + " voices: "
                    + (languageVoices.isEmpty() ? "(none)" : String.join(", ", languageVoices))
                    + (suggestion.isBlank() ? "" : ". Try " + suggestion);
        }

        return "MOSS TTS voice preset not found: " + requestedVoice
                + ". Available voices: " + availableVoiceLabels(voices);
    }

    private static String unavailableVoiceSelectorMessage(
            String requestedVoice,
            JsonNode voices,
            VoiceSelector selector) {
        List<String> languageVoices = availableVoiceLabelsForLanguage(voices, selector.language());
        String descriptor = selector.gender() == null || selector.gender().isBlank()
                ? selector.language()
                : selector.language() + " " + selector.gender();
        if (languageVoices.isEmpty()) {
            return "MOSS TTS voice selector not available: " + requestedVoice
                    + ". No " + selector.language() + " voice presets are available. Available voices: "
                    + availableVoiceLabels(voices);
        }
        String suggestion = selectorFallbackSuggestion(voices, selector);
        return "MOSS TTS voice selector not available: " + requestedVoice
                + ". No " + descriptor + " voice preset is available. Available "
                + selector.language() + " voices: " + String.join(", ", languageVoices)
                + (suggestion.isBlank() ? "" : ". Try " + suggestion);
    }

    private static List<String> availableVoiceLabelsForLanguage(JsonNode voices, String language) {
        List<String> labels = new ArrayList<>();
        if (voices == null || !voices.isArray() || language == null || language.isBlank()) {
            return labels;
        }
        for (JsonNode voice : voices) {
            String group = normalizeVoiceText(voice.path("group").asText(""));
            String display = normalizeVoiceText(voice.path("display_name").asText(""));
            if (voiceMatchesLanguage(voice, language, group, display)) {
                labels.add(voiceChoiceLabel(voice));
            }
        }
        return labels;
    }

    private static String selectorFallbackSuggestion(JsonNode voices, VoiceSelector selector) {
        if (voices == null || !voices.isArray() || selector == null) {
            return "";
        }

        String languageAlias = cliLanguageAlias(selector.language());
        String firstVoiceId = "";
        String firstGender = "";
        for (JsonNode voice : voices) {
            String group = normalizeVoiceText(voice.path("group").asText(""));
            String display = normalizeVoiceText(voice.path("display_name").asText(""));
            if (!voiceMatchesLanguage(voice, selector.language(), group, display)) {
                continue;
            }
            if (firstVoiceId.isBlank()) {
                firstVoiceId = voiceId(voice);
            }
            if (firstGender.isBlank()) {
                if (normalizedTextHasToken(group, "female")) {
                    firstGender = "female";
                } else if (normalizedTextHasToken(group, "male")) {
                    firstGender = "male";
                }
            }
        }

        List<String> suggestions = new ArrayList<>();
        if (!languageAlias.isBlank() && !firstGender.isBlank()) {
            suggestions.add("--voice " + languageAlias + " " + firstGender);
        }
        if (!firstVoiceId.isBlank()) {
            suggestions.add("--voice " + firstVoiceId);
        }
        return String.join(" or ", suggestions);
    }

    private static String cliLanguageAlias(String language) {
        return switch (language == null ? "" : language) {
            case "english" -> "en";
            case "japanese" -> "jp";
            case "chinese" -> "zh";
            default -> "";
        };
    }

    private record VoiceMatch(int score) {
        private boolean matches() {
            return score > 0;
        }

        private static VoiceMatch none() {
            return new VoiceMatch(0);
        }
    }

    private static FrameLimit resolveFrameLimit(JsonNode manifest, JsonNode codecMeta, InferenceRequest request) {
        int manifestMax = manifest.path("generation_defaults").path("max_new_frames").asInt(375);
        Integer explicitFrames = intParameter(request, "tts_max_frames", "max_frames");
        if (explicitFrames != null) {
            return new FrameLimit(clampFrames(explicitFrames, manifestMax), "tts_max_frames", "max_frames");
        }

        Double explicitSeconds = doubleParameter(
                request,
                "tts_max_seconds",
                "max_seconds",
                "tts_seconds",
                "duration_seconds",
                "duration");
        if (explicitSeconds != null) {
            double seconds = Math.max(0.001, explicitSeconds);
            int frames = (int) Math.ceil(seconds * codecFrameRate(codecMeta));
            return new FrameLimit(clampFrames(frames, manifestMax), "tts_max_seconds", "max_seconds");
        }

        int maxTokens = request.getMaxTokens();
        if (maxTokens > 0) {
            return new FrameLimit(clampFrames(maxTokens, manifestMax), "max_tokens", "max_tokens");
        }
        return new FrameLimit(manifestMax, "manifest_default", "manifest_limit");
    }

    private static int requestedStreamProgressFrames(InferenceRequest request, int maxFrames) {
        Integer explicit = intParameter(
                request,
                "tts_stream_progress_frames",
                "stream_progress_frames",
                "audio_stream_progress_frames");
        if (explicit != null) {
            return Math.max(1, explicit);
        }
        return Math.max(4, (int) Math.ceil(Math.max(1, maxFrames) / 40.0));
    }

    private static int clampFrames(int frames, int manifestMax) {
        return Math.max(1, Math.min(manifestMax, frames));
    }

    private static double codecFrameRate(JsonNode codecMeta) {
        JsonNode codecConfig = codecMeta == null ? null : codecMeta.path("codec_config");
        double sampleRate = codecConfig == null ? 48_000.0 : codecConfig.path("sample_rate").asDouble(48_000.0);
        double downsampleRate = codecConfig == null ? 3_840.0 : codecConfig.path("downsample_rate").asDouble(3_840.0);
        if (sampleRate <= 0.0 || downsampleRate <= 0.0) {
            return 12.5;
        }
        return sampleRate / downsampleRate;
    }

    private static AudioProcessingOptions audioProcessingOptions(InferenceRequest request) {
        if (falseParameter(request, "audio_polish", "tts_audio_polish")
                || booleanParameter(request, "disable_audio_polish", "no_audio_polish")) {
            return AudioProcessingOptions.none();
        }

        Double fadeMs = doubleParameter(request, "audio_fade_ms", "tts_fade_ms");
        Double fadeInMs = doubleParameter(request, "audio_fade_in_ms", "tts_fade_in_ms");
        Double fadeOutMs = doubleParameter(request, "audio_fade_out_ms", "tts_fade_out_ms");
        Double gainDb = doubleParameter(request, "audio_gain_db", "tts_gain_db", "gain_db");
        Double peakDb = doubleParameter(request, "audio_peak_db", "audio_peak_dbfs", "tts_peak_db", "tts_peak_dbfs");
        Double maxNormalizeGainDb = doubleParameter(
                request,
                "audio_max_normalize_gain_db",
                "tts_max_normalize_gain_db");
        Double trimThresholdDb = doubleParameter(
                request,
                "audio_trim_threshold_dbfs",
                "audio_trim_dbfs",
                "audio_trim_db",
                "tts_trim_db");
        Double trimPaddingMs = doubleParameter(
                request,
                "audio_trim_padding_ms",
                "tts_trim_padding_ms");

        AudioProcessingOptions.Builder builder = AudioProcessingOptions.builder()
                .removeDcOffset(!falseParameter(request, "audio_remove_dc", "tts_remove_dc"))
                .fadeInSeconds(((fadeInMs != null ? fadeInMs : (fadeMs != null ? fadeMs : 3.0))) / 1000.0)
                .fadeOutSeconds(((fadeOutMs != null ? fadeOutMs : (fadeMs != null ? fadeMs : 12.0))) / 1000.0)
                .gainDb(gainDb == null ? 0.0 : gainDb)
                .maxNormalizeGainDb(maxNormalizeGainDb == null ? 9.0 : Math.max(0.0, maxNormalizeGainDb))
                .trimSilence(booleanParameter(request, "audio_trim_silence", "tts_trim_silence"))
                .trimSilenceThresholdDbfs(trimThresholdDb == null ? -48.0 : Math.min(-0.1, trimThresholdDb))
                .trimSilencePaddingSeconds((trimPaddingMs == null ? 25.0 : Math.max(0.0, trimPaddingMs)) / 1000.0);

        if (!falseParameter(request, "audio_normalize", "tts_audio_normalize")) {
            builder.peakNormalizeDbfs(peakDb == null ? -3.0 : Math.min(0.0, peakDb));
        }
        return builder.build();
    }

    private static void copyAudioProcessingMetadata(PcmAudio pcm, Map<String, Object> metadata) {
        if (pcm == null || pcm.metadata() == null || pcm.metadata().isEmpty()) {
            return;
        }
        for (Map.Entry<String, String> entry : pcm.metadata().entrySet()) {
            if (entry.getKey() != null && entry.getKey().startsWith("audio_processing")) {
                metadata.put(entry.getKey(), entry.getValue());
            }
        }
    }

    private static AudioEncodeOptions audioEncodeOptions(
            InferenceRequest request,
            Map<String, String> audioMetadata) {
        String format = requestedAudioFormat(request);
        Integer compression = intParameter(
                request,
                "flac_compression",
                "flac_compression_level",
                "audio_compression",
                "compression_level");
        Integer bitrate = intParameter(
                request,
                "audio_bitrate_kbps",
                "bitrate_kbps",
                "bitrate");
        return AudioEncodeOptions.builder()
                .format(format)
                .compressionLevel(compression == null ? 5 : Math.max(0, Math.min(8, compression)))
                .bitrateKbps(bitrate == null ? 192 : Math.max(1, bitrate))
                .verify(booleanParameter(request, "audio_verify", "flac_verify"))
                .metadata(audioMetadata)
                .build();
    }

    private static String requestedAudioFormat(InferenceRequest request) {
        String explicit = stringParameter(request, "audio_format", "output_format", "audio_ext");
        if (explicit != null && !explicit.isBlank()) {
            return explicit;
        }
        String outputPath = stringParameter(request, "output_path");
        String ext = pathExtension(outputPath);
        return isKnownAudioExtension(ext) ? ext : "wav";
    }

    private static boolean booleanParameter(InferenceRequest request, String... keys) {
        if (request.getParameters() == null) {
            return false;
        }
        for (String key : keys) {
            Object value = request.getParameters().get(key);
            if (value instanceof Boolean bool) {
                return bool;
            }
            if (value instanceof String text && !text.isBlank()) {
                return Boolean.parseBoolean(text.trim());
            }
        }
        return false;
    }

    private static boolean falseParameter(InferenceRequest request, String... keys) {
        if (request.getParameters() == null) {
            return false;
        }
        for (String key : keys) {
            Object value = request.getParameters().get(key);
            if (value instanceof Boolean bool) {
                return !bool;
            }
            if (value instanceof Number number) {
                return number.intValue() == 0;
            }
            if (value instanceof String text && !text.isBlank()) {
                String normalized = text.trim().toLowerCase(Locale.ROOT);
                if (Set.of("false", "0", "no", "off", "disabled").contains(normalized)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static String pathExtension(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        String fileName = Path.of(value).getFileName() == null ? value : Path.of(value).getFileName().toString();
        int dot = fileName.lastIndexOf('.');
        return dot >= 0 && dot + 1 < fileName.length()
                ? fileName.substring(dot + 1).toLowerCase(Locale.ROOT)
                : "";
    }

    private static boolean isKnownAudioExtension(String ext) {
        return ext != null
                && Set.of("wav", "flac", "mp3", "opus", "ogg", "m4a", "aac").contains(ext.toLowerCase(Locale.ROOT));
    }

    private static Integer intParameter(InferenceRequest request, String... keys) {
        Double value = doubleParameter(request, keys);
        return value == null ? null : value.intValue();
    }

    private static Double doubleParameter(InferenceRequest request, String... keys) {
        if (request.getParameters() == null) {
            return null;
        }
        for (String key : keys) {
            Object value = request.getParameters().get(key);
            if (value instanceof Number number) {
                return number.doubleValue();
            }
            if (value instanceof String text && !text.isBlank()) {
                try {
                    return Double.parseDouble(text.trim());
                } catch (NumberFormatException ignored) {
                    // Continue to the next alias.
                }
            }
        }
        return null;
    }

    private static ResolvedSeed resolvedSeed(InferenceRequest request) {
        Object seed = request.getParameters() == null ? null : request.getParameters().get("seed");
        if (seed instanceof Number number) {
            return new ResolvedSeed(number.longValue(), "explicit");
        }
        if (seed instanceof String value && !value.isBlank()) {
            try {
                return new ResolvedSeed(Long.parseLong(value.trim()), "explicit");
            } catch (NumberFormatException ignored) {
                return new ResolvedSeed(value.hashCode(), "explicit-string-hash");
            }
        }
        if (booleanParameter(request, "random_seed", "tts_random_seed")) {
            return new ResolvedSeed(ThreadLocalRandom.current().nextLong(), "generated");
        }
        return new ResolvedSeed(1234L, "default");
    }

    private static Map<String, String> wavInfoMetadata(
            Path ttsDir,
            String prompt,
            VoicePrompt voicePrompt,
            ResolvedSeed seed,
            String stopReason,
            Instant createdAt) {
        Map<String, String> info = new LinkedHashMap<>();
        info.put("ISFT", "Gollek MOSS TTS ONNX");
        info.put("IART", voicePrompt.label());
        info.put("IPRD", modelDisplayName(ttsDir));
        info.put("INAM", shortenForMetadata(prompt, 240));
        info.put("ICMT", shortenForMetadata("Prompt: " + prompt
                + " | Voice: " + voicePrompt.label()
                + " | Voice mode: " + voicePrompt.mode()
                + " | Seed: " + seed.value()
                + " | Stop: " + stopReason, 1024));
        info.put("ICRD", createdAt.toString());
        return info;
    }

    private static String modelDisplayName(Path ttsDir) {
        if (ttsDir == null || ttsDir.getFileName() == null) {
            return "MOSS-TTS-Nano-100M-ONNX";
        }
        return ttsDir.getFileName().toString();
    }

    private static String shortenForMetadata(String value, int maxChars) {
        if (value == null || value.isBlank()) {
            return "";
        }
        String normalized = value.strip().replaceAll("\\s+", " ");
        if (normalized.length() <= maxChars) {
            return normalized;
        }
        return normalized.substring(0, Math.max(0, maxChars - 3)).stripTrailing() + "...";
    }

    private static String normalizePrompt(String prompt) {
        String text = prompt == null ? "" : prompt.strip();
        if (text.isEmpty()) {
            throw new IllegalArgumentException("Text prompt cannot be empty for MOSS TTS.");
        }

        if (containsCjk(text)) {
            char last = text.charAt(text.length() - 1);
            if (".!?;:\u3002\uff01\uff1f\uff1b".indexOf(last) < 0) {
                text = text + "\u3002";
            }
            return text;
        }

        if (Character.isLowerCase(text.charAt(0))) {
            text = Character.toUpperCase(text.charAt(0)) + text.substring(1);
        }
        char last = text.charAt(text.length() - 1);
        if (Character.isLetterOrDigit(last)) {
            text = text + ".";
        }
        if (wordCount(text) < 5) {
            text = " " + text;
        }
        return text;
    }

    private static boolean containsCjk(String text) {
        if (text == null || text.isBlank()) {
            return false;
        }
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            Character.UnicodeScript script = Character.UnicodeScript.of(cp);
            if (script == Character.UnicodeScript.HAN
                    || script == Character.UnicodeScript.HIRAGANA
                    || script == Character.UnicodeScript.KATAKANA
                    || script == Character.UnicodeScript.HANGUL) {
                return true;
            }
        }
        return false;
    }

    private static int wordCount(String text) {
        if (text == null || text.isBlank()) {
            return 0;
        }
        int count = 0;
        for (String part : text.strip().split("\\s+")) {
            if (!part.isBlank()) {
                count++;
            }
        }
        return count;
    }

    private static JsonNode readJson(Path path) throws Exception {
        if (!Files.isRegularFile(path)) {
            throw new IllegalStateException("Required MOSS TTS file not found: " + path);
        }
        return JSON.readTree(path.toFile());
    }

    private static void ensureOrtJavaNativeDependency() throws Exception {
        String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (!osName.contains("mac")) {
            return;
        }
        Path target = Path.of(System.getProperty("user.home"), ".gollek", "libs", ORT_JAVA_VERSIONED_MAC_DYLIB);
        if (Files.isRegularFile(target)) {
            return;
        }
        Path gollekNative = target.resolveSibling("libonnxruntime.dylib");
        if (Files.isRegularFile(gollekNative)) {
            try {
                Files.createSymbolicLink(target, gollekNative.getFileName());
                return;
            } catch (Exception ignored) {
                Files.copy(gollekNative, target, StandardCopyOption.REPLACE_EXISTING);
                return;
            }
        }
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        String platform = arch.contains("aarch64") || arch.contains("arm64") ? "osx-aarch64" : "osx-x64";
        String resourcePath = "/ai/onnxruntime/native/" + platform + "/libonnxruntime.dylib";
        try (var in = MossTtsOnnxRunner.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                return;
            }
            Files.createDirectories(target.getParent());
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static List<Integer> intList(JsonNode node) {
        List<Integer> values = new ArrayList<>();
        if (node != null && node.isArray()) {
            for (JsonNode item : node) {
                values.add(item.asInt());
            }
        }
        return values;
    }

    private static List<String> stringList(JsonNode node, int skip) {
        List<String> values = new ArrayList<>();
        if (node != null && node.isArray()) {
            for (int i = skip; i < node.size(); i++) {
                values.add(node.get(i).asText());
            }
        }
        return values;
    }

    private static List<JsonNode> jsonArray(JsonNode node) {
        List<JsonNode> values = new ArrayList<>();
        if (node != null && node.isArray()) {
            for (JsonNode item : node) {
                values.add(item);
            }
        }
        return values;
    }

    private static String stringParameter(InferenceRequest request, String... names) {
        if (request.getParameters() == null) {
            return null;
        }
        for (String name : names) {
            Object value = request.getParameters().get(name);
            if (value instanceof String string && !string.isBlank()) {
                return string.trim();
            }
        }
        return null;
    }

    private static float drawClampedRandomU(Random random) {
        return Math.min(0.99999994f, Math.max(0.0f, random.nextFloat()));
    }

    private static OnnxTensor tensor(OrtSession.Result result, String name) throws OrtException {
        OnnxValue value = result.get(name)
                .orElseThrow(() -> new IllegalStateException("Missing ONNX output: " + name));
        if (value instanceof OnnxTensor tensor) {
            return tensor;
        }
        throw new IllegalStateException("ONNX output is not a tensor: " + name);
    }

    private static float[] readFloatVector(OnnxTensor tensor) {
        TensorInfo info = tensor.getInfo();
        int size = Math.toIntExact(info.getNumElements());
        FloatBuffer buffer = tensor.getFloatBuffer();
        FloatBuffer copy = buffer.duplicate();
        copy.rewind();
        float[] values = new float[size];
        copy.get(values);
        return values;
    }

    private static int[] readIntVector(OnnxTensor tensor, int expectedCount) throws OrtException {
        List<Integer> values = new ArrayList<>(expectedCount);
        collectInts(tensor.getValue(), values, expectedCount);
        if (values.size() < expectedCount) {
            throw new IllegalStateException("Expected " + expectedCount + " int values from "
                    + tensor.getInfo() + ", got " + values.size());
        }
        int[] out = new int[expectedCount];
        for (int i = 0; i < expectedCount; i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static int firstScalarAsInt(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof Boolean bool) {
            return bool ? 1 : 0;
        }
        if (value instanceof int[] array && array.length > 0) {
            return array[0];
        }
        if (value instanceof long[] array && array.length > 0) {
            return Math.toIntExact(array[0]);
        }
        if (value instanceof byte[] array && array.length > 0) {
            return array[0];
        }
        if (value instanceof boolean[] array && array.length > 0) {
            return array[0] ? 1 : 0;
        }
        if (value instanceof Object[] array && array.length > 0) {
            return firstScalarAsInt(array[0]);
        }
        throw new IllegalStateException("Cannot read scalar int from ONNX value: " + value);
    }

    private static void collectInts(Object value, List<Integer> out, int limit) {
        if (out.size() >= limit || value == null) {
            return;
        }
        if (value instanceof Number number) {
            out.add(number.intValue());
            return;
        }
        if (value instanceof Boolean bool) {
            out.add(bool ? 1 : 0);
            return;
        }
        if (value instanceof int[] array) {
            for (int item : array) {
                if (out.size() >= limit) return;
                out.add(item);
            }
            return;
        }
        if (value instanceof long[] array) {
            for (long item : array) {
                if (out.size() >= limit) return;
                out.add(Math.toIntExact(item));
            }
            return;
        }
        if (value instanceof boolean[] array) {
            for (boolean item : array) {
                if (out.size() >= limit) return;
                out.add(item ? 1 : 0);
            }
            return;
        }
        if (value instanceof Object[] array) {
            for (Object item : array) {
                collectInts(item, out, limit);
                if (out.size() >= limit) return;
            }
        }
    }

    private static void closeQuietly(AutoCloseable closeable) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (Exception ignored) {
        }
    }

    private static void configureSessionLogging(OrtSession.SessionOptions options) throws OrtException {
        options.setLoggerId("gollek-moss-tts");
        options.setSessionLogLevel(ortLoggingLevel());
    }

    private static OrtLoggingLevel ortLoggingLevel() {
        return switch (ortLoggingLevelName()) {
            case "verbose" -> OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
            case "info" -> OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            case "warning", "warn" -> OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
            case "fatal" -> OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            default -> OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
        };
    }

    private static String ortLoggingLevelName() {
        String raw = firstNonBlank(
                System.getProperty(ORT_LOG_LEVEL_PROPERTY),
                System.getenv(ORT_LOG_LEVEL_ENV),
                "error");
        String normalized = raw.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "verbose", "info", "warning", "warn", "error", "fatal" -> normalized;
            default -> "error";
        };
    }

    public record MossTtsAudio(
            byte[] bytes,
            String format,
            String mimeType,
            int sampleRate,
            int channels,
            int samples,
            int frames,
            String normalizedPrompt,
            Map<String, Object> metadata) {
    }

    public record MossTtsProgress(String stage, int frames, int maxFrames, String message) {
    }

    public record MossTtsWarmup(long durationMs, Map<String, Object> metadata) {
    }

    @FunctionalInterface
    public interface MossTtsProgressListener {
        void onProgress(MossTtsProgress progress);
    }

    public record MossTtsPcmChunk(
            byte[] pcmS16le,
            int sampleRate,
            int channels,
            int samples,
            int sourceChannels,
            int frameIndex,
            int totalFrames,
            String codecDecodeMode,
            String channelMode) {

        public MossTtsPcmChunk {
            pcmS16le = pcmS16le == null ? new byte[0] : pcmS16le.clone();
        }

        @Override
        public byte[] pcmS16le() {
            return pcmS16le.clone();
        }
    }

    @FunctionalInterface
    public interface MossTtsPcmChunkListener {
        void onPcmChunk(MossTtsPcmChunk chunk);
    }

    private record LocalFrame(boolean shouldContinue, int[] tokenIds) {
    }

    private record GeneratedFrames(List<int[]> frames, String stopReason, int streamedPcmChunks) {
    }

    private record FrameLimit(int frames, String source, String stopReason) {
    }

    private record DecodedAudio(
            float[] channelMajorAudio,
            int channels,
            int samples,
            int sampleRate,
            int sourceChannels,
            String codecDecodeMode,
            String channelMode) {
    }

    private static final class CodecStreamingDecodeState implements AutoCloseable {
        private final OrtEnvironment env;
        private final OrtSession session;
        private final JsonNode codecMeta;
        private final List<JsonNode> transformerSpecs;
        private final List<JsonNode> attentionSpecs;
        private Map<String, OnnxTensorLike> stateFeeds;
        private final List<AutoCloseable> ownedStateFeeds = new ArrayList<>();
        private OrtSession.Result stateResult;

        private CodecStreamingDecodeState(
                OrtEnvironment env,
                OrtSession session,
                JsonNode codecMeta,
                List<JsonNode> transformerSpecs,
                List<JsonNode> attentionSpecs,
                Map<String, OnnxTensorLike> stateFeeds,
                List<AutoCloseable> ownedStateFeeds) {
            this.env = env;
            this.session = session;
            this.codecMeta = codecMeta;
            this.transformerSpecs = transformerSpecs;
            this.attentionSpecs = attentionSpecs;
            this.stateFeeds = stateFeeds;
            this.ownedStateFeeds.addAll(ownedStateFeeds);
        }

        static CodecStreamingDecodeState create(OrtEnvironment env, OrtSession session, JsonNode codecMeta)
                throws OrtException {
            List<JsonNode> transformerSpecs = jsonArray(codecMeta.path("streaming_decode").path("transformer_offsets"));
            List<JsonNode> attentionSpecs = jsonArray(codecMeta.path("streaming_decode").path("attention_caches"));
            Map<String, OnnxTensorLike> stateFeeds = new LinkedHashMap<>();
            List<AutoCloseable> owned = new ArrayList<>();

            for (JsonNode spec : transformerSpecs) {
                OnnxTensor tensor = OnnxTensor.createTensor(env, new int[] { 0 });
                stateFeeds.put(spec.path("input_name").asText(), tensor);
                owned.add(tensor);
            }
            for (JsonNode spec : attentionSpecs) {
                int heads = spec.path("cache_shape").path(1).asInt(4);
                int context = spec.path("cache_shape").path(2).asInt(500);
                int headDim = spec.path("cache_shape").path(3).asInt(64);

                OnnxTensor offset = OnnxTensor.createTensor(env, new int[] { 0 });
                OnnxTensor keys = OnnxTensor.createTensor(env, new float[1][heads][context][headDim]);
                OnnxTensor values = OnnxTensor.createTensor(env, new float[1][heads][context][headDim]);
                int[][] positionsArray = new int[1][context];
                java.util.Arrays.fill(positionsArray[0], -1);
                OnnxTensor positions = OnnxTensor.createTensor(env, positionsArray);

                stateFeeds.put(spec.path("offset_input_name").asText(), offset);
                stateFeeds.put(spec.path("cached_keys_input_name").asText(), keys);
                stateFeeds.put(spec.path("cached_values_input_name").asText(), values);
                stateFeeds.put(spec.path("cached_positions_input_name").asText(), positions);
                owned.add(offset);
                owned.add(keys);
                owned.add(values);
                owned.add(positions);
            }

            return new CodecStreamingDecodeState(env, session, codecMeta, transformerSpecs, attentionSpecs, stateFeeds, owned);
        }

        DecodedAudio runFrames(List<int[]> frames) throws OrtException {
            int nVq = codecMeta.path("codec_config").path("num_quantizers").asInt(16);
            int[][][] codes = new int[1][frames.size()][nVq];
            for (int frame = 0; frame < frames.size(); frame++) {
                int[] src = frames.get(frame);
                for (int channel = 0; channel < Math.min(nVq, src.length); channel++) {
                    codes[0][frame][channel] = src[channel];
                }
            }

            try (OnnxTensor audioCodes = OnnxTensor.createTensor(env, codes);
                 OnnxTensor audioCodeLengths = OnnxTensor.createTensor(env, new int[] { frames.size() })) {
                Map<String, OnnxTensorLike> feeds = new LinkedHashMap<>();
                feeds.put("audio_codes", audioCodes);
                feeds.put("audio_code_lengths", audioCodeLengths);
                feeds.putAll(stateFeeds);

                OrtSession.Result result = session.run(feeds);
                DecodedAudio audio;
                try {
                    audio = readStreamingDecodedAudio(result);
                    Map<String, OnnxTensorLike> nextStateFeeds = nextStateFeeds(result);
                    closePreviousState();
                    stateFeeds = nextStateFeeds;
                    stateResult = result;
                    result = null;
                    return audio;
                } finally {
                    closeQuietly(result);
                }
            }
        }

        private DecodedAudio readStreamingDecodedAudio(OrtSession.Result result) throws OrtException {
            OnnxTensor audio = tensor(result, "audio");
            OnnxTensor audioLengths = tensor(result, "audio_lengths");
            long[] shape = audio.getInfo().getShape();
            if (shape.length != 3 || shape[0] != 1) {
                throw new IllegalStateException("Unexpected MOSS streaming codec audio shape: "
                        + java.util.Arrays.toString(shape));
            }
            int channels = Math.toIntExact(shape[1]);
            int totalSamples = Math.toIntExact(shape[2]);
            int samples = Math.min(totalSamples, firstScalarAsInt(audioLengths.getValue()));
            float[] flat = readFloatVector(audio);
            if (samples < totalSamples) {
                float[] compact = new float[channels * samples];
                for (int channel = 0; channel < channels; channel++) {
                    System.arraycopy(flat, channel * totalSamples, compact, channel * samples, samples);
                }
                flat = compact;
            }
            int sampleRate = codecMeta.path("codec_config").path("sample_rate").asInt(48_000);
            return new DecodedAudio(flat, channels, samples, sampleRate, channels, "streaming_decode_step", "native");
        }

        private Map<String, OnnxTensorLike> nextStateFeeds(OrtSession.Result result) throws OrtException {
            Map<String, OnnxTensorLike> next = new LinkedHashMap<>();
            for (JsonNode spec : transformerSpecs) {
                next.put(spec.path("input_name").asText(), tensor(result, spec.path("output_name").asText()));
            }
            for (JsonNode spec : attentionSpecs) {
                next.put(spec.path("offset_input_name").asText(), tensor(result, spec.path("offset_output_name").asText()));
                next.put(spec.path("cached_keys_input_name").asText(), tensor(result, spec.path("cached_keys_output_name").asText()));
                next.put(spec.path("cached_values_input_name").asText(), tensor(result, spec.path("cached_values_output_name").asText()));
                next.put(spec.path("cached_positions_input_name").asText(), tensor(result, spec.path("cached_positions_output_name").asText()));
            }
            return next;
        }

        private void closePreviousState() {
            if (!ownedStateFeeds.isEmpty()) {
                for (AutoCloseable closeable : ownedStateFeeds) {
                    closeQuietly(closeable);
                }
                ownedStateFeeds.clear();
            }
            closeQuietly(stateResult);
            stateResult = null;
        }

        @Override
        public void close() {
            closePreviousState();
        }
    }

    private record VoicePrompt(String label, String id, String mode, String language, int[][] promptAudioCodes) {
    }

    private record VoiceSelector(String language, String gender) {
        private boolean isLanguageOnly() {
            return gender == null || gender.isBlank();
        }
    }

    private record VoiceSelectorParse(String language, String gender, List<String> unknownTerms) {
    }

    private record ResolvedSeed(long value, String source) {
    }

    private record TtsConfig(
            int nVq,
            int rowWidth,
            int audioPadTokenId,
            int audioStartTokenId,
            int audioEndTokenId,
            int audioUserSlotTokenId,
            int audioAssistantSlotTokenId,
            int audioCodebookSize) {
        static TtsConfig from(JsonNode manifest, JsonNode ttsMeta) {
            JsonNode config = manifest.path("tts_config");
            JsonNode modelConfig = ttsMeta == null ? null : ttsMeta.path("model_config");
            int nVq = config.path("n_vq").asInt(modelConfig == null ? 16 : modelConfig.path("n_vq").asInt(16));
            int codebook = 1024;
            if (modelConfig != null && modelConfig.path("audio_codebook_sizes").isArray()
                    && !modelConfig.path("audio_codebook_sizes").isEmpty()) {
                codebook = modelConfig.path("audio_codebook_sizes").get(0).asInt(1024);
            }
            return new TtsConfig(
                    nVq,
                    config.path("row_width").asInt(nVq + 1),
                    config.path("audio_pad_token_id").asInt(1024),
                    config.path("audio_start_token_id").asInt(6),
                    config.path("audio_end_token_id").asInt(7),
                    config.path("audio_user_slot_token_id").asInt(8),
                    config.path("audio_assistant_slot_token_id").asInt(9),
                    codebook);
        }
    }

    private record MossSessions(
            OrtSession prefill,
            OrtSession decode,
            OrtSession localFixedSampledFrame,
            LazyOrtSession lazyCodecDecode,
            LazyOrtSession lazyCodecDecodeStep,
            List<OptimizedModelCacheEvent> eagerCacheEvents) implements AutoCloseable {
        static MossSessions open(
                OrtEnvironment env,
                Path ttsDir,
                Path codecDir,
                JsonNode ttsMeta,
                JsonNode codecMeta,
                int threads) throws OrtException {
            JsonNode ttsFiles = ttsMeta.path("files");
            JsonNode codecFiles = codecMeta.path("files");
            Path prefillPath = ttsDir.resolve(ttsFiles.path("prefill").asText("moss_tts_prefill.onnx"));
            Path decodePath = ttsDir.resolve(ttsFiles.path("decode_step").asText("moss_tts_decode_step.onnx"));
            Path localFixedSampledFramePath = ttsDir.resolve(ttsFiles.path("local_fixed_sampled_frame")
                    .asText("moss_tts_local_fixed_sampled_frame.onnx"));
            Path codecDecodePath = codecDir.resolve(codecFiles.path("decode_full")
                    .asText("moss_audio_tokenizer_decode_full.onnx"));
            Path codecDecodeStepPath = codecDir.resolve(codecFiles.path("decode_step")
                    .asText("moss_audio_tokenizer_decode_step.onnx"));
            SessionOpen prefill = createSession(env, prefillPath, threads);
            SessionOpen decode = createSession(env, decodePath, threads);
            SessionOpen localFixedSampledFrame = createSession(env, localFixedSampledFramePath, threads);
            LazyOrtSession codecDecode = new LazyOrtSession(env, codecDecodePath, threads);
            LazyOrtSession codecDecodeStep = new LazyOrtSession(env, codecDecodeStepPath, threads);
            List<OptimizedModelCacheEvent> eagerCacheEvents = List.of(
                    prefill.cache(),
                    decode.cache(),
                    localFixedSampledFrame.cache());
            return new MossSessions(
                    prefill.session(),
                    decode.session(),
                    localFixedSampledFrame.session(),
                    codecDecode,
                    codecDecodeStep,
                    eagerCacheEvents);
        }

        private static String sessionCacheKey(
                Path ttsDir,
                Path codecDir,
                JsonNode ttsMeta,
                JsonNode codecMeta,
                int threads) {
            JsonNode ttsFiles = ttsMeta.path("files");
            JsonNode codecFiles = codecMeta.path("files");
            List<Path> paths = List.of(
                    ttsDir.resolve(ttsFiles.path("prefill").asText("moss_tts_prefill.onnx")),
                    ttsDir.resolve(ttsFiles.path("decode_step").asText("moss_tts_decode_step.onnx")),
                    ttsDir.resolve(ttsFiles.path("local_fixed_sampled_frame")
                            .asText("moss_tts_local_fixed_sampled_frame.onnx")),
                    codecDir.resolve(codecFiles.path("decode_full")
                            .asText("moss_audio_tokenizer_decode_full.onnx")),
                    codecDir.resolve(codecFiles.path("decode_step")
                            .asText("moss_audio_tokenizer_decode_step.onnx")));
            try {
                MessageDigest digest = MessageDigest.getInstance("SHA-256");
                digest.update(("threads=" + Math.max(1, threads) + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
                digest.update(("os=" + System.getProperty("os.name", "") + "\n")
                        .getBytes(java.nio.charset.StandardCharsets.UTF_8));
                digest.update(("arch=" + System.getProperty("os.arch", "") + "\n")
                        .getBytes(java.nio.charset.StandardCharsets.UTF_8));
                for (Path path : paths) {
                    updateGraphIdentityDigest(digest, path.toAbsolutePath().normalize());
                }
                return HexFormat.of().formatHex(digest.digest()).substring(0, 24);
            } catch (Exception e) {
                String fallback = ttsDir.toAbsolutePath().normalize()
                        + "|" + codecDir.toAbsolutePath().normalize()
                        + "|threads=" + Math.max(1, threads);
                return Integer.toHexString(fallback.hashCode());
            }
        }

        private OrtSession codecDecode() throws OrtException {
            return lazyCodecDecode.session();
        }

        private OrtSession codecDecodeStep() throws OrtException {
            return lazyCodecDecodeStep.session();
        }

        private OptimizedModelCacheStats cacheStats() {
            List<OptimizedModelCacheEvent> events = new ArrayList<>(eagerCacheEvents);
            lazyCodecDecode.addCacheEventTo(events);
            lazyCodecDecodeStep.addCacheEventTo(events);
            return OptimizedModelCacheStats.from(events);
        }

        private static SessionOpen createSession(OrtEnvironment env, Path path, int threads) throws OrtException {
            Path modelPath = path.toAbsolutePath().normalize();
            Path optimizedPath = optimizedModelCachePath(modelPath);
            int sidecars = 0;
            if (optimizedPath != null) {
                sidecars = prepareOptimizedModelSidecars(modelPath, optimizedPath);
            }
            boolean staleCacheRebuilt = false;
            if (optimizedPath != null && Files.isRegularFile(optimizedPath)) {
                try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                    configureSessionLogging(options);
                    options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT);
                    options.setIntraOpNumThreads(Math.max(1, threads));
                    options.setInterOpNumThreads(1);
                    return new SessionOpen(
                            env.createSession(optimizedPath.toString(), options),
                            OptimizedModelCacheEvent.hit(optimizedPath, sidecars));
                } catch (OrtException e) {
                    try {
                        Files.deleteIfExists(optimizedPath);
                    } catch (Exception ignored) {
                    }
                    staleCacheRebuilt = true;
                }
            }
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                configureSessionLogging(options);
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                options.setIntraOpNumThreads(Math.max(1, threads));
                options.setInterOpNumThreads(1);
                boolean cacheWriteEnabled = false;
                if (optimizedPath != null) {
                    try {
                        Files.createDirectories(optimizedPath.getParent());
                        options.setOptimizedModelFilePath(optimizedPath.toString());
                        cacheWriteEnabled = true;
                    } catch (Exception ignored) {
                    }
                }
                OrtSession session = env.createSession(modelPath.toString(), options);
                OptimizedModelCacheEvent cacheEvent = cacheWriteEnabled
                        ? (staleCacheRebuilt
                                ? OptimizedModelCacheEvent.rebuilt(optimizedPath, sidecars)
                                : OptimizedModelCacheEvent.created(optimizedPath, sidecars))
                        : OptimizedModelCacheEvent.disabled(sidecars);
                return new SessionOpen(session, cacheEvent);
            }
        }

        private static int prepareOptimizedModelSidecars(Path modelPath, Path optimizedPath) {
            Path sourceDir = modelPath.getParent();
            Path cacheDir = optimizedPath.getParent();
            if (sourceDir == null || cacheDir == null || !Files.isDirectory(sourceDir)) {
                return 0;
            }
            try {
                Files.createDirectories(cacheDir);
            } catch (Exception e) {
                return 0;
            }
            int prepared = 0;
            try {
                List<Path> sidecars = externalDataSidecars(modelPath);
                for (Path source : sidecars) {
                    try {
                        linkOrCopySidecar(source, cacheDir.resolve(source.getFileName().toString()));
                        prepared++;
                    } catch (Exception ignored) {
                    }
                }
            } catch (Exception ignored) {
            }
            return prepared;
        }

        private static void linkOrCopySidecar(Path source, Path target) throws Exception {
            if (Files.isRegularFile(target) && Files.size(target) == Files.size(source)) {
                return;
            }
            try {
                Files.deleteIfExists(target);
                Files.createLink(target, source);
            } catch (Exception e) {
                Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.COPY_ATTRIBUTES);
            }
        }

        private static Path optimizedModelCachePath(Path modelPath) {
            if (!optimizedModelCacheEnabled()) {
                return null;
            }
            try {
                String root = firstNonBlank(
                        System.getProperty(OPTIMIZED_CACHE_DIR_PROPERTY),
                        System.getenv(OPTIMIZED_CACHE_DIR_ENV));
                Path cacheRoot = root.isBlank()
                        ? Path.of(System.getProperty("user.home"), ".gollek", "cache", "onnx", "optimized", "moss-tts")
                        : Path.of(root).resolve("moss-tts");
                String fingerprint = optimizedModelFingerprint(modelPath);
                String filename = sanitizeCacheFilename(modelPath.getFileName().toString())
                        + "-" + fingerprint.substring(0, 16) + ".optimized.onnx";
                return cacheRoot.resolve(filename);
            } catch (Exception e) {
                return null;
            }
        }

        private static boolean optimizedModelCacheEnabled() {
            String raw = firstNonBlank(
                    System.getProperty(OPTIMIZED_CACHE_ENABLED_PROPERTY),
                    System.getenv(OPTIMIZED_CACHE_ENABLED_ENV),
                    "true");
            return !Set.of("0", "false", "no", "off", "disable", "disabled")
                    .contains(raw.trim().toLowerCase(Locale.ROOT));
        }

        private static String optimizedModelFingerprint(Path modelPath) throws Exception {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            updateDigestLine(digest, "os=" + System.getProperty("os.name", ""));
            updateDigestLine(digest, "arch=" + System.getProperty("os.arch", ""));
            updateDigestLine(digest, "runtime=ort-java");
            updateDigestLine(digest, "optimization=all-opt");
            updateDigestLine(digest, "backend=cpu");
            updateGraphIdentityDigest(digest, modelPath);
            return HexFormat.of().formatHex(digest.digest());
        }

        private static void updateGraphIdentityDigest(MessageDigest digest, Path modelPath) throws Exception {
            updateFileIdentityDigest(digest, "onnx", modelPath);
            List<Path> sidecars = externalDataSidecars(modelPath);
            updateDigestLine(digest, "external_data_sidecars=" + sidecars.size());
            for (Path sidecar : sidecars) {
                updateFileIdentityDigest(digest, "data", sidecar);
            }
        }

        private static void updateFileIdentityDigest(MessageDigest digest, String kind, Path path) throws Exception {
            Path modelPath = path.toAbsolutePath().normalize();
            updateDigestLine(digest, String.join("\t",
                    kind,
                    modelPath.toRealPath().toString(),
                    String.valueOf(Files.size(modelPath)),
                    String.valueOf(Files.getLastModifiedTime(modelPath).toMillis())));
        }

        private static void updateDigestLine(MessageDigest digest, String text) {
            digest.update(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            digest.update((byte) '\n');
        }

        private static List<Path> externalDataSidecars(Path modelPath) throws Exception {
            Path sourceDir = modelPath.toAbsolutePath().normalize().getParent();
            if (sourceDir == null || !Files.isDirectory(sourceDir)) {
                return List.of();
            }
            try (var stream = Files.list(sourceDir)) {
                return stream
                        .filter(path -> Files.isRegularFile(path) && path.getFileName().toString().endsWith(".data"))
                        .sorted((left, right) -> left.getFileName().toString()
                                .compareTo(right.getFileName().toString()))
                        .toList();
            }
        }

        private static String sanitizeCacheFilename(String value) {
            String normalized = value == null ? "model" : value.toLowerCase(Locale.ROOT)
                    .replaceAll("[^a-z0-9._-]+", "-")
                    .replaceAll("[-_.]{2,}", "-")
                    .replaceAll("^[-_.]+|[-_.]+$", "");
            return normalized.isBlank() ? "model" : normalized;
        }

        private static String firstNonBlank(String... values) {
            if (values == null) {
                return "";
            }
            for (String value : values) {
                if (value != null && !value.isBlank()) {
                    return value.trim();
                }
            }
            return "";
        }

        @Override
        public void close() {
            closeQuietly(lazyCodecDecodeStep);
            closeQuietly(lazyCodecDecode);
            closeQuietly(localFixedSampledFrame);
            closeQuietly(decode);
            closeQuietly(prefill);
        }
    }

    private static final class LazyOrtSession implements AutoCloseable {
        private final OrtEnvironment env;
        private final Path path;
        private final int threads;
        private OrtSession session;
        private OptimizedModelCacheEvent cacheEvent;

        private LazyOrtSession(OrtEnvironment env, Path path, int threads) {
            this.env = env;
            this.path = path;
            this.threads = threads;
        }

        private OrtSession session() throws OrtException {
            if (session == null) {
                SessionOpen opened = MossSessions.createSession(env, path, threads);
                session = opened.session();
                cacheEvent = opened.cache();
            }
            return session;
        }

        private void addCacheEventTo(List<OptimizedModelCacheEvent> events) {
            if (cacheEvent != null) {
                events.add(cacheEvent);
            }
        }

        @Override
        public void close() {
            closeQuietly(session);
        }
    }

    private record SessionOpen(OrtSession session, OptimizedModelCacheEvent cache) {
    }

    private record OptimizedModelCacheEvent(String state, Path path, int sidecars) {
        static OptimizedModelCacheEvent hit(Path path, int sidecars) {
            return new OptimizedModelCacheEvent("hit", path, sidecars);
        }

        static OptimizedModelCacheEvent created(Path path, int sidecars) {
            return new OptimizedModelCacheEvent("created", path, sidecars);
        }

        static OptimizedModelCacheEvent rebuilt(Path path, int sidecars) {
            return new OptimizedModelCacheEvent("rebuilt", path, sidecars);
        }

        static OptimizedModelCacheEvent disabled(int sidecars) {
            return new OptimizedModelCacheEvent("disabled", null, sidecars);
        }
    }

    private record OptimizedModelCacheStats(
            boolean enabled,
            int sessions,
            int hits,
            int created,
            int rebuilt,
            int disabled,
            int sidecars,
            String root) {
        static OptimizedModelCacheStats from(List<OptimizedModelCacheEvent> events) {
            int sessions = events.size();
            int hits = 0;
            int created = 0;
            int rebuilt = 0;
            int disabled = 0;
            int sidecars = 0;
            String root = "";
            for (OptimizedModelCacheEvent event : events) {
                if (event == null) {
                    disabled++;
                    continue;
                }
                sidecars += Math.max(0, event.sidecars());
                if (root.isBlank() && event.path() != null && event.path().getParent() != null) {
                    root = event.path().getParent().toString();
                }
                switch (event.state()) {
                    case "hit" -> hits++;
                    case "created" -> created++;
                    case "rebuilt" -> rebuilt++;
                    default -> disabled++;
                }
            }
            return new OptimizedModelCacheStats(
                    hits + created + rebuilt > 0,
                    sessions,
                    hits,
                    created,
                    rebuilt,
                    disabled,
                    sidecars,
                    root);
        }

        void putMetadata(Map<String, Object> metadata) {
            metadata.put("onnx_optimized_model_cache_enabled", enabled);
            metadata.put("onnx_optimized_model_cache_sessions", sessions);
            metadata.put("onnx_optimized_model_cache_hits", hits);
            metadata.put("onnx_optimized_model_cache_created", created);
            metadata.put("onnx_optimized_model_cache_rebuilt", rebuilt);
            metadata.put("onnx_optimized_model_cache_disabled", disabled);
            metadata.put("onnx_optimized_model_cache_sidecars", sidecars);
            if (root != null && !root.isBlank()) {
                metadata.put("onnx_optimized_model_cache_root", root);
            }
        }
    }
}
