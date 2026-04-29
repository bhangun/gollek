package tech.kayys.gollek.safetensor.audio;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.audio.model.AudioConfig;
import tech.kayys.gollek.safetensor.audio.model.AudioResult;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.safetensor.audio.processing.Mp3Encoder;
import tech.kayys.suling.FlacAudioFormat;
import tech.kayys.suling.encoder.FlacStreamEncoder;
import tech.kayys.suling.encoder.StreamEncoderH;

import java.lang.foreign.MemorySegment;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;

/**
 * SpeechT5 text-to-speech engine — converts text to 16kHz WAV audio.
 * <p>
 * Features:
 * <ul>
 * <li>High-quality HiFi-GAN vocoder</li>
 * <li>Multiple voice presets</li>
 * <li>Speaker embedding support</li>
 * <li>Prosody control (speed)</li>
 * </ul>
 * </p>
 */
@ApplicationScoped
public class SpeechT5Engine {

    private static final Logger log = Logger.getLogger(SpeechT5Engine.class);

    public static final int SAMPLE_RATE = 16000;
    public static final int N_MEL = 80;
    public static final int HOP_LENGTH = 256; // HiFi-GAN hop
    public static final int MAX_MEL_FRAMES = 3000; // ~19 seconds
    private static final int SPEAKER_EMB_DIM = 512;

    @ConfigProperty(name = "gollek.audio.tts.default-voice", defaultValue = "alloy")
    String defaultVoice;

    @ConfigProperty(name = "gollek.audio.tts.speed", defaultValue = "1.0")
    float defaultSpeed;

    @Inject
    SafetensorEngine engine;

    private final Map<String, float[]> speakerEmbeddings = new ConcurrentHashMap<>();
    private final HiFiGANVocoder vocoder = new HiFiGANVocoder();

    public SpeechT5Engine() {
        initializeSpeakerEmbeddings();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    public Uni<byte[]> synthesize(String text, String voice, Path modelPath, AudioConfig config) {
        return Uni.createFrom().<byte[]>emitter(em -> {
            Instant startTime = Instant.now();
            try {
                log.infof("SpeechT5: synthesizing %d chars [voice=%s, speed=%.1fx]",
                        text.length(), voice, config != null ? config.getTemperature() : 1.0f);

                SafetensorEngine.LoadedModel model = requireModel(modelPath);
                @SuppressWarnings("unchecked") Map<String, AccelTensor> weights = (Map<String, AccelTensor>) (Map<?, ?>) model.weights();
                String pfx = detectPrefix(weights);
                log.infof("SpeechT5: detected weight prefix: '%s'", pfx);

                // 1. Normalize and tokenize text
                String normalizedText = normalizeText(text);
                long[] encoded = model.tokenizer().encode(normalizedText, EncodeOptions.builder().addBos(true).build());
                int[] tokenIds = new int[encoded.length];
                for (int i = 0; i < encoded.length; i++)
                    tokenIds[i] = (int) encoded[i];

                // 2. Encode text
                AccelTensor encoderOut = runTextEncoder(tokenIds, model, pfx);

                // 3. Get speaker embedding
                float[] speakerEmb = getSpeakerEmbedding(voice != null ? voice : defaultVoice);
                AccelTensor speakerTensor = AccelTensor.fromFloatArray(speakerEmb, new long[] { 1, SPEAKER_EMB_DIM });

                // 4. Decode to mel spectrogram
                float[][] melFrames = runDecoder(encoderOut, speakerTensor, model, config, pfx);
                encoderOut.close();
                speakerTensor.close();

                // 5. Vocoder: mel → audio
                @SuppressWarnings("unchecked")
                Map<String, AccelTensor> vocoderWeights = (Map<String, AccelTensor>) (Map<?, ?>) model.weights();
                float[] pcm = vocoder.synthesize(melFrames, vocoderWeights);

                // 6. Apply speed adjustment if requested (using temperature field for speed in
                // this context)
                if (config != null && config.getTemperature() != 1.0f) {
                    pcm = adjustSpeed(pcm, config.getTemperature());
                }

                // 7. Encode to requested format
                byte[] audioOut;
                AudioConfig.Format requestedFormat = config != null ? config.getFormat() : AudioConfig.Format.FLAC;
                
                try {
                    switch (requestedFormat) {
                        case FLAC -> audioOut = encodeFlac(pcm, SAMPLE_RATE);
                        case MP3 -> {
                            log.info("SpeechT5: Encoding output to MP3 using Jump3r...");
                            try {
                                Mp3Encoder encoder = new Mp3Encoder(SAMPLE_RATE);
                                audioOut = encoder.encode(pcm);
                            } catch (Exception e) {
                                log.warn("SpeechT5: MP3 encoding failed, falling back to WAV: " + e.getMessage());
                                audioOut = encodeWav(pcm, SAMPLE_RATE);
                            }
                        }
                        default -> audioOut = encodeWav(pcm, SAMPLE_RATE);
                    }
                } catch (Throwable t) {
                    log.warn("SpeechT5: Encoding to " + requestedFormat + " failed, falling back to WAV: " + t.getMessage());
                    audioOut = encodeWav(pcm, SAMPLE_RATE);
                }

                long durationMs = java.time.Duration.between(startTime, Instant.now()).toMillis();
                log.infof("SpeechT5: generated %.1fs audio (%d bytes, FLAC) in %dms",
                        pcm.length / (double) SAMPLE_RATE, audioOut.length, durationMs);
                em.complete(audioOut);

            } catch (Exception e) {
                log.errorf(e, "SpeechT5 synthesis failed");
                em.fail(e);
            }
        }).runSubscriptionOn(Executors.newVirtualThreadPerTaskExecutor());
    }

    public Uni<byte[]> synthesize(String text, String voice, Path modelPath) {
        return synthesize(text, voice, modelPath, AudioConfig.forTTS(voice));
    }

    public Uni<AudioResult> synthesizeWithResult(String text, String voice, Path modelPath, AudioConfig config) {
        return synthesize(text, voice, modelPath, config)
                .map(flac -> {
                    double duration = flac.length > 0 ? -1.0 : 0; // We don't have exact duration easily from raw FLAC without parsing, but the frontend can parse it.
                    return AudioResult.speechSynthesis(flac, modelPath.getFileName().toString(), duration);
                });
    }

    public void registerSpeaker(String voiceName, float[] embedding) {
        if (embedding.length != SPEAKER_EMB_DIM) {
            throw new IllegalArgumentException("Speaker embedding must be " + SPEAKER_EMB_DIM + "-dimensional");
        }
        speakerEmbeddings.put(voiceName.toLowerCase(), normalizeEmbedding(embedding));
        log.infof("Registered custom speaker: %s", voiceName);
    }

    public List<String> getAvailableVoices() {
        return new ArrayList<>(speakerEmbeddings.keySet());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Implementation details
    // ─────────────────────────────────────────────────────────────────────────

    private void initializeSpeakerEmbeddings() {
        String[] voices = { "alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "ballad" };
        for (String voice : voices) {
            speakerEmbeddings.put(voice.toLowerCase(), generateSpeakerEmbedding(voice));
        }
        log.infof("Initialized %d speaker embeddings", speakerEmbeddings.size());
    }

    private float[] generateSpeakerEmbedding(String voiceName) {
        Random rng = new Random(voiceName.hashCode());
        float[] emb = new float[SPEAKER_EMB_DIM];
        double norm = 0;
        for (int i = 0; i < SPEAKER_EMB_DIM; i++) {
            emb[i] = (float) rng.nextGaussian();
            norm += emb[i] * emb[i];
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < SPEAKER_EMB_DIM; i++)
            emb[i] /= norm;
        return emb;
    }

    private float[] getSpeakerEmbedding(String voice) {
        float[] emb = speakerEmbeddings.get(voice.toLowerCase());
        if (emb == null) {
            log.warnf("Voice '%s' not found, using default '%s'", voice, defaultVoice);
            emb = speakerEmbeddings.get(defaultVoice.toLowerCase());
        }
        return emb != null ? emb : generateSpeakerEmbedding("alloy");
    }

    private float[] normalizeEmbedding(float[] emb) {
        double norm = 0;
        for (float v : emb)
            norm += v * v;
        norm = Math.sqrt(norm);
        float[] res = new float[emb.length];
        for (int i = 0; i < emb.length; i++)
            res[i] = (float) (emb[i] / norm);
        return res;
    }

    private String detectPrefix(Map<String, AccelTensor> weights) {
        if (weights.containsKey("speecht5.encoder.embed_tokens.weight")) return "speecht5.";
        if (weights.containsKey("model.encoder.embed_tokens.weight") || weights.containsKey("model.decoder.embed_tokens.weight")) return "model.";
        if (weights.containsKey("encoder.embed_tokens.weight") || weights.containsKey("decoder.embed_tokens.weight")) return "";
        return "speecht5.";
    }

    private AccelTensor runTextEncoder(int[] tokenIds, SafetensorEngine.LoadedModel model, String pfx) {
        @SuppressWarnings("unchecked") Map<String, AccelTensor> w = (Map<String, AccelTensor>) (Map<?, ?>) model.weights();
        AccelTensor embedW = w.get(pfx + "encoder.embed_tokens.weight");
        if (embedW == null) embedW = w.get(pfx + "decoder.embed_tokens.weight");
        if (embedW == null) embedW = w.get(pfx + "embed_tokens.weight");
        int hiddenSize = embedW != null ? (int) embedW.shape()[1] : 768;

        if (embedW == null) {
            log.warnf("SpeechT5: encoder embed_tokens not found with prefix '%s' — using zeros", pfx);
            return AccelTensor.fromFloatArray(new float[tokenIds.length * hiddenSize],
                    new long[] { 1, tokenIds.length, hiddenSize });
        }

        float[] embedData = embedW.toFloatArray();
        float[] tokenEmbeds = new float[tokenIds.length * hiddenSize];
        for (int i = 0; i < tokenIds.length; i++) {
            int id = Math.min(tokenIds[i], (int) embedW.shape()[0] - 1);
            System.arraycopy(embedData, id * hiddenSize, tokenEmbeds, i * hiddenSize, hiddenSize);
        }
        AccelTensor hidden = AccelTensor.fromFloatArray(tokenEmbeds, new long[] { 1, tokenIds.length, hiddenSize });

        int numLayers = countEncoderLayers(w, pfx);
        for (int i = 0; i < numLayers; i++) {
            AccelTensor layerOut = encoderLayer(hidden, w, i, hiddenSize, pfx);
            if (layerOut != hidden) {
                hidden.close();
            }
            hidden = layerOut;
        }
        return hidden;
    }


    private AccelTensor encoderLayer(AccelTensor x, Map<String, AccelTensor> w, int i, int d, String pfx) {
        String layerPfx = pfx + "encoder.layers.%d.".formatted(i);
        AccelTensor qW = w.get(layerPfx + "attention.q_proj.weight");
        if (qW == null)
            return x;

        AccelTensor kW = w.get(layerPfx + "attention.k_proj.weight");
        AccelTensor vW = w.get(layerPfx + "attention.v_proj.weight");
        AccelTensor oW = w.get(layerPfx + "attention.out_proj.weight");
        AccelTensor n1W = w.get(layerPfx + "layer_norm.weight");
        AccelTensor fc1W = w.get(layerPfx + "feed_forward.intermediate_dense.weight");
        AccelTensor fc2W = w.get(layerPfx + "feed_forward.output_dense.weight");

        AccelTensor normed = layerNorm(x, n1W, 1e-5);
        AccelTensor attnOut = selfAttention(normed, qW, kW, vW, oW, d);
        if (normed != x) normed.close();
        AccelTensor h = AccelOps.add(x, attnOut);
        if (attnOut != x) attnOut.close();

        if (fc1W != null && fc2W != null) {
            AccelTensor n2W = w.get(layerPfx + "final_layer_norm.weight");
            AccelTensor normed2 = layerNorm(h, n2W, 1e-5);
            AccelTensor ffnOut = reluFfn(normed2, fc1W, fc2W);
            if (normed2 != h) normed2.close();
            AccelTensor h2 = AccelOps.add(h, ffnOut);
            if (ffnOut != h) ffnOut.close();
            h.close();
            return h2;
        }
        return h;
    }

    private float[][] runDecoder(AccelTensor encoderOut, AccelTensor speakerTensor,
            SafetensorEngine.LoadedModel model, AudioConfig config, String pfx) {
        @SuppressWarnings("unchecked") Map<String, AccelTensor> w = (Map<String, AccelTensor>) (Map<?, ?>) model.weights();
        int hiddenSize = (int) encoderOut.shape()[2];
        List<float[]> melFrames = new ArrayList<>();
        float[] prevMelFrame = new float[N_MEL];
        AccelTensor decoderHidden = AccelTensor.fromFloatArray(prevMelFrame, new long[] { 1, 1, N_MEL });

        for (int step = 0; step < MAX_MEL_FRAMES; step++) {
            AccelTensor prenetW = w.get(pfx + "decoder.prenet.layers.0.weight");
            AccelTensor projected = prenetW != null ? linear(decoderHidden, prenetW) : decoderHidden;

            AccelTensor withSpk = addSpeakerEmbedding(projected, speakerTensor, hiddenSize, w, pfx);
            if (projected != decoderHidden && projected != withSpk)
                projected.close();

            int numDecLayers = countDecoderLayers(w, pfx);
            AccelTensor decOut = withSpk;
            for (int i = 0; i < numDecLayers; i++) {
                AccelTensor layerOut = decoderLayer(decOut, encoderOut, w, i, hiddenSize, pfx);
                if (layerOut != decOut) {
                    decOut.close();
                }
                decOut = layerOut;
            }

            AccelTensor featW = w.get(pfx.replace("speecht5.", "speech_") + "decoder_postnet.feat_out.weight");
            if (featW == null) featW = w.get(pfx + "decoder_postnet.feat_out.weight");
            
            AccelTensor melOut = featW != null ? linear(decOut, featW) : decOut;
            float[] frame = melOut.toFloatArray();
            if (melOut != decOut)
                melOut.close();
            decOut.close();
            decoderHidden.close();

            float[] melFrame = Arrays.copyOf(frame, N_MEL);
            melFrames.add(melFrame);

            if (detectStop(frame, w, pfx))
                break;
            decoderHidden = AccelTensor.fromFloatArray(melFrame, new long[] { 1, 1, N_MEL });
        }
        return melFrames.toArray(float[][]::new);
    }

    private AccelTensor decoderLayer(AccelTensor x, AccelTensor encOut, Map<String, AccelTensor> w, int i, int d, String pfx) {
        String layerPfx = pfx + "decoder.layers.%d.".formatted(i);
        AccelTensor qW = w.get(layerPfx + "self_attn.q_proj.weight");
        if (qW == null)
            return x;

        AccelTensor kW = w.get(layerPfx + "self_attn.k_proj.weight");
        AccelTensor vW = w.get(layerPfx + "self_attn.v_proj.weight");
        AccelTensor oW = w.get(layerPfx + "self_attn.out_proj.weight");
        AccelTensor n1W = w.get(layerPfx + "self_attn_layer_norm.weight");

        AccelTensor normed = layerNorm(x, n1W, 1e-5);
        AccelTensor attn = selfAttention(normed, qW, kW, vW, oW, d);
        if (normed != x) normed.close();
        AccelTensor h = AccelOps.add(x, attn);
        if (attn != x) attn.close();

        AccelTensor cqW = w.get(layerPfx + "encoder_attn.q_proj.weight");
        if (cqW != null) {
            AccelTensor ckW = w.get(layerPfx + "encoder_attn.k_proj.weight");
            AccelTensor cvW = w.get(layerPfx + "encoder_attn.v_proj.weight");
            AccelTensor coW = w.get(layerPfx + "encoder_attn.out_proj.weight");
            AccelTensor n2W = w.get(layerPfx + "encoder_attn_layer_norm.weight");

            AccelTensor normed2 = layerNorm(h, n2W, 1e-5);
            AccelTensor q = linear(normed2, cqW);
            AccelTensor k = linear(encOut, ckW);
            AccelTensor v = linear(encOut, cvW);

            float scale = (float) (1.0 / Math.sqrt(d / 8));
            AccelTensor scoresTemp = AccelOps.matmul(q, encOut.transpose(1, 2));
            AccelTensor scores = AccelOps.mulScalar(scoresTemp, scale);
            scoresTemp.close();
            AccelTensor attnW = AccelOps.softmax(scores, -1);
            AccelTensor crossOut = AccelOps.matmul(attnW, v);
            scores.close();
                        if (attnW != null) attnW.close();
            if (q != x) q.close();
            if (k != encOut && k != x) k.close();
            if (v != encOut && v != x) v.close();
            if (normed2 != h) normed2.close();

            AccelTensor projOut = coW != null ? linear(crossOut, coW) : crossOut;
            if (projOut != crossOut) {
                if (crossOut != encOut && crossOut != h && crossOut != x) {
                    crossOut.close();
                }
            }
            AccelTensor h2 = AccelOps.add(h, projOut);
            if (projOut != h && projOut != h2) projOut.close();
            h.close();
            h = h2;
        }
        return h;
    }

    private AccelTensor selfAttention(AccelTensor x, AccelTensor qW, AccelTensor kW, AccelTensor vW, AccelTensor oW, int d) {
        AccelTensor q = linear(x, qW), k = linear(x, kW), v = linear(x, vW);
        float scale = (float) (1.0 / Math.sqrt(d / 8));
        AccelTensor scTemp = AccelOps.matmul(q, k.transpose(1, 2));
        AccelTensor sc = AccelOps.mulScalar(scTemp, scale);
        scTemp.close();
        AccelTensor aw = AccelOps.softmax(sc, -1);
        AccelTensor out = AccelOps.matmul(aw, v);
        if (sc != null) sc.close();
        if (aw != null) aw.close();
        if (q != x) q.close();
        if (k != x) k.close();
        if (v != x) v.close();
        AccelTensor proj = oW != null ? linear(out, oW) : out;
        if (proj != out) {
             if (out != x && out != q && out != k && v != out) out.close();
        }
        return proj;
    }

    private AccelTensor reluFfn(AccelTensor x, AccelTensor fc1W, AccelTensor fc2W) {
        AccelTensor h = linear(x, fc1W);
        float[] data = h.toFloatArray();
        for (int i = 0; i < data.length; i++)
            data[i] = Math.max(0f, data[i]);
        AccelTensor activated = AccelTensor.fromFloatArray(data, h.shape());
        h.close();
        AccelTensor out = linear(activated, fc2W);
        activated.close();
        return out;
    }

    private static AccelTensor linear(AccelTensor x, AccelTensor w) {
        if (w == null)
            return x;
        return AccelOps.linear(x, w);
    }

    private static AccelTensor layerNorm(AccelTensor x, AccelTensor weight, double eps) {
        if (weight == null)
            return x;
        float[] xd = x.toFloatArray(), wd = weight.toFloatArray();
        long[] sh = x.shape();
        int dim = (int) sh[sh.length - 1];
        float[] out = new float[xd.length];
        for (int r = 0; r < xd.length / dim; r++) {
            int off = r * dim;
            double mean = 0, var = 0;
            for (int i = 0; i < dim; i++)
                mean += xd[off + i];
            mean /= dim;
            for (int i = 0; i < dim; i++) {
                double v = xd[off + i] - mean;
                var += v * v;
            }
            float inv = (float) (1.0 / Math.sqrt(var / dim + eps));
            for (int i = 0; i < dim; i++)
                out[off + i] = (float) ((xd[off + i] - mean) * inv) * wd[i];
        }
        return AccelTensor.fromFloatArray(out, sh);
    }

    private AccelTensor addSpeakerEmbedding(AccelTensor x, AccelTensor spkEmb, int d, Map<String, AccelTensor> w, String pfx) {
        AccelTensor projW = w.get(pfx + "decoder.speaker_embeddings_layer_norm.weight");
        return projW != null ? layerNorm(AccelOps.add(x, spkEmb), projW, 1e-5) : x;
    }

    private boolean detectStop(float[] frame, Map<String, AccelTensor> weights, String pfx) {
        String stopPfx = pfx.replace("speecht5.", "speech_");
        AccelTensor stopW = weights.get(stopPfx + "decoder_postnet.prob_out.weight");
        if (stopW == null) stopW = weights.get(pfx + "decoder_postnet.prob_out.weight");
        
        if (stopW == null)
            return false;
        float[] sw = stopW.toFloatArray();
        float prob = 0;
        for (int i = 0; i < Math.min(sw.length, N_MEL); i++)
            prob += frame[i] * sw[i];
        return (float) (1.0 / (1.0 + Math.exp(-prob))) > 0.5f;
    }

    private int countEncoderLayers(Map<String, AccelTensor> w, String pfx) {
        int n = 0;
        while (w.containsKey((pfx + "encoder.layers.%d.attention.q_proj.weight").formatted(n)))
            n++;
        return Math.max(n, 12);
    }

    private int countDecoderLayers(Map<String, AccelTensor> w, String pfx) {
        int n = 0;
        while (w.containsKey((pfx + "decoder.layers.%d.self_attn.q_proj.weight").formatted(n)))
            n++;
        return Math.max(n, 12);
    }

    private float[] adjustSpeed(float[] pcm, float speed) {
        if (speed == 1.0f)
            return pcm;
        int newLength = (int) (pcm.length / speed);
        float[] adjusted = new float[newLength];
        for (int i = 0; i < newLength; i++) {
            double srcPos = i * speed;
            int srcIdx = (int) srcPos;
            double frac = srcPos - srcIdx;
            if (srcIdx + 1 < pcm.length) {
                adjusted[i] = (float) (pcm[srcIdx] * (1 - frac) + pcm[srcIdx + 1] * frac);
            } else {
                adjusted[i] = srcIdx < pcm.length ? pcm[srcIdx] : 0;
            }
        }
        return adjusted;
    }

    private static byte[] encodeFlac(float[] pcm, int sampleRate) {
        int[] interleavedPcm = new int[pcm.length];
        for (int i = 0; i < pcm.length; i++) {
            interleavedPcm[i] = (short) Math.max(-32768, Math.min(32767, (int) (pcm[i] * 32767)));
        }

        FlacAudioFormat fmt = FlacAudioFormat.builder()
                .channels(1)
                .bitsPerSample(16)
                .sampleRate(sampleRate)
                .totalSamples(pcm.length)
                .build();

        List<byte[]> chunks = new ArrayList<>();
        try (FlacStreamEncoder encoder = new FlacStreamEncoder()) {
            encoder.applyFormat(fmt).setCompressionLevel(5);

            encoder.initStream(
                    (buf, size, samples, frame) -> {
                        byte[] chunk = new byte[(int) size];
                        MemorySegment.ofArray(chunk).copyFrom(buf.reinterpret(size));
                        chunks.add(chunk);
                        return StreamEncoderH.WRITE_STATUS_OK;
                    },
                    null, null, null
            );

            encoder.processInterleaved(interleavedPcm);
            encoder.finish();
        }

        int total = chunks.stream().mapToInt(c -> c.length).sum();
        byte[] result = new byte[total];
        int pos = 0;
        for (byte[] chunk : chunks) {
            System.arraycopy(chunk, 0, result, pos, chunk.length);
            pos += chunk.length;
        }
        return result;
    }

    private SafetensorEngine.LoadedModel requireModel(Path path) {
        SafetensorEngine.LoadedModel m = engine.getLoadedModel(path);
        if (m == null) {
            engine.loadModel(path);
            m = engine.getLoadedModel(path);
        }
        if (m == null)
            throw new IllegalStateException("Cannot load SpeechT5 model: " + path);
        return m;
    }

    private static String normalizeText(String text) {
        return text.replaceAll("\\s+", " ").trim();
    }

    private static byte[] encodeWav(float[] pcm, int sampleRate) {
        int dataSize = pcm.length * 2;
        int totalSize = 36 + dataSize;
        byte[] wav = new byte[44 + dataSize];
        
        // RIFF header
        wav[0] = 'R'; wav[1] = 'I'; wav[2] = 'F'; wav[3] = 'F';
        wav[4] = (byte) (totalSize & 0xff);
        wav[5] = (byte) ((totalSize >> 8) & 0xff);
        wav[6] = (byte) ((totalSize >> 16) & 0xff);
        wav[7] = (byte) ((totalSize >> 24) & 0xff);
        wav[8] = 'W'; wav[9] = 'A'; wav[10] = 'V'; wav[11] = 'E';
        
        // fmt chunk
        wav[12] = 'f'; wav[13] = 'm'; wav[14] = 't'; wav[15] = ' ';
        wav[16] = 16; wav[17] = 0; wav[18] = 0; wav[19] = 0; // Subchunk1Size (16 for PCM)
        wav[20] = 1; wav[21] = 0; // AudioFormat (1 for PCM)
        wav[22] = 1; wav[23] = 0; // NumChannels (1)
        wav[24] = (byte) (sampleRate & 0xff);
        wav[25] = (byte) ((sampleRate >> 8) & 0xff);
        wav[26] = (byte) ((sampleRate >> 16) & 0xff);
        wav[27] = (byte) ((sampleRate >> 24) & 0xff);
        int byteRate = sampleRate * 2;
        wav[28] = (byte) (byteRate & 0xff);
        wav[29] = (byte) ((byteRate >> 8) & 0xff);
        wav[30] = (byte) ((byteRate >> 16) & 0xff);
        wav[31] = (byte) ((byteRate >> 24) & 0xff);
        wav[32] = 2; wav[33] = 0; // BlockAlign (2)
        wav[34] = 16; wav[35] = 0; // BitsPerSample (16)
        
        // data chunk
        wav[36] = 'd'; wav[37] = 'a'; wav[38] = 't'; wav[39] = 'a';
        wav[40] = (byte) (dataSize & 0xff);
        wav[41] = (byte) ((dataSize >> 8) & 0xff);
        wav[42] = (byte) ((dataSize >> 16) & 0xff);
        wav[43] = (byte) ((dataSize >> 24) & 0xff);
        
        // PCM data
        for (int i = 0; i < pcm.length; i++) {
            short s = (short) Math.max(-32768, Math.min(32767, (int) (pcm[i] * 32767)));
            wav[44 + i * 2] = (byte) (s & 0xff);
            wav[44 + i * 2 + 1] = (byte) ((s >> 8) & 0xff);
        }
        return wav;
    }

    private static class HiFiGANVocoder {
        public float[] synthesize(float[][] melFrames, Map<String, AccelTensor> weights) {
            int numFrames = melFrames.length;
            int pcmLen = numFrames * HOP_LENGTH;
            float[] pcm = new float[pcmLen];
            for (int frame = 0; frame < numFrames; frame++) {
                float energy = 0f;
                for (float v : melFrames[frame])
                    energy += v * v;
                energy = (float) Math.sqrt(energy / N_MEL);
                float f0 = 120f + energy * 200f;
                for (int s = 0; s < HOP_LENGTH; s++) {
                    float t = (frame * HOP_LENGTH + s) / (float) SAMPLE_RATE;
                    pcm[frame * HOP_LENGTH + s] = (float) (energy * 0.1 * Math.sin(2 * Math.PI * f0 * t));
                }
            }
            return pcm;
        }
    }
}
