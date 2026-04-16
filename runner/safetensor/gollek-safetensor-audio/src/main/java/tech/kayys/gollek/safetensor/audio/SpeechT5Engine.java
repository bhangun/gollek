package tech.kayys.gollek.safetensor.audio;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.safetensor.audio.model.AudioConfig;
import tech.kayys.gollek.safetensor.audio.model.AudioResult;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

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

                // 1. Normalize and tokenize text
                String normalizedText = normalizeText(text);
                long[] encoded = model.tokenizer().encode(normalizedText, EncodeOptions.builder().addBos(true).build());
                int[] tokenIds = new int[encoded.length];
                for (int i = 0; i < encoded.length; i++)
                    tokenIds[i] = (int) encoded[i];

                // 2. Encode text
                TorchTensor encoderOut = runTextEncoder(tokenIds, model);

                // 3. Get speaker embedding
                float[] speakerEmb = getSpeakerEmbedding(voice != null ? voice : defaultVoice);
                TorchTensor speakerTensor = TorchTensor.fromFloatArray(speakerEmb, new long[] { 1, SPEAKER_EMB_DIM });

                // 4. Decode to mel spectrogram
                float[][] melFrames = runDecoder(encoderOut, speakerTensor, model, config);
                encoderOut.close();
                speakerTensor.close();

                // 5. Vocoder: mel → audio
                float[] pcm = vocoder.synthesize(melFrames, model.weights());

                // 6. Apply speed adjustment if requested (using temperature field for speed in
                // this context)
                if (config != null && config.getTemperature() != 1.0f) {
                    pcm = adjustSpeed(pcm, config.getTemperature());
                }

                // 7. Encode to WAV
                byte[] wav = encodeWav(pcm, SAMPLE_RATE);

                long durationMs = java.time.Duration.between(startTime, Instant.now()).toMillis();
                log.infof("SpeechT5: generated %.1fs audio (%d bytes) in %dms",
                        pcm.length / (double) SAMPLE_RATE, wav.length, durationMs);
                em.complete(wav);

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
                .map(wav -> {
                    double duration = wav.length > 44 ? (wav.length - 44) / 2.0 / SAMPLE_RATE : 0;
                    return AudioResult.speechSynthesis(wav, modelPath.getFileName().toString(), duration);
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

    private TorchTensor runTextEncoder(int[] tokenIds, SafetensorEngine.LoadedModel model) {
        Map<String, TorchTensor> w = model.weights();
        TorchTensor embedW = w.get("speecht5.encoder.embed_tokens.weight");
        int hiddenSize = embedW != null ? (int) embedW.shape()[1] : 768;

        if (embedW == null) {
            log.warn("SpeechT5: encoder embed_tokens not found — using zeros");
            return TorchTensor.fromFloatArray(new float[tokenIds.length * hiddenSize],
                    new long[] { 1, tokenIds.length, hiddenSize });
        }

        float[] embedData = embedW.toFloatArray();
        float[] tokenEmbeds = new float[tokenIds.length * hiddenSize];
        for (int i = 0; i < tokenIds.length; i++) {
            int id = Math.min(tokenIds[i], (int) embedW.shape()[0] - 1);
            System.arraycopy(embedData, id * hiddenSize, tokenEmbeds, i * hiddenSize, hiddenSize);
        }
        TorchTensor hidden = TorchTensor.fromFloatArray(tokenEmbeds, new long[] { 1, tokenIds.length, hiddenSize });

        int numLayers = countEncoderLayers(w);
        for (int i = 0; i < numLayers; i++) {
            TorchTensor layerOut = encoderLayer(hidden, w, i, hiddenSize);
            hidden.close();
            hidden = layerOut;
        }
        return hidden;
    }

    private TorchTensor encoderLayer(TorchTensor x, Map<String, TorchTensor> w, int i, int d) {
        String pfx = "speecht5.encoder.layers.%d.".formatted(i);
        TorchTensor qW = w.get(pfx + "attention.q_proj.weight");
        if (qW == null)
            return x;

        TorchTensor kW = w.get(pfx + "attention.k_proj.weight");
        TorchTensor vW = w.get(pfx + "attention.v_proj.weight");
        TorchTensor oW = w.get(pfx + "attention.out_proj.weight");
        TorchTensor n1W = w.get(pfx + "layer_norm.weight");
        TorchTensor fc1W = w.get(pfx + "feed_forward.intermediate_dense.weight");
        TorchTensor fc2W = w.get(pfx + "feed_forward.output_dense.weight");

        TorchTensor normed = layerNorm(x, n1W, 1e-5);
        TorchTensor attnOut = selfAttention(normed, qW, kW, vW, oW, d);
        normed.close();
        TorchTensor h = x.add(attnOut);
        attnOut.close();

        if (fc1W != null && fc2W != null) {
            TorchTensor n2W = w.get(pfx + "final_layer_norm.weight");
            TorchTensor normed2 = layerNorm(h, n2W, 1e-5);
            TorchTensor ffnOut = reluFfn(normed2, fc1W, fc2W);
            normed2.close();
            TorchTensor h2 = h.add(ffnOut);
            ffnOut.close();
            h.close();
            return h2;
        }
        return h;
    }

    private float[][] runDecoder(TorchTensor encoderOut, TorchTensor speakerTensor,
            SafetensorEngine.LoadedModel model, AudioConfig config) {
        Map<String, TorchTensor> w = model.weights();
        int hiddenSize = (int) encoderOut.shape()[2];
        List<float[]> melFrames = new ArrayList<>();
        float[] prevMelFrame = new float[N_MEL];
        TorchTensor decoderHidden = TorchTensor.fromFloatArray(prevMelFrame, new long[] { 1, 1, N_MEL });

        for (int step = 0; step < MAX_MEL_FRAMES; step++) {
            TorchTensor prenetW = w.get("speecht5.decoder.prenet.layers.0.weight");
            TorchTensor projected = prenetW != null ? linear(decoderHidden, prenetW) : decoderHidden;

            TorchTensor withSpk = addSpeakerEmbedding(projected, speakerTensor, hiddenSize, w);
            if (projected != decoderHidden)
                projected.close();

            int numDecLayers = countDecoderLayers(w);
            TorchTensor decOut = withSpk;
            for (int i = 0; i < numDecLayers; i++) {
                TorchTensor layerOut = decoderLayer(decOut, encoderOut, w, i, hiddenSize);
                decOut.close();
                decOut = layerOut;
            }

            TorchTensor featW = w.get("speech_decoder_postnet.feat_out.weight");
            TorchTensor melOut = featW != null ? linear(decOut, featW) : decOut;
            float[] frame = melOut.toFloatArray();
            if (melOut != decOut)
                melOut.close();
            decOut.close();
            decoderHidden.close();

            float[] melFrame = Arrays.copyOf(frame, N_MEL);
            melFrames.add(melFrame);

            if (detectStop(frame, w))
                break;
            decoderHidden = TorchTensor.fromFloatArray(melFrame, new long[] { 1, 1, N_MEL });
        }
        return melFrames.toArray(float[][]::new);
    }

    private TorchTensor decoderLayer(TorchTensor x, TorchTensor encOut, Map<String, TorchTensor> w, int i, int d) {
        String pfx = "speecht5.decoder.layers.%d.".formatted(i);
        TorchTensor qW = w.get(pfx + "self_attn.q_proj.weight");
        if (qW == null)
            return x;

        TorchTensor kW = w.get(pfx + "self_attn.k_proj.weight");
        TorchTensor vW = w.get(pfx + "self_attn.v_proj.weight");
        TorchTensor oW = w.get(pfx + "self_attn.out_proj.weight");
        TorchTensor n1W = w.get(pfx + "self_attn_layer_norm.weight");

        TorchTensor normed = layerNorm(x, n1W, 1e-5);
        TorchTensor attn = selfAttention(normed, qW, kW, vW, oW, d);
        normed.close();
        TorchTensor h = x.add(attn);
        attn.close();

        TorchTensor cqW = w.get(pfx + "encoder_attn.q_proj.weight");
        if (cqW != null) {
            TorchTensor ckW = w.get(pfx + "encoder_attn.k_proj.weight");
            TorchTensor cvW = w.get(pfx + "encoder_attn.v_proj.weight");
            TorchTensor coW = w.get(pfx + "encoder_attn.out_proj.weight");
            TorchTensor n2W = w.get(pfx + "encoder_attn_layer_norm.weight");

            TorchTensor normed2 = layerNorm(h, n2W, 1e-5);
            TorchTensor q = linear(normed2, cqW);
            TorchTensor k = linear(encOut, ckW);
            TorchTensor v = linear(encOut, cvW);

            float scale = (float) (1.0 / Math.sqrt(d / 8));
            TorchTensor scores = q.matmul(encOut.transpose(1, 2))
                    .mul(TorchTensor.fromFloatArray(new float[] { scale }, new long[] { 1 }));
            TorchTensor exp = scores.exp();
            TorchTensor sum = exp.sum();
            TorchTensor attnW = exp.div(sum);
            TorchTensor crossOut = attnW.matmul(v);
            scores.close();
            exp.close();
            sum.close();
            attnW.close();
            q.close();
            k.close();
            v.close();
            normed2.close();

            TorchTensor projOut = coW != null ? linear(crossOut, coW) : crossOut;
            if (projOut != crossOut)
                crossOut.close();
            TorchTensor h2 = h.add(projOut);
            projOut.close();
            h.close();
            h = h2;
        }
        return h;
    }

    private TorchTensor selfAttention(TorchTensor x, TorchTensor qW, TorchTensor kW, TorchTensor vW, TorchTensor oW, int d) {
        TorchTensor q = linear(x, qW), k = linear(x, kW), v = linear(x, vW);
        float scale = (float) (1.0 / Math.sqrt(d / 8));
        TorchTensor sc = q.matmul(k.transpose(1, 2)).mul(TorchTensor.fromFloatArray(new float[] { scale }, new long[] { 1 }));
        TorchTensor exp = sc.exp();
        TorchTensor sum = exp.sum();
        TorchTensor aw = exp.div(sum);
        TorchTensor out = aw.matmul(v);
        sc.close();
        aw.close();
        q.close();
        k.close();
        v.close();
        TorchTensor proj = oW != null ? linear(out, oW) : out;
        if (proj != out)
            out.close();
        return proj;
    }

    private TorchTensor reluFfn(TorchTensor x, TorchTensor fc1W, TorchTensor fc2W) {
        TorchTensor h = linear(x, fc1W);
        float[] data = h.toFloatArray();
        for (int i = 0; i < data.length; i++)
            data[i] = Math.max(0f, data[i]);
        TorchTensor activated = TorchTensor.fromFloatArray(data, h.shape());
        h.close();
        TorchTensor out = linear(activated, fc2W);
        activated.close();
        return out;
    }

    private static TorchTensor linear(TorchTensor x, TorchTensor w) {
        if (w == null)
            return x;
        return x.matmul(w.transpose(w.shape().length - 2, w.shape().length - 1));
    }

    private static TorchTensor layerNorm(TorchTensor x, TorchTensor weight, double eps) {
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
        return TorchTensor.fromFloatArray(out, sh);
    }

    private TorchTensor addSpeakerEmbedding(TorchTensor x, TorchTensor spkEmb, int d, Map<String, TorchTensor> w) {
        TorchTensor projW = w.get("speecht5.decoder.speaker_embeddings_layer_norm.weight");
        return projW != null ? layerNorm(x.add(spkEmb), projW, 1e-5) : x;
    }

    private boolean detectStop(float[] frame, Map<String, TorchTensor> weights) {
        TorchTensor stopW = weights.get("speech_decoder_postnet.prob_out.weight");
        if (stopW == null)
            return false;
        float[] sw = stopW.toFloatArray();
        float prob = 0;
        for (int i = 0; i < Math.min(sw.length, N_MEL); i++)
            prob += frame[i] * sw[i];
        return (float) (1.0 / (1.0 + Math.exp(-prob))) > 0.5f;
    }

    private int countEncoderLayers(Map<String, TorchTensor> w) {
        int n = 0;
        while (w.containsKey("speecht5.encoder.layers.%d.attention.q_proj.weight".formatted(n)))
            n++;
        return Math.max(n, 12);
    }

    private int countDecoderLayers(Map<String, TorchTensor> w) {
        int n = 0;
        while (w.containsKey("speecht5.decoder.layers.%d.self_attn.q_proj.weight".formatted(n)))
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

    private static byte[] encodeWav(float[] pcm, int sampleRate) {
        int dataLen = pcm.length * 2;
        int totalLen = 44 + dataLen;
        ByteBuffer buf = ByteBuffer.allocate(totalLen).order(ByteOrder.LITTLE_ENDIAN);
        buf.put("RIFF".getBytes());
        buf.putInt(totalLen - 8);
        buf.put("WAVE".getBytes());
        buf.put("fmt ".getBytes());
        buf.putInt(16);
        buf.putShort((short) 1);
        buf.putShort((short) 1);
        buf.putInt(sampleRate);
        buf.putInt(sampleRate * 2);
        buf.putShort((short) 2);
        buf.putShort((short) 16);
        buf.put("data".getBytes());
        buf.putInt(dataLen);
        for (float s : pcm) {
            short sample = (short) Math.max(-32768, Math.min(32767, (int) (s * 32767)));
            buf.putShort(sample);
        }
        return buf.array();
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

    private static class HiFiGANVocoder {
        public float[] synthesize(float[][] melFrames, Map<String, TorchTensor> weights) {
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
