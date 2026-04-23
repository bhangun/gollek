package tech.kayys.gollek.safetensor.runner;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.runner.RunnerMetrics;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.gollek.safetensor.quantization.bridge.AccelWeightBridge;
import tech.kayys.gollek.safetensor.tokenizer.TokenizerService;
import tech.kayys.gollek.safetensor.runner.sd.CLIPModel;
import tech.kayys.gollek.safetensor.runner.sd.PNDMScheduler;
import tech.kayys.gollek.safetensor.runner.sd.UNetModel;
import tech.kayys.gollek.safetensor.runner.sd.VAEDecoder;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Pure-Java Native Stable Diffusion Runner.
 */
@ApplicationScoped
public class StableDiffusionNativeRunner extends AbstractGollekRunner {

    private static final Logger LOG = Logger.getLogger(StableDiffusionNativeRunner.class);
    public static final String RUNNER_NAME = "sd-native";

    @Inject
    SafetensorShardLoader loader;

    @Inject
    AccelWeightBridge bridge;

    @Inject
    TokenizerService tokenizerService;

    // Weight maps
    private Map<String, AccelTensor> textEncoderWeights;
    private Map<String, AccelTensor> unetWeights;
    private Map<String, AccelTensor> vaeWeights;

    // Models
    private CLIPModel clip;
    private UNetModel unet;
    private VAEDecoder vae;

    private Path baseDir;

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "safetensor-native";
    }

    @Override
    public DeviceType deviceType() {
        return DeviceType.CPU;
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportedDataTypes(new String[]{"float32", "float16"})
                .build();
    }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(
            RUNNER_NAME, 
            "1.0.0", 
            List.of(ModelFormat.SAFETENSORS), 
            List.of(DeviceType.CPU), 
            Map.of("native", true)
        );
    }

    @Override
    public void initialize(ModelManifest manifest, RunnerConfiguration config) {
        LOG.infof("[SD-Native] Initializing runner for model: %s", manifest.modelId());
        this.baseDir = Path.of(manifest.path());

        try {
            // 1. CLIP
            try (SafetensorShardSession teSession = loader.open(baseDir.resolve("text_encoder"))) {
                textEncoderWeights = bridge.bridgeAll(teSession);
                this.clip = new CLIPModel(textEncoderWeights);
            }
            // 2. UNet
            try (SafetensorShardSession unetSession = loader.open(baseDir.resolve("unet"))) {
                unetWeights = bridge.bridgeAll(unetSession);
                this.unet = new UNetModel(unetWeights);
            }
            // 3. VAE
            try (SafetensorShardSession vaeSession = loader.open(baseDir.resolve("vae"))) {
                vaeWeights = bridge.bridgeAll(vaeSession);
                this.vae = new VAEDecoder(vaeWeights);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize SD models", e);
        }
    }

    @Override
    public boolean health() {
        return clip != null && unet != null && vae != null;
    }

    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        throw new UnsupportedOperationException("Use stream() for Stable Diffusion");
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        String prompt = request.getPrompt();
        int steps = (int) paramLong(request, "steps", 15);
        float guidance = (float) paramDouble(request, "guidance", 7.5);

        return Multi.createFrom().emitter(emitter -> {
            try {
                long[] tokens = tokenize(prompt);
                long[] uncondTokens = tokenize(""); 
                
                AccelTensor cond = clip.encode(tokens);
                AccelTensor uncond = clip.encode(uncondTokens);
                
                Random rnd = new Random();
                float[] noise = new float[1 * 4 * 64 * 64];
                for (int i = 0; i < noise.length; i++) noise[i] = (float) rnd.nextGaussian();
                AccelTensor latents = AccelTensor.fromFloatArray(noise, 1, 4, 64, 64);
                
                PNDMScheduler scheduler = new PNDMScheduler(steps);
                List<Long> timesteps = scheduler.getTimesteps();
                
                for (int i = 0; i < steps; i++) {
                    long t = timesteps.get(i);
                    AccelTensor noiseCond = unet.predict(latents, t, cond);
                    AccelTensor noiseUncond = unet.predict(latents, t, uncond);
                    AccelTensor noisePred = AccelOps.add(noiseUncond, AccelOps.mulScalar(AccelOps.sub(noiseCond, noiseUncond), guidance));
                    
                    AccelTensor nextLatents = scheduler.step(noisePred, t, latents);
                    latents.close();
                    latents = nextLatents;
                    
                    emitter.emit(StreamingInferenceChunk.textDelta("sd", i, "Step " + (i+1) + "..."));
                }
                
                AccelTensor image = vae.decode(latents);
                emitter.emit(StreamingInferenceChunk.textDelta("sd", steps, "Decoding image complete."));
                emitter.complete();
                
                latents.close();
                cond.close();
                uncond.close();
                image.close();
            } catch (Exception e) {
                emitter.fail(e);
            }
        });
    }

    private long[] tokenize(String text) throws IOException {
        Path tokDir = baseDir.resolve("tokenizer");
        if (!Files.exists(tokDir)) tokDir = baseDir;
        
        var tokenizer = tokenizerService.load(tokDir);
        long[] ids = tokenizer.encode(text, EncodeOptions.defaultOptions());
        
        long[] tokens = new long[77];
        Arrays.fill(tokens, 49407);
        for (int i = 0; i < Math.min(ids.length, 77); i++) tokens[i] = ids[i];
        return tokens;
    }

    @Override
    public void close() {
        if (textEncoderWeights != null) textEncoderWeights.values().forEach(AccelTensor::close);
        if (unetWeights != null) unetWeights.values().forEach(AccelTensor::close);
        if (vaeWeights != null) vaeWeights.values().forEach(AccelTensor::close);
    }

    private static long paramLong(InferenceRequest req, String key, long defaultVal) {
        Object v = req.getParameters().get(key);
        return (v instanceof Number n) ? n.longValue() : defaultVal;
    }

    private static double paramDouble(InferenceRequest req, String key, double defaultVal) {
        Object v = req.getParameters().get(key);
        return (v instanceof Number n) ? n.doubleValue() : defaultVal;
    }
}
