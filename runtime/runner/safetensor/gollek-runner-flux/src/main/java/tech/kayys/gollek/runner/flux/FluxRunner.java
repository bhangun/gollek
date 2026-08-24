package tech.kayys.gollek.runner.flux;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.core.model.ModelFormat;
import tech.kayys.alkhawarizm.core.random.GaussianNoise;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.models.flux.FluxModelArchitecture;
import tech.kayys.alkhawarizm.models.flux.FluxVariant;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.safetensor.loader.SafetensorShardLoader;
import tech.kayys.alkhawarizm.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.alkhawarizm.safetensor.quantization.bridge.AccelWeightBridge;
import tech.kayys.alkhawarizm.spi.model.ModelManifest;
import tech.kayys.alkhawarizm.spi.model.RunnerMetadata;
import tech.kayys.gollek.exception.RunnerInitializationException;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.spi.image.GeneratedImage;
import tech.kayys.gollek.spi.image.ImageGenProgress;
import tech.kayys.gollek.spi.image.ImageGenRequest;
import tech.kayys.gollek.spi.image.ImageGenerationPipeline;
import tech.kayys.gollek.spi.image.PipelineCapability;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * FLUX.1 / FLUX.2-klein SafeTensors inference runner for Gollek.
 */
@ApplicationScoped
public class FluxRunner extends AbstractGollekRunner implements ImageGenerationPipeline {

    private static final Logger LOG = Logger.getLogger(FluxRunner.class);
    public static final String RUNNER_NAME = "flux-safetensor";

    private static final int DEFAULT_LATENT_CHANNELS = FluxModelArchitecture.LATENT_CHANNELS;
    private static final float VAE_SCALE = FluxModelArchitecture.VAE_SCALE_FACTOR;

    @Inject
    SafetensorShardLoader loader;

    @Inject
    AccelWeightBridge bridge;

    private FluxDualTextEncoder textEncoder;
    private FluxDiTDenoiser transformer;
    private FluxVaeDecoder vaeDecoder;

    private FluxVariant variant;
    private Path baseDir;

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "safetensor-flux";
    }

    @Override
    public DeviceType deviceType() {
        return DeviceType.CPU;
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportedDataTypes(new String[]{"float32", "bfloat16", "float16"})
                .build();
    }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(
                RUNNER_NAME,
                "1.0.0",
                List.of(ModelFormat.SAFETENSORS),
                List.of(DeviceType.CPU),
                Map.of(
                        "architecture", "flux-mm-dit",
                        "textEncoders", "clip-l+t5-xxl",
                        "latentChannels", DEFAULT_LATENT_CHANNELS,
                        "scheduler", "flow_matching_euler",
                        "vaeScaleFactor", VAE_SCALE,
                        "variant", variant != null ? variant.modelCode() : "unknown"
                ));
    }

    @Override
    public String pipelineId() {
        return variant != null ? "flux-" + variant.modelCode() : RUNNER_NAME;
    }

    @Override
    public Set<PipelineCapability> pipelineCapabilities() {
        return EnumSet.of(PipelineCapability.TEXT_TO_IMAGE);
    }

    @Override
    public ImageGenRequest defaultRequest(String prompt) {
        int steps = variant != null ? variant.defaultSteps() : 20;
        float guidance = variant != null ? variant.defaultGuidanceScale() : 3.5f;
        return ImageGenRequest.builder()
                .prompt(prompt)
                .width(1024).height(1024)
                .steps(steps)
                .guidanceScale(guidance)
                .scheduler("flow_euler")
                .outputFormat("png")
                .build();
    }

    @Override
    public GeneratedImage generate(ImageGenRequest request) throws InferenceException {
        LOG.infof("[FLUX] Generating image: prompt='%s', size=%dx%d, steps=%d, seed=%d, variant=%s",
                request.prompt(), request.width(), request.height(),
                request.steps(), request.seed(),
                variant != null ? variant.modelCode() : "unknown");

        long t0 = System.currentTimeMillis();

        try {
            FluxDualTextEncoder.DualEmbeddings embeddings = textEncoder.encode(request.prompt());

            int latentH = request.height() / 8;
            int latentW = request.width() / 8;
            float[] noise = new float[1 * DEFAULT_LATENT_CHANNELS * latentH * latentW];
            GaussianNoise.fill(noise, new Random(request.seed()));
            AccelTensor latents = AccelTensor.fromFloatArray(noise, 1, DEFAULT_LATENT_CHANNELS, latentH, latentW);

            FlowEulerScheduler scheduler = new FlowEulerScheduler(request.steps());

            float[] timesteps = scheduler.timesteps();
            for (int i = 0; i < request.steps(); i++) {
                float t = timesteps[i];
                float guidance = (variant != null && !variant.requiresGuidanceEmbedding())
                        ? 1.0f : request.guidanceScale();

                AccelTensor velocity = transformer.predict(
                        latents,
                        embeddings.t5Hidden(),
                        embeddings.clipPooled(),
                        t,
                        guidance);

                AccelTensor nextLatents = scheduler.step(latents, velocity, i);
                latents.close();
                latents = nextLatents;
                velocity.close();
            }

            embeddings.close();

            float[] pixels = vaeDecoder.decode(latents, request.width(), request.height());
            latents.close();

            byte[] png = FluxVaeDecoder.toPng(pixels, request.width(), request.height());
            long elapsed = System.currentTimeMillis() - t0;

            return GeneratedImage.of(
                    request.requestId(),
                    png, "image/png",
                    request.width(), request.height(),
                    pipelineId(), elapsed,
                    Map.of(
                            "steps", request.steps(),
                            "seed", request.seed(),
                            "scheduler", "flow_euler",
                            "variant", variant != null ? variant.modelCode() : "unknown"
                    ));

        } catch (Exception e) {
            throw new InferenceException("FLUX generation failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Multi<ImageGenProgress> generateStreaming(ImageGenRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                FluxDualTextEncoder.DualEmbeddings embeddings = textEncoder.encode(request.prompt());

                int latentH = request.height() / 8;
                int latentW = request.width() / 8;
                float[] noise = new float[1 * DEFAULT_LATENT_CHANNELS * latentH * latentW];
                GaussianNoise.fill(noise, new Random(request.seed()));
                AccelTensor latents = AccelTensor.fromFloatArray(noise, 1, DEFAULT_LATENT_CHANNELS, latentH, latentW);

                FlowEulerScheduler scheduler = new FlowEulerScheduler(request.steps());
                float[] timesteps = scheduler.timesteps();

                emitter.emit(ImageGenProgress.step(request.requestId(), 0, request.steps(), "Encoding prompt..."));

                for (int i = 0; i < request.steps(); i++) {
                    float t = timesteps[i];
                    float guidance = (variant != null && !variant.requiresGuidanceEmbedding())
                            ? 1.0f : request.guidanceScale();

                    AccelTensor velocity = transformer.predict(
                            latents, embeddings.t5Hidden(), embeddings.clipPooled(), t, guidance);
                    AccelTensor next = scheduler.step(latents, velocity, i);
                    latents.close();
                    latents = next;
                    velocity.close();

                    emitter.emit(ImageGenProgress.step(
                            request.requestId(), i + 1, request.steps(),
                            String.format("Step %d/%d (t=%.3f)", i + 1, request.steps(), t)));
                }

                embeddings.close();
                float[] pixels = vaeDecoder.decode(latents, request.width(), request.height());
                latents.close();

                byte[] png = FluxVaeDecoder.toPng(pixels, request.width(), request.height());
                emitter.emit(ImageGenProgress.stepWithPreview(
                        request.requestId(), request.steps(), request.steps(), "Done", png));
                emitter.complete();

            } catch (Exception e) {
                emitter.fail(e);
            }
        });
    }

    @Override
    public void initialize(ModelManifest manifest, RunnerConfiguration config)
            throws RunnerInitializationException {
        this.baseDir = Path.of(manifest.path());
        this.variant = detectVariant(manifest.modelId());

        LOG.infof("[FLUX] Initializing runner for: %s (variant=%s)", manifest.modelId(), variant);

        try {
            Map<String, AccelTensor> clipWeights, t5Weights, transformerWeights, vaeWeights;

            try (SafetensorShardSession s = loader.open(baseDir.resolve("text_encoder"))) {
                clipWeights = bridge.bridgeAll(s);
            }
            try (SafetensorShardSession s = loader.open(baseDir.resolve("text_encoder_2"))) {
                t5Weights = bridge.bridgeAll(s);
            }
            try (SafetensorShardSession s = loader.open(baseDir.resolve("transformer"))) {
                transformerWeights = bridge.bridgeAll(s);
            }
            try (SafetensorShardSession s = loader.open(baseDir.resolve("vae"))) {
                vaeWeights = bridge.bridgeAll(s);
            }

            this.textEncoder = new FluxDualTextEncoder(baseDir, clipWeights, t5Weights);
            this.transformer = new FluxDiTDenoiser(transformerWeights, variant);
            this.vaeDecoder = new FluxVaeDecoder(vaeWeights);

            this.initialized = true;
            LOG.infof("[FLUX] Runner ready — variant=%s", variant.modelCode());

        } catch (Exception e) {
            throw new RunnerInitializationException(
                    tech.kayys.alkhawarizm.error.ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "Failed to load FLUX weights from " + baseDir + ": " + e.getMessage());
        }
    }

    private static FluxVariant detectVariant(String modelId) {
        if (modelId == null) return FluxVariant.FLUX_1_DEV;
        String id = modelId.toLowerCase();
        if (id.contains("schnell")) return FluxVariant.FLUX_1_SCHNELL;
        if (id.contains("klein")) return FluxVariant.FLUX_2_KLEIN_9B;
        return FluxVariant.FLUX_1_DEV;
    }

    @Override
    public boolean health() {
        return textEncoder != null && transformer != null && vaeDecoder != null;
    }

    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        throw new UnsupportedOperationException(
                "Use generate(ImageGenRequest) or generateStreaming(ImageGenRequest) for FLUX image generation.");
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().failure(new UnsupportedOperationException(
                "Use generateStreaming(ImageGenRequest) directly on FluxRunner."));
    }

    @Override
    public void close() {
        if (textEncoder != null) textEncoder.close();
        if (transformer != null) transformer.close();
        if (vaeDecoder != null) vaeDecoder.close();
    }
}
