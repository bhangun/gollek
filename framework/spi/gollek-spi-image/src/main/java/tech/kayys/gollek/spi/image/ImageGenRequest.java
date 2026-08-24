package tech.kayys.gollek.spi.image;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Strongly typed image generation, editing, and inpainting request.
 * Follows SOLID principles: single model parameter encapsulation with immutability.
 */
public record ImageGenRequest(
        String requestId,
        String prompt,
        String negativePrompt,
        int width,
        int height,
        int steps,
        float guidanceScale,
        long seed,
        String scheduler,
        String outputFormat,
        float strength,
        Optional<byte[]> initImage,
        Optional<byte[]> maskImage,
        Map<String, Object> extras) {

    public ImageGenRequest {
        requestId = (requestId == null || requestId.isBlank())
                ? java.util.UUID.randomUUID().toString()
                : requestId;
        prompt = Objects.requireNonNullElse(prompt, "");
        negativePrompt = Objects.requireNonNullElse(negativePrompt, "");
        width = width <= 0 ? 512 : width;
        height = height <= 0 ? 512 : height;
        steps = steps <= 0 ? 20 : steps;
        guidanceScale = guidanceScale <= 0f ? 7.5f : guidanceScale;
        seed = seed == 0L ? System.nanoTime() : seed;
        scheduler = (scheduler == null || scheduler.isBlank()) ? "default" : scheduler;
        outputFormat = (outputFormat == null || outputFormat.isBlank()) ? "png" : outputFormat.toLowerCase();
        strength = strength <= 0.0f || strength > 1.0f ? 0.75f : strength;
        initImage = initImage == null ? Optional.empty() : initImage;
        maskImage = maskImage == null ? Optional.empty() : maskImage;
        extras = extras == null ? Collections.emptyMap() : Collections.unmodifiableMap(extras);
    }

    public boolean isImageToImage() {
        return initImage.isPresent() && maskImage.isEmpty();
    }

    public boolean isInpainting() {
        return initImage.isPresent() && maskImage.isPresent();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private String prompt = "";
        private String negativePrompt = "";
        private int width = 512;
        private int height = 512;
        private int steps = 20;
        private float guidanceScale = 7.5f;
        private long seed = 0L;
        private String scheduler = "default";
        private String outputFormat = "png";
        private float strength = 0.75f;
        private byte[] initImage;
        private byte[] maskImage;
        private Map<String, Object> extras = Collections.emptyMap();

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder prompt(String prompt) {
            this.prompt = prompt;
            return this;
        }

        public Builder negativePrompt(String negativePrompt) {
            this.negativePrompt = negativePrompt;
            return this;
        }

        public Builder width(int width) {
            this.width = width;
            return this;
        }

        public Builder height(int height) {
            this.height = height;
            return this;
        }

        public Builder dimensions(int width, int height) {
            this.width = width;
            this.height = height;
            return this;
        }

        public Builder steps(int steps) {
            this.steps = steps;
            return this;
        }

        public Builder guidanceScale(float guidanceScale) {
            this.guidanceScale = guidanceScale;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder scheduler(String scheduler) {
            this.scheduler = scheduler;
            return this;
        }

        public Builder outputFormat(String outputFormat) {
            this.outputFormat = outputFormat;
            return this;
        }

        public Builder strength(float strength) {
            this.strength = strength;
            return this;
        }

        public Builder initImage(byte[] initImage) {
            this.initImage = initImage;
            return this;
        }

        public Builder maskImage(byte[] maskImage) {
            this.maskImage = maskImage;
            return this;
        }

        public Builder extras(Map<String, Object> extras) {
            this.extras = extras;
            return this;
        }

        public ImageGenRequest build() {
            return new ImageGenRequest(
                    requestId, prompt, negativePrompt,
                    width, height, steps, guidanceScale,
                    seed, scheduler, outputFormat, strength,
                    Optional.ofNullable(initImage),
                    Optional.ofNullable(maskImage),
                    extras);
        }
    }
}
