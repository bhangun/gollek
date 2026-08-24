package tech.kayys.gollek.spi.image;

import java.util.Optional;

/**
 * Progress event emitted during iterative image generation / diffusion loops.
 */
public record ImageGenProgress(
        String requestId,
        int currentStep,
        int totalSteps,
        String message,
        Optional<byte[]> previewLatent) {

    public ImageGenProgress {
        previewLatent = previewLatent == null ? Optional.empty() : previewLatent;
    }

    public static ImageGenProgress step(String requestId, int currentStep, int totalSteps, String message) {
        return new ImageGenProgress(requestId, currentStep, totalSteps, message, Optional.empty());
    }

    public static ImageGenProgress stepWithPreview(
            String requestId,
            int currentStep,
            int totalSteps,
            String message,
            byte[] preview) {
        return new ImageGenProgress(requestId, currentStep, totalSteps, message, Optional.ofNullable(preview));
    }

    public double progressPercentage() {
        if (totalSteps <= 0) return 0.0;
        return ((double) currentStep / totalSteps) * 100.0;
    }
}
