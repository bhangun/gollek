package tech.kayys.gollek.inference.libtorch;

import java.nio.file.Path;

/**
 * Resolves whether FP8 rowwise path can be activated for a given model artifact.
 */
final class LibTorchFp8RowwisePlanner {

    private final LibTorchFp8CalibrationValidator calibrationValidator;
    private final LibTorchFp8CalibrationLoader calibrationLoader;

    LibTorchFp8RowwisePlanner() {
        this(new LibTorchFp8CalibrationValidator(), new LibTorchFp8CalibrationLoader());
    }

    LibTorchFp8RowwisePlanner(
            LibTorchFp8CalibrationValidator calibrationValidator,
            LibTorchFp8CalibrationLoader calibrationLoader) {
        this.calibrationValidator = calibrationValidator;
        this.calibrationLoader = calibrationLoader;
    }

    RowwisePlan resolve(Path modelPath, LibTorchAdvancedModeResolver.EffectiveAdvancedMode mode) {
        if (mode == null || !mode.advancedEnabled()) {
            return new RowwisePlan(false, "advanced.disabled", 0, 0.0d, "none", null);
        }
        if (!mode.fp8RowwiseEnabled()) {
            return new RowwisePlan(false, "fp8.rowwise.disabled", 0, 0.0d, "none", null);
        }
        if (modelPath == null) {
            return new RowwisePlan(false, "fp8.rowwise.model-path.missing", 0, 0.0d, "none", null);
        }
        Path calibration = calibrationValidator.resolveCandidate(modelPath);
        if (calibration == null) {
            return new RowwisePlan(false, "fp8.rowwise.calibration.missing", 0, 0.0d, "none", null);
        }
        LibTorchFp8CalibrationValidator.ValidationResult validation = calibrationValidator.validate(calibration);
        if (!validation.valid()) {
            return new RowwisePlan(false, validation.reason(), 0, 0.0d, calibration.toString(), null);
        }
        var loaded = calibrationLoader.load(calibration);
        if (loaded.isEmpty()) {
            return new RowwisePlan(false, "fp8.rowwise.calibration.load-failed", 0, 0.0d, calibration.toString(), null);
        }
        return new RowwisePlan(
                true,
                "fp8.rowwise.enabled",
                loaded.get().scaleCount(),
                loaded.get().meanScale(),
                loaded.get().path().toString(),
                loaded.get().rowScales());
    }

    record RowwisePlan(
            boolean enabled,
            String reason,
            int scaleCount,
            double scaleMean,
            String calibrationSource,
            float[] rowScales) {
    }
}
