package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

class LibTorchFp8RowwisePlannerTest {

    @TempDir
    Path tempDir;

    @Test
    void disablesWhenRowwiseFlagIsOff() throws Exception {
        Path model = tempDir.resolve("model.pt");
        Files.writeString(model, "x");
        LibTorchFp8RowwisePlanner planner = new LibTorchFp8RowwisePlanner();
        var plan = planner.resolve(model, mode(true, false));
        assertThat(plan.enabled()).isFalse();
        assertThat(plan.reason()).isEqualTo("fp8.rowwise.disabled");
        assertThat(plan.scaleCount()).isZero();
        assertThat(plan.scaleMean()).isZero();
        assertThat(plan.calibrationSource()).isEqualTo("none");
    }

    @Test
    void disablesWhenCalibrationMissing() throws Exception {
        Path model = tempDir.resolve("model.pt");
        Files.writeString(model, "x");
        LibTorchFp8RowwisePlanner planner = new LibTorchFp8RowwisePlanner();
        var plan = planner.resolve(model, mode(true, true));
        assertThat(plan.enabled()).isFalse();
        assertThat(plan.reason()).isEqualTo("fp8.rowwise.calibration.missing");
        assertThat(plan.scaleCount()).isZero();
        assertThat(plan.scaleMean()).isZero();
        assertThat(plan.calibrationSource()).isEqualTo("none");
    }

    @Test
    void enablesWhenCalibrationExists() throws Exception {
        Path model = tempDir.resolve("model.pt");
        Files.writeString(model, "x");
        Files.writeString(tempDir.resolve("model.pt.fp8calib.json"), "{\"version\":\"1\",\"row_scales\":[1.0,0.9]}");
        LibTorchFp8RowwisePlanner planner = new LibTorchFp8RowwisePlanner();
        var plan = planner.resolve(model, mode(true, true));
        assertThat(plan.enabled()).isTrue();
        assertThat(plan.reason()).isEqualTo("fp8.rowwise.enabled");
        assertThat(plan.scaleCount()).isEqualTo(2);
        assertThat(plan.scaleMean()).isCloseTo(0.95d, offset(1e-6d));
        assertThat(plan.calibrationSource()).contains("model.pt.fp8calib.json");
    }

    @Test
    void disablesWhenCalibrationSchemaInvalid() throws Exception {
        Path model = tempDir.resolve("model.pt");
        Files.writeString(model, "x");
        Files.writeString(tempDir.resolve("model.pt.fp8calib.json"), "{\"version\":\"1\",\"row_scales\":[]}");
        LibTorchFp8RowwisePlanner planner = new LibTorchFp8RowwisePlanner();
        var plan = planner.resolve(model, mode(true, true));
        assertThat(plan.enabled()).isFalse();
        assertThat(plan.reason()).contains("invalid.scales");
        assertThat(plan.scaleCount()).isZero();
        assertThat(plan.scaleMean()).isZero();
        assertThat(plan.calibrationSource()).contains("model.pt.fp8calib.json");
    }

    private LibTorchAdvancedModeResolver.EffectiveAdvancedMode mode(boolean advancedEnabled, boolean rowwiseEnabled) {
        return new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                advancedEnabled,
                "hybrid_fp8_bf16",
                rowwiseEnabled,
                false,
                false,
                "none",
                advancedEnabled ? "advanced.enabled" : "advanced.disabled",
                Optional.of(90),
                Set.of(89, 90));
    }
}
