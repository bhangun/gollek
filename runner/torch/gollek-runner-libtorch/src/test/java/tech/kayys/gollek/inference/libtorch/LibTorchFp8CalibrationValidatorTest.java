package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

class LibTorchFp8CalibrationValidatorTest {

    @TempDir
    Path tempDir;

    @Test
    void validatesMinimalCalibrationSchema() throws Exception {
        Path file = tempDir.resolve("model.pt.fp8.json");
        Files.writeString(file, "{\"version\":\"1\",\"row_scales\":[1.0,0.8]}");
        LibTorchFp8CalibrationValidator validator = new LibTorchFp8CalibrationValidator();
        var result = validator.validate(file);
        assertThat(result.valid()).isTrue();
    }

    @Test
    void failsWhenVersionMissing() throws Exception {
        Path file = tempDir.resolve("model.pt.fp8.json");
        Files.writeString(file, "{\"row_scales\":[1.0]}");
        LibTorchFp8CalibrationValidator validator = new LibTorchFp8CalibrationValidator();
        var result = validator.validate(file);
        assertThat(result.valid()).isFalse();
        assertThat(result.reason()).contains("version");
    }

    @Test
    void failsWhenScalesInvalid() throws Exception {
        Path file = tempDir.resolve("model.pt.fp8.json");
        Files.writeString(file, "{\"version\":\"1\",\"row_scales\":[\"bad\"]}");
        LibTorchFp8CalibrationValidator validator = new LibTorchFp8CalibrationValidator();
        var result = validator.validate(file);
        assertThat(result.valid()).isFalse();
        assertThat(result.reason()).contains("scales");
    }
}
