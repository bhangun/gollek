package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

class LibTorchFp8CalibrationLoaderTest {

    @TempDir
    Path tempDir;

    @Test
    void reloadsCalibrationWhenFileChanges() throws Exception {
        Path file = tempDir.resolve("model.pt.fp8.json");
        Files.writeString(file, "{\"version\":\"1\",\"row_scales\":[1.0,0.9]}");

        LibTorchFp8CalibrationLoader loader = new LibTorchFp8CalibrationLoader();
        var first = loader.load(file);
        assertThat(first).isPresent();
        assertThat(first.get().scaleCount()).isEqualTo(2);

        Thread.sleep(5);
        Files.writeString(file, "{\"version\":\"1\",\"row_scales\":[1.0,0.9,0.8]}");

        var second = loader.load(file);
        assertThat(second).isPresent();
        assertThat(second.get().scaleCount()).isEqualTo(3);
    }

    @Test
    void supportsExplicitInvalidation() throws Exception {
        Path file = tempDir.resolve("model.pt.fp8.json");
        Files.writeString(file, "{\"version\":\"1\",\"row_scales\":[1.0]}");

        LibTorchFp8CalibrationLoader loader = new LibTorchFp8CalibrationLoader();
        assertThat(loader.load(file)).isPresent();

        loader.invalidate(file);
        loader.invalidateAll();
        assertThat(loader.load(file)).isPresent();
    }
}
