package tech.kayys.gollek.inference.libtorch;

import jakarta.json.Json;
import jakarta.json.JsonArray;
import jakarta.json.JsonObject;
import jakarta.json.JsonValue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Validates FP8 rowwise calibration artifacts.
 */
final class LibTorchFp8CalibrationValidator {

    ValidationResult validate(Path calibrationFile) {
        if (calibrationFile == null) {
            return new ValidationResult(false, "fp8.rowwise.calibration.missing");
        }
        if (!Files.exists(calibrationFile)) {
            return new ValidationResult(false, "fp8.rowwise.calibration.missing");
        }
        try (InputStream in = Files.newInputStream(calibrationFile)) {
            JsonObject root = Json.createReader(in).readObject();
            if (!hasValidVersion(root)) {
                return new ValidationResult(false, "fp8.rowwise.calibration.invalid.version");
            }
            if (!hasValidScales(root)) {
                return new ValidationResult(false, "fp8.rowwise.calibration.invalid.scales");
            }
            return new ValidationResult(true, "fp8.rowwise.calibration.valid");
        } catch (IOException e) {
            return new ValidationResult(false, "fp8.rowwise.calibration.io-error");
        } catch (Exception e) {
            return new ValidationResult(false, "fp8.rowwise.calibration.invalid-json");
        }
    }

    Path resolveCandidate(Path modelPath) {
        if (modelPath == null || modelPath.getParent() == null) {
            return null;
        }
        String fileName = modelPath.getFileName().toString();
        Path dir = modelPath.getParent();
        List<Path> candidates = List.of(
                dir.resolve(fileName + ".fp8.json"),
                dir.resolve(fileName + ".fp8calib.json"),
                dir.resolve(fileName + ".calibration.json"));
        return candidates.stream().filter(Files::exists).findFirst().orElse(null);
    }

    private boolean hasValidVersion(JsonObject root) {
        String version = root.getString("version", "").trim();
        return !version.isEmpty();
    }

    private boolean hasValidScales(JsonObject root) {
        JsonArray scales = root.getJsonArray("row_scales");
        if (scales == null) {
            scales = root.getJsonArray("scales");
        }
        if (scales == null || scales.isEmpty()) {
            return false;
        }
        for (JsonValue value : scales) {
            if (value.getValueType() != JsonValue.ValueType.NUMBER) {
                return false;
            }
        }
        return true;
    }

    record ValidationResult(boolean valid, String reason) {
    }
}
