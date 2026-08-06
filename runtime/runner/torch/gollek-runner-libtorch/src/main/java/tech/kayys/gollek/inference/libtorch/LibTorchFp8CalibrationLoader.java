package tech.kayys.gollek.inference.libtorch;

import jakarta.json.Json;
import jakarta.json.JsonArray;
import jakarta.json.JsonObject;
import jakarta.json.JsonValue;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Loads and caches validated FP8 rowwise calibration artifacts.
 */
final class LibTorchFp8CalibrationLoader {

    private final LibTorchFp8CalibrationValidator validator;
    private final ConcurrentMap<Path, CachedCalibration> cache = new ConcurrentHashMap<>();

    LibTorchFp8CalibrationLoader() {
        this(new LibTorchFp8CalibrationValidator());
    }

    LibTorchFp8CalibrationLoader(LibTorchFp8CalibrationValidator validator) {
        this.validator = validator;
    }

    Optional<CalibrationData> load(Path calibrationFile) {
        if (calibrationFile == null || !Files.exists(calibrationFile)) {
            return Optional.empty();
        }
        FileFingerprint fingerprint = fingerprint(calibrationFile);
        if (fingerprint == null) {
            return Optional.empty();
        }
        CachedCalibration cached = cache.get(calibrationFile);
        if (cached != null && cached.fingerprint().equals(fingerprint)) {
            return Optional.of(cached.data());
        }

        LibTorchFp8CalibrationValidator.ValidationResult valid = validator.validate(calibrationFile);
        if (!valid.valid()) {
            return Optional.empty();
        }

        try (InputStream in = Files.newInputStream(calibrationFile)) {
            JsonObject root = Json.createReader(in).readObject();
            JsonArray scales = root.getJsonArray("row_scales");
            if (scales == null) {
                scales = root.getJsonArray("scales");
            }
            if (scales == null || scales.isEmpty()) {
                return Optional.empty();
            }
            int count = scales.size();
            double sum = 0.0d;
            float[] rowScales = new float[count];
            int index = 0;
            for (JsonValue value : scales) {
                float v = (float) ((jakarta.json.JsonNumber) value).doubleValue();
                sum += v;
                rowScales[index++] = v;
            }
            CalibrationData data = new CalibrationData(
                    calibrationFile,
                    count,
                    sum / count,
                    rowScales);
            cache.put(calibrationFile, new CachedCalibration(fingerprint, data));
            return Optional.of(data);
        } catch (Exception e) {
            return Optional.empty();
        }
    }

    void invalidate(Path calibrationFile) {
        if (calibrationFile != null) {
            cache.remove(calibrationFile);
        }
    }

    void invalidateAll() {
        cache.clear();
    }

    private FileFingerprint fingerprint(Path calibrationFile) {
        try {
            FileTime modified = Files.getLastModifiedTime(calibrationFile);
            long size = Files.size(calibrationFile);
            return new FileFingerprint(modified.toMillis(), size);
        } catch (Exception e) {
            return null;
        }
    }

    record CalibrationData(
            Path path,
            int scaleCount,
            double meanScale,
            float[] rowScales) {
    }

    private record FileFingerprint(long modifiedMillis, long sizeBytes) {
    }

    private record CachedCalibration(FileFingerprint fingerprint, CalibrationData data) {
    }
}
