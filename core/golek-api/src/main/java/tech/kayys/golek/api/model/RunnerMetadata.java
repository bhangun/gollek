package tech.kayys.golek.api.model;

import java.util.List;
import java.util.Map;

/**
 * Runner metadata for selection and diagnostics
 */
public record RunnerMetadata(
                String name,
                String version,
                List<ModelFormat> supportedFormats,
                List<DeviceType> supportedDevices,
                Map<String, Object> capabilities) {
}
