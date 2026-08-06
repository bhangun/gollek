package tech.kayys.gollek.runner;

import java.util.List;
import java.util.Map;

import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.model.ModelFormat;

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
