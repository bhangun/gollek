package tech.kayys.gollek.spi.model;

import tech.kayys.alkhawarizm.core.model.ModelFormat;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import java.util.List;
import java.util.Map;

/**
 * Runner metadata for selection and diagnostics.
 *
 * <p>
 * This lives in {@code spi.model} because the active {@code ModelRunner}
 * contract references {@code tech.kayys.gollek.spi.model.RunnerMetadata}.
 */
public record RunnerMetadata(
                String name,
                String version,
                List<ModelFormat> supportedFormats,
                List<DeviceType> supportedDevices,
                Map<String, Object> capabilities) {
}
