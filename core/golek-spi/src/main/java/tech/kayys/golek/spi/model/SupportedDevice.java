package tech.kayys.golek.spi.model;

/**
 * Represents a device that is supported by a model, including minimum
 * requirements.
 */
public record SupportedDevice(
        DeviceType type,
        String minVersion,
        MemoryRequirements memoryRequirements) {

    public SupportedDevice {
        if (type == null) {
            throw new IllegalArgumentException("Device type cannot be null");
        }
    }
}