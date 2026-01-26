package tech.kayys.golek.model.core;

import tech.kayys.golek.api.model.DeviceType;

/**
 * Device configuration supported by a model
 */
public record SupportedDevice(
                DeviceType type,
                long minVramBytes,
                boolean preferred) {
}
