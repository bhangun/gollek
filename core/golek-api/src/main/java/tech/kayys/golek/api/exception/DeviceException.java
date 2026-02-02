package tech.kayys.golek.api.exception;

import tech.kayys.golek.api.error.ErrorCode;

/**
 * Exception for device-related errors (GPU/TPU/NPU unavailable, OOM, etc.).
 * 
 * @author bhangun
 * @since 1.0.0
 */
public class DeviceException extends InferenceException {

    private final String deviceType;

    public DeviceException(ErrorCode errorCode, String message, String deviceType) {
        super(errorCode, message);
        this.deviceType = deviceType;
        addContext("deviceType", deviceType);
    }

    public DeviceException(ErrorCode errorCode, String message, String deviceType, Throwable cause) {
        super(errorCode, message, cause);
        this.deviceType = deviceType;
        addContext("deviceType", deviceType);
    }

    public String getDeviceType() {
        return deviceType;
    }
}
