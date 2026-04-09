package tech.kayys.gollek.engine.context;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.model.DeviceType;

import java.util.Locale;
import java.util.Map;
import java.util.Optional;

@ApplicationScoped
public class DevicePreferenceResolver {

    private static final Logger LOG = Logger.getLogger(DevicePreferenceResolver.class);

    @ConfigProperty(name = "gollek.engine.device.preference", defaultValue = "auto")
    String devicePreference;

    @ConfigProperty(name = "gollek.engine.device.preference.allow-request-override", defaultValue = "true")
    boolean allowRequestOverride;

    DevicePreferenceResolver() {
    }

    DevicePreferenceResolver(String devicePreference, boolean allowRequestOverride) {
        this.devicePreference = devicePreference;
        this.allowRequestOverride = allowRequestOverride;
    }

    public RequestContext apply(RequestContext context, InferenceRequest request) {
        if (context == null && request == null) {
            return null;
        }
        RequestContext base = context;
        if (base == null) {
            base = RequestContext.of(request != null ? request.getRequestId() : null);
        }
        if (base.preferredDevice() != null && base.preferredDevice().isPresent()) {
            return base;
        }

        Optional<DeviceType> preferred = Optional.empty();
        if (allowRequestOverride && request != null) {
            preferred = resolveFromRequest(request);
        }
        if (preferred.isEmpty()) {
            preferred = resolveDefault();
        }
        final RequestContext ctx = base;
        return preferred.map(d -> ctx.withPreferredDevice(d.getId())).orElse(ctx);
    }

    private Optional<DeviceType> resolveFromRequest(InferenceRequest request) {
        Map<String, Object> parameters = request.getParameters();
        if (parameters == null || parameters.isEmpty()) {
            return Optional.empty();
        }
        Object raw = parameters.get("device");
        if (raw == null) {
            raw = parameters.get("preferredDevice");
        }
        return parseDevice(raw != null ? raw.toString() : null);
    }

    private Optional<DeviceType> resolveDefault() {
        if (devicePreference == null || devicePreference.isBlank()) {
            return Optional.empty();
        }
        String normalized = devicePreference.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "none", "disabled", "off" -> Optional.empty();
            case "auto" -> isAppleSilicon() ? Optional.of(DeviceType.METAL) : Optional.empty();
            case "metal" -> Optional.of(DeviceType.METAL);
            case "cuda", "gpu" -> Optional.of(DeviceType.CUDA);
            case "rocm" -> Optional.of(DeviceType.ROCM);
            case "cpu" -> Optional.of(DeviceType.CPU);
            case "tpu" -> Optional.of(DeviceType.TPU);
            case "npu" -> Optional.of(DeviceType.NPU);
            default -> {
                LOG.warnf("Unknown gollek.engine.device.preference=%s; ignoring", devicePreference);
                yield Optional.empty();
            }
        };
    }

    private Optional<DeviceType> parseDevice(String raw) {
        if (raw == null || raw.isBlank()) {
            return Optional.empty();
        }
        String normalized = raw.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "metal", "mps" -> Optional.of(DeviceType.METAL);
            case "cuda", "gpu" -> Optional.of(DeviceType.CUDA);
            case "rocm" -> Optional.of(DeviceType.ROCM);
            case "cpu" -> Optional.of(DeviceType.CPU);
            case "tpu" -> Optional.of(DeviceType.TPU);
            case "npu" -> Optional.of(DeviceType.NPU);
            default -> Optional.empty();
        };
    }

    private boolean isAppleSilicon() {
        String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (!osName.contains("mac")) {
            return false;
        }
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        return arch.contains("aarch64") || arch.contains("arm64");
    }
}
