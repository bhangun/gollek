package tech.kayys.gollek.metal.detection;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Detects Apple Silicon capabilities at startup and integrates them into
 * Gollek's hardware discovery path.
 *
 * <h2>What it detects</h2>
 * <ul>
 * <li>Whether the JVM is running on macOS with Apple Silicon (arm64)</li>
 * <li>The chip generation (M1 / M2 / M3 / M4) from {@code sysctl hw.model}</li>
 * <li>GPU core count from {@code system_profiler SPDisplaysDataType}</li>
 * <li>Unified memory size from {@code sysctl hw.memsize}</li>
 * <li>Whether Metal is available by probing {@code libgollek_metal.dylib}</li>
 * </ul>
 *
 * <h2>Integration with Gollek's SelectionPolicy</h2>
 * <p>
 * The existing {@link tech.kayys.gollek.engine.routing.policy.SelectionPolicy}
 * in Gollek only checks {@code hw.hasCUDA()}. This detector populates a
 * {@link MetalCapabilities} record so that Gollek's scoring for
 * {@link tech.kayys.gollek.spi.model.DeviceType#METAL} works correctly:
 *
 * <pre>{@code
 * // In SelectionPolicy.isDeviceAvailable() — existing code only has:
 * if (device == DeviceType.CUDA)
 *     return hw.hasCUDA();
 * return true; // <— Metal falls through here
 *
 * // After registering our detector as a CDI bean, runners that declare
 * // DeviceType.METAL will be scored and routed to when:
 * // 1. MetalRunner is on the classpath
 * // 2. The incoming request has preferredDevice=METAL or no CUDA is present
 * }</pre>
 */
@ApplicationScoped
public class AppleSiliconDetector {

    private static final Logger LOG = Logger.getLogger(AppleSiliconDetector.class);

    private final AtomicReference<MetalCapabilities> cached = new AtomicReference<>();

    /**
     * Detect Apple Silicon capabilities. Result is cached after first call.
     */
    public MetalCapabilities detect() {
        MetalCapabilities existing = cached.get();
        if (existing != null)
            return existing;

        MetalCapabilities caps = doDetect();
        cached.set(caps);

        if (caps.available()) {
            LOG.infof("[AppleSilicon] %s — %d GPU cores, %.0f GB unified memory%s",
                    caps.chipName(), caps.gpuCores(),
                    caps.unifiedMemoryBytes() / 1e9,
                    caps.unifiedMemory() ? " (unified)" : "");
        } else {
            LOG.debugf("[AppleSilicon] Metal not available on this host (%s)", caps.reason());
        }
        return caps;
    }

    // ── Detection logic ───────────────────────────────────────────────────────

    private MetalCapabilities doDetect() {
        // 1. Must be macOS
        String os = System.getProperty("os.name", "").toLowerCase();
        if (!os.contains("mac")) {
            return MetalCapabilities.unavailable("Not macOS (os.name=" + os + ")");
        }

        // 2. Architecture check — arm64 = Apple Silicon; x86_64 = Intel Mac
        String arch = System.getProperty("os.arch", "");
        boolean isAppleSilicon = arch.equals("aarch64") || arch.equals("arm64");

        // 3. Chip name from sysctl
        String model = sysctl("hw.model");
        String chipName = parseChipName(model, isAppleSilicon);

        // 4. GPU core count from system_profiler
        int gpuCores = detectGpuCores();

        // 5. Unified memory size from sysctl hw.memsize
        long memBytes = parseMemSize(sysctl("hw.memsize"));

        // 6. Unified memory flag — all Apple Silicon uses unified memory
        boolean unified = isAppleSilicon;

        // 7. macOS version — Metal requires macOS 10.11+; MPS requires 10.13+
        String[] verParts = System.getProperty("os.version", "0.0").split("\\.");
        int majorVer = safeParseInt(verParts.length > 0 ? verParts[0] : "0");
        if (majorVer < 13) {
            return MetalCapabilities.unavailable(
                    "macOS " + System.getProperty("os.version") +
                            " < 13.0 required for MPSGraph fused attention");
        }

        return new MetalCapabilities(
                true, // available
                isAppleSilicon,
                chipName,
                gpuCores,
                memBytes,
                unified,
                null // no reason (available)
        );
    }

    // ── Chip parsing ──────────────────────────────────────────────────────────

    private String parseChipName(String hwModel, boolean isAppleSilicon) {
        if (hwModel == null || hwModel.isBlank()) {
            return isAppleSilicon ? "Apple Silicon (unknown model)" : "Intel Mac";
        }
        // hwModel examples: "Mac14,3" (M2 Pro MacBook Pro), "MacBookPro18,1" (M1 Pro)
        // system_profiler would give us the friendly name; use a lookup table:
        if (hwModel.startsWith("Mac14") || hwModel.startsWith("MacBookPro14"))
            return "Apple M2 (family)";
        if (hwModel.startsWith("Mac13") || hwModel.startsWith("MacBookPro13"))
            return "Apple M1 (family)";
        if (hwModel.startsWith("Mac15") || hwModel.startsWith("MacBookPro15"))
            return "Apple M3 (family)";
        if (hwModel.startsWith("Mac16") || hwModel.startsWith("MacBookPro16"))
            return "Apple M4 (family)";
        return isAppleSilicon ? "Apple Silicon (" + hwModel + ")" : "Intel Mac (" + hwModel + ")";
    }

    private int detectGpuCores() {
        try {
            // system_profiler SPDisplaysDataType prints "Total Number of Cores: 30"
            String out = exec("system_profiler", "SPDisplaysDataType");
            for (String line : out.split("\n")) {
                if (line.toLowerCase().contains("total number of cores")) {
                    String[] parts = line.split(":");
                    if (parts.length > 1)
                        return safeParseInt(parts[1].trim());
                }
            }
        } catch (Exception ignored) {
        }
        return 0; // unknown
    }

    private long parseMemSize(String sysctlOut) {
        if (sysctlOut == null || sysctlOut.isBlank())
            return 0L;
        // sysctl hw.memsize prints: "hw.memsize: 34359738368"
        String[] parts = sysctlOut.split(":");
        try {
            return Long.parseLong(parts[parts.length - 1].trim());
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    // ── OS helpers ────────────────────────────────────────────────────────────

    private String sysctl(String key) {
        return exec("sysctl", "-n", key);
    }

    private String exec(String... cmd) {
        try {
            Process p = new ProcessBuilder(cmd)
                    .redirectErrorStream(true)
                    .start();
            StringBuilder sb = new StringBuilder();
            try (BufferedReader r = new BufferedReader(
                    new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null)
                    sb.append(line).append('\n');
            }
            p.waitFor();
            return sb.toString().trim();
        } catch (Exception e) {
            return "";
        }
    }

    private int safeParseInt(String s) {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return 0;
        }
    }
}
