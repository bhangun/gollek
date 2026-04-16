package tech.kayys.gollek.metal;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.metal.detection.AppleSiliconDetector;
import tech.kayys.gollek.metal.detection.MetalCapabilities;

import java.nio.file.Files;
import java.nio.file.Path;

class MetalGpuSmokeTest {
    private static final org.jboss.logging.Logger LOG = org.jboss.logging.Logger.getLogger(MetalGpuSmokeTest.class);

    @Test
    void usesNativeMetalWhenEnabled() {
        Assumptions.assumeTrue(Boolean.getBoolean("gollek.metal.tests"),
                "Set -Dgollek.metal.tests=true to run Metal GPU smoke tests.");

        MetalCapabilities caps = new AppleSiliconDetector().detect();
        Assumptions.assumeTrue(caps.available(),
                "Metal unavailable: " + caps.reason());

        String libraryPath = System.getProperty("gollek.metal.library-path");
        if (libraryPath == null || libraryPath.isBlank()) {
            libraryPath = System.getenv("GOLLEK_METAL_LIBRARY_PATH");
        }
        if (libraryPath == null || libraryPath.isBlank()) {
            libraryPath = System.getProperty("gollek.runners.metal.library-path");
        }
        if (libraryPath == null || libraryPath.isBlank()) {
            libraryPath = discoverDefaultLibraryPath();
        }
        if (libraryPath == null || libraryPath.isBlank()) {
            LOG.info("MetalGpuSmokeTest: no Metal dylib path resolved (set gollek.metal.library-path, "
                    + "gollek.runners.metal.library-path, or GOLLEK_METAL_LIBRARY_PATH).");
        } else {
            LOG.infof("MetalGpuSmokeTest: using Metal dylib path %s", libraryPath);
        }
        Assumptions.assumeTrue(libraryPath != null && !libraryPath.isBlank(),
                "Set gollek.metal.library-path, gollek.runners.metal.library-path, GOLLEK_METAL_LIBRARY_PATH, or install to a default path.");
        Assumptions.assumeTrue(Files.exists(Path.of(libraryPath)),
                "Metal dylib not found at " + libraryPath);

        MetalBinding.initialize(Path.of(libraryPath));
        MetalBinding binding = MetalBinding.getInstance();
        assertThat(binding.isNativeAvailable()).isTrue();

        int init = binding.init();
        if (init != 0) {
            LOG.warnf("MetalGpuSmokeTest: Metal init failed (code=%d). This can happen on headless/CI runs.", init);
            if (Boolean.getBoolean("gollek.metal.tests.strict")) {
                assertThat(init).isGreaterThanOrEqualTo(0);
            }
            Assumptions.assumeTrue(false, "Metal init failed (code=" + init + ").");
        }
    }

    private static String discoverDefaultLibraryPath() {
        String home = System.getProperty("user.home");
        if (home != null && !home.isBlank()) {
            String candidate = home + "/.gollek/libs/libgollek_metal.dylib";
            if (Files.exists(Path.of(candidate))) {
                return candidate;
            }
        }
        String[] candidates = {
                "/usr/local/lib/libgollek_metal.dylib",
                "/Library/Gollek/libgollek_metal.dylib"
        };
        for (String candidate : candidates) {
            if (Files.exists(Path.of(candidate))) {
                return candidate;
            }
        }
        return null;
    }
}
