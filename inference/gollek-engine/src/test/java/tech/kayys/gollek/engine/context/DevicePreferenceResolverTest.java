package tech.kayys.gollek.engine.context;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.model.DeviceType;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;

class DevicePreferenceResolverTest {

    @Test
    void autoPrefersMetalOnAppleSilicon() {
        String originalOs = System.getProperty("os.name");
        String originalArch = System.getProperty("os.arch");
        try {
            System.setProperty("os.name", "Mac OS X");
            System.setProperty("os.arch", "arm64");

            DevicePreferenceResolver resolver = new DevicePreferenceResolver("auto", true);
            InferenceRequest request = InferenceRequest.builder()
                    .model("m")
                    .message(Message.user("hi"))
                    .build();
            RequestContext context = RequestContext.of("req");

            RequestContext resolved = resolver.apply(context, request);
            assertTrue(resolved.preferredDevice().isPresent());
            assertEquals(DeviceType.METAL, resolved.preferredDevice().get());
        } finally {
            restoreProperty("os.name", originalOs);
            restoreProperty("os.arch", originalArch);
        }
    }

    @Test
    void autoSkipsMetalOnNonApple() {
        String originalOs = System.getProperty("os.name");
        String originalArch = System.getProperty("os.arch");
        try {
            System.setProperty("os.name", "Linux");
            System.setProperty("os.arch", "x86_64");

            DevicePreferenceResolver resolver = new DevicePreferenceResolver("auto", true);
            InferenceRequest request = InferenceRequest.builder()
                    .model("m")
                    .message(Message.user("hi"))
                    .build();
            RequestContext context = RequestContext.of("req");

            RequestContext resolved = resolver.apply(context, request);
            assertFalse(resolved.preferredDevice().isPresent());
        } finally {
            restoreProperty("os.name", originalOs);
            restoreProperty("os.arch", originalArch);
        }
    }

    @Test
    void requestDeviceOverridesAuto() {
        String originalOs = System.getProperty("os.name");
        String originalArch = System.getProperty("os.arch");
        try {
            System.setProperty("os.name", "Mac OS X");
            System.setProperty("os.arch", "arm64");

            DevicePreferenceResolver resolver = new DevicePreferenceResolver("auto", true);
            InferenceRequest request = InferenceRequest.builder()
                    .model("m")
                    .message(Message.user("hi"))
                    .parameter("device", "cpu")
                    .build();
            RequestContext context = RequestContext.of("req");

            RequestContext resolved = resolver.apply(context, request);
            assertTrue(resolved.preferredDevice().isPresent());
            assertEquals(DeviceType.CPU, resolved.preferredDevice().get());
        } finally {
            restoreProperty("os.name", originalOs);
            restoreProperty("os.arch", originalArch);
        }
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}
