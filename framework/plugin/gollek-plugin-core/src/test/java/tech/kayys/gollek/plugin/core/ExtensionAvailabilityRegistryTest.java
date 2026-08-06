/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.plugin.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExtensionAvailabilityRegistryTest {
    @Test
    void availabilityNormalizesNullCollectionsAndSummarizesState() {
        ExtensionAvailability availability = new ExtensionAvailability(
                "example",
                "Example",
                "audio",
                true,
                false,
                true,
                false,
                "fallback",
                null,
                List.of("wav"),
                Map.of("flacAvailable", "false"),
                "fallback active",
                null);

        assertEquals(List.of(), availability.capabilities());
        assertEquals(List.of("wav"), availability.formats());
        assertFalse(availability.attributeBoolean("flacAvailable"));
        assertEquals("fallback", availability.status());
        assertTrue(availability.compactSummary().contains("productionReady=false"));
    }

    @Test
    void registryConvertsProviderFailureIntoAvailabilityReport() {
        ExtensionAvailabilityProvider provider = new ExtensionAvailabilityProvider() {
            @Override
            public String extensionId() {
                return "broken";
            }

            @Override
            public String extensionName() {
                return "Broken Extension";
            }

            @Override
            public ExtensionAvailability availability() {
                throw new IllegalStateException("boom");
            }
        };

        ExtensionAvailabilityRegistry registry = new ExtensionAvailabilityRegistry();
        registry.register(provider);

        ExtensionAvailability availability = registry.availability("broken").orElseThrow();
        assertEquals("error", availability.status());
        assertFalse(availability.healthy());
        assertEquals("IllegalStateException", availability.attributes().get("errorType"));
        assertTrue(availability.remediationHints().getFirst().contains("broken"));
        assertTrue(registry.contractViolations().stream()
                .anyMatch(violation -> "availability_error".equals(violation.code())));
    }

    @Test
    void registryConvertsNullAvailabilityIntoErrorReport() {
        ExtensionAvailabilityProvider provider = new ExtensionAvailabilityProvider() {
            @Override
            public String extensionId() {
                return "null-report";
            }

            @Override
            public String extensionName() {
                return "Null Report";
            }

            @Override
            public ExtensionAvailability availability() {
                return null;
            }
        };

        ExtensionAvailabilityRegistry registry = new ExtensionAvailabilityRegistry();
        registry.register(provider);

        ExtensionAvailability availability = registry.availability("null-report").orElseThrow();
        assertEquals("error", availability.status());
        assertEquals("NullPointerException", availability.attributes().get("errorType"));
        assertTrue(registry.contractViolations().stream()
                .anyMatch(violation -> "availability_error".equals(violation.code())));
    }

    @Test
    void isolatedRegistryDoesNotMutateGlobalRegistry() {
        ExtensionAvailabilityRegistry.global().unregister("isolated-extension-fixture");
        ExtensionAvailabilityProvider provider = new ExtensionAvailabilityProvider() {
            @Override
            public String extensionId() {
                return "isolated-extension-fixture";
            }

            @Override
            public String extensionName() {
                return "Isolated Extension Fixture";
            }

            @Override
            public ExtensionAvailability availability() {
                return new ExtensionAvailability(
                        extensionId(),
                        extensionName(),
                        extensionKind(),
                        true,
                        false,
                        true,
                        true,
                        "ready",
                        List.of("tokenizer"),
                        List.of("sentencepiece"),
                        Map.of(),
                        "isolated fixture ready",
                        List.of());
            }
        };
        ExtensionAvailabilityRegistry isolated = ExtensionAvailabilityRegistry.create();

        isolated.register(provider);

        assertTrue(isolated.availability("isolated-extension-fixture").isPresent());
        assertTrue(ExtensionAvailabilityRegistry.global()
                .availability("isolated-extension-fixture")
                .isEmpty());
    }

    @Test
    void registryReportsAvailabilityContractViolations() {
        ExtensionAvailabilityProvider provider = new ExtensionAvailabilityProvider() {
            @Override
            public String extensionId() {
                return "Bad Extension";
            }

            @Override
            public String extensionName() {
                return "";
            }

            @Override
            public String extensionKind() {
                return "Audio";
            }

            @Override
            public ExtensionAvailability availability() {
                return new ExtensionAvailability(
                        "different",
                        "Different",
                        "tokenizer",
                        true,
                        true,
                        false,
                        true,
                        "ready",
                        List.of("tokenizer"),
                        List.of(),
                        Map.of(),
                        "bad state",
                        List.of());
            }
        };

        ExtensionAvailabilityRegistry registry = new ExtensionAvailabilityRegistry();
        registry.register(provider);

        List<String> codes = registry.contractViolations().stream()
                .map(ExtensionAvailabilityContractViolation::code)
                .toList();
        assertTrue(codes.contains("provider_id_invalid"));
        assertTrue(codes.contains("provider_kind_invalid"));
        assertTrue(codes.contains("provider_name_blank"));
        assertTrue(codes.contains("availability_id_mismatch"));
        assertTrue(codes.contains("availability_kind_mismatch"));
        assertTrue(codes.contains("attached_detached_conflict"));
        assertTrue(codes.contains("production_ready_detached"));
        assertTrue(codes.contains("production_ready_unhealthy"));

        ExtensionAvailabilityContractReport report = registry.contractReport();
        assertFalse(report.passed());
        assertTrue(report.failed());
        assertEquals("failed", report.status());
        assertEquals(codes.size(), report.violationCount());
        assertTrue(report.byExtensionId().containsKey("Bad Extension"));
        assertTrue(report.summaries().stream()
                .anyMatch(summary -> summary.contains("availability_id_mismatch")));
    }

    @Test
    void registryDiscoversProvidersFromExplicitClassLoader(@TempDir Path tempDir) throws Exception {
        Path serviceDirectory = tempDir.resolve("META-INF/services");
        Files.createDirectories(serviceDirectory);
        Files.writeString(
                serviceDirectory.resolve(ExtensionAvailabilityProvider.class.getName()),
                TestClassLoaderExtensionAvailabilityProvider.class.getName() + System.lineSeparator(),
                StandardCharsets.UTF_8);

        URL[] urls = { tempDir.toUri().toURL() };
        try (URLClassLoader pluginClassLoader =
                new URLClassLoader(urls, ExtensionAvailabilityRegistryTest.class.getClassLoader())) {
            ExtensionAvailabilityRegistry registry = new ExtensionAvailabilityRegistry();

            registry.discoverServiceLoaderProviders(pluginClassLoader);

            ExtensionAvailability availability =
                    registry.availability(TestClassLoaderExtensionAvailabilityProvider.ID).orElseThrow();
            assertEquals("tokenizer", availability.kind());
            assertEquals("ready", availability.status());
            assertTrue(availability.productionReady());
        }
    }
}
