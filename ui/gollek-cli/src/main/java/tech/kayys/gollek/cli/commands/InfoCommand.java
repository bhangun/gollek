package tech.kayys.gollek.cli.commands;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.cli.util.CLIUtils;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.model.SystemInfo;
import picocli.CommandLine.Parameters;
import io.quarkus.arc.Unremovable;

import java.nio.file.Path;
import java.util.List;
import java.util.Comparator;

/**
 * Display system information and available adapters/providers using GollekSdk.
 * Usage: gollek info
 */
@Dependent
@Unremovable
@Command(name = "info", description = "Display system info and available adapters")
public class InfoCommand implements Runnable {

    @Inject
    GollekSdk sdk;
    @Inject
    ProviderRegistry providerRegistry;

    @Parameters(index = "0", description = "Optional model reference (ID, short ID, or name) to inspect", arity = "0..1")
    String modelRef;

    @Override
    public void run() {
        try {
            if (modelRef != null && !modelRef.isBlank()) {
                printDetailedModelInfo(modelRef.trim());
                return;
            }

            // Default behavior: system info
            SystemInfo systemInfo = createSystemInfo();
            printSystemInfo(systemInfo);
            printProviders();
            printLocalModels();

        } catch (Exception e) {
            System.err.println("Failed to retrieve information: " + e.getMessage());
        }
    }

    private void printDetailedModelInfo(String ref) {
        var entryOpt = LocalModelIndex.find(ref);
        if (entryOpt.isEmpty()) {
            System.err.println("Model not found: " + ref);
            return;
        }
        LocalModelIndex.Entry e = entryOpt.get();

        System.out.println("\n┌──────────────────────── Model Information ─────────────────────────┐");
        System.out.printf("│ %-18s │ %-36s │%n", "Name", truncate(e.name, 36));
        System.out.printf("│ %-18s │ %-36s │%n", "Internal ID", truncate(e.id, 36));
        System.out.printf("│ %-18s │ %-36s │%n", "Short ID", e.shortId);
        System.out.printf("│ %-18s │ %-36s │%n", "Architecture", e.architecture != null ? e.architecture : "unknown");
        System.out.printf("│ %-18s │ %-36s │%n", "Format", e.format != null ? e.format.toUpperCase() : "n/a");
        System.out.printf("│ %-18s │ %-36s │%n", "Size", CLIUtils.formatSize(e.sizeBytes));
        System.out.printf("│ %-18s │ %-36s │%n", "Modified", e.updatedAt != null ? e.updatedAt.substring(0, 10) : "n/a");
        System.out.printf("│ %-18s │ %-36s │%n", "Runnable", e.runnable ? "yes" : "no");
        
        if (e.quantStrategy != null) {
            System.out.println("├─────────────────── Quantization Details ───────────────────────────┤");
            System.out.printf("│ %-18s │ %-36s │%n", "Strategy", e.quantStrategy);
            System.out.printf("│ %-18s │ %-36d │%n", "Bits", e.quantBits);
            if (e.quantGroupSize > 0) {
                System.out.printf("│ %-18s │ %-36d │%n", "Group Size", e.quantGroupSize);
            }
            if (e.quantSourceModel != null) {
                System.out.printf("│ %-18s │ %-36s │%n", "Source Model", truncate(e.quantSourceModel, 36));
            }
        }
        
        System.out.println("├─────────────────── Filesystem Location ────────────────────────────┤");
        String path = e.path;
        if (path != null) {
            if (path.length() <= 36) {
                System.out.printf("│ %-18s │ %-36s │%n", "Path", path);
            } else {
                System.out.printf("│ %-18s │ %-36s │%n", "Path", truncate(path, 36));
                System.out.printf("│ %-18s │ %-36s │%n", "", path.substring(Math.min(path.length(), 36)));
            }
        }
        System.out.println("└────────────────────────────────────────────────────────────────────┘");
    }

    private SystemInfo createSystemInfo() {
        Runtime runtime = Runtime.getRuntime();

        return SystemInfo.builder()
                .cliVersion(getCliVersion())
                .javaVersion(System.getProperty("java.version"))
                .osName(System.getProperty("os.name"))
                .osVersion(System.getProperty("os.version"))
                .osArch(System.getProperty("os.arch"))
                .userName(System.getProperty("user.name"))
                .userHome(System.getProperty("user.home"))
                .totalMemory(runtime.totalMemory())
                .freeMemory(runtime.freeMemory())
                .maxMemory(runtime.maxMemory())
                .availableProcessors(runtime.availableProcessors())
                .build();
    }

    private String getCliVersion() {
        // Try to get version from manifest or fallback to default
        String version = getClass().getPackage().getImplementationVersion();
        return version != null ? version : "1.0.0";
    }

    private void printSystemInfo(SystemInfo systemInfo) throws Exception {
        // Detect kernel platform
        tech.kayys.gollek.plugin.kernel.KernelPlatform platform =
                tech.kayys.gollek.plugin.kernel.KernelPlatformDetector.detect();
        java.util.Map<String, String> platformMeta =
                tech.kayys.gollek.plugin.kernel.KernelPlatformDetector.getPlatformMetadata(platform);
        java.util.List<tech.kayys.gollek.plugin.kernel.KernelPlatform> availablePlatforms =
                tech.kayys.gollek.plugin.kernel.KernelPlatformDetector.getAvailablePlatforms();

        System.out.println("┌────────────────────────────────────────────────────────────┐");
        System.out.println("│                   Gollek Inference CLI                      │");
        System.out.println("├────────────────────────────────────────────────────────────┤");
        System.out.printf("│ %-18s │ %-36s │%n", "Version", systemInfo.getCliVersion());
        System.out.printf("│ %-18s │ %-36s │%n", "Java Version", systemInfo.getJavaVersion());
        System.out.printf("│ %-18s │ %-36s │%n", "OS Name", systemInfo.getOsName());
        System.out.printf("│ %-18s │ %-36s │%n", "OS Version", systemInfo.getOsVersion());
        System.out.printf("│ %-18s │ %-36s │%n", "OS Architecture", systemInfo.getOsArch());
        System.out.printf("│ %-18s │ %-36d │%n", "Available Cores", systemInfo.getAvailableProcessors());
        System.out.printf("│ %-18s │ %-36s │%n", "Memory", systemInfo.getMemoryFormatted());
        System.out.printf("│ %-18s │ %-36s │%n", "User", systemInfo.getUserName());
        System.out.printf("│ %-18s │ %-36s │%n", "Home Directory", systemInfo.getUserHome());
        System.out.println("├──────────────────── Kernel / GPU ──────────────────────────┤");
        System.out.printf("│ %-18s │ %-36s │%n", "Active Kernel",
                platform.getDisplayName());
        System.out.printf("│ %-18s │ %-36s │%n", "Available",
                availablePlatforms.stream()
                        .map(tech.kayys.gollek.plugin.kernel.KernelPlatform::getDisplayName)
                        .reduce((a, b) -> a + ", " + b).orElse("CPU"));
        if (!platformMeta.isEmpty()) {
            platformMeta.forEach((k, v) ->
                    System.out.printf("│ %-18s │ %-36s │%n", "  " + k, truncate(v, 36)));
        }
        System.out.println("├──────────────────── Runtime Context ───────────────────────┤");
        // Active model & provider from SDK
        String activeModel = sdk.resolveDefaultModel().orElse("(none)");
        String activeProvider = sdk.getPreferredProvider().orElse("auto");
        String activeFormat = resolveActiveFormat(activeModel);
        System.out.printf("│ %-18s │ %-36s │%n", "Active Model", truncate(activeModel, 36));
        System.out.printf("│ %-18s │ %-36s │%n", "Model Format", activeFormat);
        System.out.printf("│ %-18s │ %-36s │%n", "Provider", activeProvider);
        System.out.println("└────────────────────────────────────────────────────────────┘");
    }

    private String resolveActiveFormat(String modelId) {
        try {
            var models = sdk.listModels(0, 100);
            for (var m : models) {
                if (m.getModelId() != null && m.getModelId().equals(modelId)) {
                    return m.getFormat() != null ? m.getFormat() : "N/A";
                }
            }
        } catch (Exception ignored) {
        }
        return "N/A";
    }

    private void printProviders() {
        try {
            List<LLMProvider> providers = providerRegistry.getAllProviders().stream()
                    .sorted(Comparator.comparing(LLMProvider::id))
                    .toList();

            if (providers.isEmpty()) {
                System.out.println("\nNo providers available.");
                return;
            }

            System.out.println("\n┌──────────────────────── Available Providers ───────────────────────┐");
            System.out.printf("│ %-12s │ %-18s │ %-14s │ %-12s │%n", "ID", "NAME", "DEFAULT MODEL", "FEATURES");
            System.out.println("├────────────────────────────────────────────────────────────────────┤");

            for (LLMProvider provider : providers) {

                var caps = provider.capabilities();
                String features = caps != null
                        ? (caps.isStreaming() ? "stream" : "sync") + (caps.isMultimodal() ? ",mm" : "")
                        : "n/a";
                System.out.printf("│ %-12s │ %-18s │ %-14s │ %-12s │%n",
                        provider.id(),
                        truncate(provider.name(), 18),
                        "N/A",
                        truncate(features, 12));
            }
            System.out.printf("│ Total: %-52d │%n", providers.size());
            System.out.println("└────────────────────────────────────────────────────────────────────┘");

        } catch (Exception e) {
            System.err.println("Failed to retrieve provider information: " + e.getMessage());
        }
    }

    private void printLocalModels() {
        try {
            List<LocalModelIndex.Entry> entries = LocalModelIndex.refreshFromDisk().stream()
                    .sorted(Comparator.comparing((LocalModelIndex.Entry e) -> LocalModelIndex.parseInstant(e.updatedAt))
                            .reversed())
                    .toList();

            if (entries.isEmpty()) {
                System.out.println("\nNo local models found.");
                return;
            }

            System.out.println("\n┌──────────────────────── Local Models ──────────────────────────────┐");
            System.out.printf("│ %-20s │ %-12s │ %-7s │ %-7s │%n", "MODEL", "FORMAT", "SOURCE", "RUN");
            System.out.println("├────────────────────────────────────────────────────────────────────┤");
            for (LocalModelIndex.Entry e : entries) {
                String display = modelDisplay(e);
                System.out.printf("│ %-20s │ %-12s │ %-7s │ %-7s │%n",
                        truncate(display, 20),
                        truncate(e.format != null ? e.format : "n/a", 12),
                        truncate(e.source != null ? e.source : "local", 7),
                        e.runnable ? "yes" : "no");
                if (e.path != null && !e.path.isBlank()) {
                    System.out.println("│   path: " + truncate(e.path, 58));
                }
            }
            System.out.printf("│ Total: %-52d │%n", entries.size());
            System.out.println("└────────────────────────────────────────────────────────────────────┘");
        } catch (Exception e) {
            System.err.println("Failed to retrieve local model details: " + e.getMessage());
        }
    }

    private String modelDisplay(LocalModelIndex.Entry e) {
        if (e == null) {
            return "unknown";
        }
        if (e.name != null && !e.name.isBlank()) {
            return e.name;
        }
        if (e.path != null && !e.path.isBlank()) {
            try {
                Path p = Path.of(e.path);
                if (p.getFileName() != null) {
                    return p.getFileName().toString();
                }
            } catch (Exception ignored) {
                // fallback below
            }
        }
        return e.id != null ? e.id : "unknown";
    }

    private String truncate(String str, int maxLen) {
        if (str == null)
            return "";
        return str.length() > maxLen ? str.substring(0, maxLen - 3) + "..." : str;
    }
}
