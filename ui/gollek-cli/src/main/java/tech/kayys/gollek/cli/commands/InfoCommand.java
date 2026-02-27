package tech.kayys.gollek.cli.commands;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.model.SystemInfo;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.provider.LLMProvider;
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

    @Override
    public void run() {
        try {
            // Create system info
            SystemInfo systemInfo = createSystemInfo();

            // Print system information
            printSystemInfo(systemInfo);

            // Print available providers
            printProviders();
            // Print local model details
            printLocalModels();

        } catch (Exception e) {
            System.err.println("Failed to retrieve system information: " + e.getMessage());
            e.printStackTrace();
        }
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

    private void printSystemInfo(SystemInfo systemInfo) {
        System.out.println("┌────────────────────────────────────────────────────────────┐");
        System.out.println("│                   Gollek Inference CLI                      │");
        System.out.println("├────────────────────────────────────────────────────────────┤");
        System.out.printf("│ %-18s │ %s                                   │%n", "Version", systemInfo.getCliVersion());
        System.out.printf("│ %-18s │ %s                                   │%n", "Java Version",
                systemInfo.getJavaVersion());
        System.out.printf("│ %-18s │ %s                                   │%n", "OS Name", systemInfo.getOsName());
        System.out.printf("│ %-18s │ %s                                   │%n", "OS Version",
                systemInfo.getOsVersion());
        System.out.printf("│ %-18s │ %s                                   │%n", "OS Architecture",
                systemInfo.getOsArch());
        System.out.printf("│ %-18s │ %d                                   │%n", "Available Cores",
                systemInfo.getAvailableProcessors());
        System.out.printf("│ %-18s │ %s                                   │%n", "Memory",
                systemInfo.getMemoryFormatted());
        System.out.printf("│ %-18s │ %s                                   │%n", "User", systemInfo.getUserName());
        System.out.printf("│ %-18s │ %s                                   │%n", "Home Directory",
                systemInfo.getUserHome());
        System.out.println("└────────────────────────────────────────────────────────────┘");
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
                var meta = provider.metadata();
                var caps = provider.capabilities();
                String features = caps != null
                        ? (caps.isStreaming() ? "stream" : "sync") + (caps.isMultimodal() ? ",mm" : "")
                        : "n/a";
                System.out.printf("│ %-12s │ %-18s │ %-14s │ %-12s │%n",
                        provider.id(),
                        truncate(provider.name(), 18),
                        truncate(meta != null ? meta.getDefaultModel() : "N/A", 14),
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
