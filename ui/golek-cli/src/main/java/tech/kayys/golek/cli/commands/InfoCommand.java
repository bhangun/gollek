package tech.kayys.golek.cli.commands;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.model.SystemInfo;
import tech.kayys.golek.spi.provider.ProviderInfo;
import io.quarkus.arc.Unremovable;

import java.util.List;

/**
 * Display system information and available adapters/providers using GolekSdk.
 * Usage: golek info
 */
@Dependent
@Unremovable
@Command(name = "info", description = "Display system info and available adapters")
public class InfoCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Override
    public void run() {
        try {
            // Create system info
            SystemInfo systemInfo = createSystemInfo();

            // Print system information
            printSystemInfo(systemInfo);

            // Print available providers
            printProviders();

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
        System.out.println("│                   Golek Inference CLI                      │");
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
            List<ProviderInfo> providers = sdk.listAvailableProviders();

            if (providers.isEmpty()) {
                System.out.println("\nNo providers available.");
                return;
            }

            System.out.println("\n┌──────────────────────── Available Providers ───────────────────────┐");
            System.out.printf("│ %-15s │ %-20s │ %-10s │ %-10s │%n", "ID", "NAME", "VERSION", "STATUS");
            System.out.println("├────────────────────────────────────────────────────────────────────┤");

            for (ProviderInfo provider : providers) {
                String status = provider.healthStatus() != null
                        ? provider.healthStatus().toString()
                        : "UNKNOWN";

                System.out.printf("│ %-15s │ %-20s │ %-10s │ %-10s │%n",
                        provider.id(),
                        truncate(provider.name(), 20),
                        provider.version() != null ? provider.version() : "N/A",
                        status);
            }
            System.out.printf("│ Total: %-52d │%n", providers.size());
            System.out.println("└────────────────────────────────────────────────────────────────────┘");

        } catch (Exception e) {
            System.err.println("Failed to retrieve provider information: " + e.getMessage());
        }
    }

    private String truncate(String str, int maxLen) {
        if (str == null)
            return "";
        return str.length() > maxLen ? str.substring(0, maxLen - 3) + "..." : str;
    }
}
