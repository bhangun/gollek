package tech.kayys.golek.cli.commands;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.spi.provider.ProviderInfo;

import java.util.List;

/**
 * List available providers using GolekSdk.
 * Usage: golek providers [-v]
 */
@Command(name = "providers", aliases = "ps", description = "List available inference providers")
@Dependent
public class ProvidersCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Option(names = { "-v", "--verbose" }, description = "Show detailed capabilities")
    boolean verbose;

    @Override
    public void run() {
        try {
            List<ProviderInfo> providers = sdk.listAvailableProviders();

            if (providers.isEmpty()) {
                System.out.println("No providers available.");
                return;
            }

            System.out.printf("%-15s %-20s %-10s %-10s%n", "ID", "NAME", "VERSION", "STATUS");
            System.out.println("-".repeat(60));

            for (ProviderInfo provider : providers) {
                String status = provider.getHealthStatus() != null
                        ? provider.getHealthStatus().toString()
                        : "UNKNOWN";

                System.out.printf("%-15s %-20s %-10s %-10s%n",
                        provider.getId(),
                        truncate(provider.getName(), 20),
                        provider.getVersion() != null ? provider.getVersion() : "N/A",
                        status);

                if (verbose && provider.getCapabilities() != null) {
                    printCapabilities(provider);
                }
            }
            System.out.printf("%n%d provider(s) available%n", providers.size());

        } catch (Exception e) {
            System.err.println("Failed to list providers: " + e.getMessage());
        }
    }

    private void printCapabilities(ProviderInfo provider) {
        var caps = provider.getCapabilities();
        System.out.printf("    Streaming: %s, Function Calling: %s, Multimodal: %s%n",
                caps.isStreaming() ? "✓" : "✗",
                caps.isFunctionCalling() ? "✓" : "✗",
                caps.isMultimodal() ? "✓" : "✗");
        System.out.printf("    Max Context: %d, Max Output: %d%n",
                caps.getMaxContextTokens(),
                caps.getMaxOutputTokens());
    }

    private String truncate(String str, int maxLen) {
        if (str == null)
            return "";
        return str.length() > maxLen ? str.substring(0, maxLen - 3) + "..." : str;
    }
}
