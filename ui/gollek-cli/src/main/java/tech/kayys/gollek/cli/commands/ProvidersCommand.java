package tech.kayys.gollek.cli.commands;

import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import io.quarkus.arc.Unremovable;

import java.util.List;
import jakarta.enterprise.context.Dependent;

/**
 * List available providers using GollekSdk.
 * Usage: gollek providers [-v]
 */
@Dependent
@Unremovable
@Command(name = "providers", description = "List available LLM providers and their health")
public class ProvidersCommand implements Runnable {

    @Inject
    GollekSdk sdk;

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
                String status = provider.healthStatus() != null
                        ? provider.healthStatus().toString()
                        : "UNKNOWN";

                System.out.printf("%-15s %-20s %-10s %-10s%n",
                        provider.id(),
                        truncate(provider.name(), 20),
                        provider.version() != null ? provider.version() : "N/A",
                        status);

                if (verbose && provider.capabilities() != null) {
                    printCapabilities(provider);
                }
            }
            System.out.printf("%n%d provider(s) available%n", providers.size());

        } catch (Exception e) {
            System.err.println("Failed to list providers: " + e.getMessage());
        }
    }

    private void printCapabilities(ProviderInfo provider) {
        var caps = provider.capabilities();
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
