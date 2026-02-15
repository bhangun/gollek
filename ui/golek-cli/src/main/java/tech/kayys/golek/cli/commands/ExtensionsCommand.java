package tech.kayys.golek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.spi.provider.ProviderInfo;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Show extension modules packaged in the current CLI build.
 */
@Dependent
@Unremovable
@Command(name = "extensions", description = "Show packaged extension modules and runtime providers")
public class ExtensionsCommand implements Runnable {

    private record ExtensionDef(String type, String name, String profile, String markerClass, String providerId) {
    }

    private static final List<ExtensionDef> EXTENSIONS = List.of(
            new ExtensionDef("runtime", "GGUF", "base", "tech.kayys.golek.inference.gguf.GGUFProvider", "gguf"),
            new ExtensionDef("runtime", "LibTorch", "base", "tech.kayys.golek.inference.libtorch.LibTorchProvider",
                    "libtorch"),
            new ExtensionDef("cloud", "Ollama", "ext-cloud-ollama", "tech.kayys.golek.provider.ollama.OllamaProvider",
                    "ollama"),
            new ExtensionDef("cloud", "Gemini", "ext-cloud-gemini", "tech.kayys.golek.provider.gemini.GeminiProvider",
                    "gemini"),
            new ExtensionDef("cloud", "Cerebras", "ext-cloud-cerebras",
                    "tech.kayys.golek.provider.cerebras.CerebrasProvider", "cerebras"));

    @Inject
    GolekSdk sdk;

    @Option(names = { "-a", "--all" }, description = "Show missing extensions too")
    boolean showAll;

    @Override
    public void run() {
        Set<String> runtimeProviders = getRuntimeProviderIds();

        System.out.printf("%-8s %-10s %-18s %-10s %-10s%n", "TYPE", "NAME", "PROFILE", "PACKAGED", "PROVIDER");
        System.out.println("-".repeat(62));

        int shown = 0;
        for (ExtensionDef ext : EXTENSIONS) {
            boolean packaged = isClassPresent(ext.markerClass());
            boolean providerAvailable = runtimeProviders.contains(ext.providerId());
            if (!showAll && !packaged) {
                continue;
            }
            System.out.printf("%-8s %-10s %-18s %-10s %-10s%n",
                    ext.type(),
                    ext.name(),
                    ext.profile(),
                    packaged ? "yes" : "no",
                    providerAvailable ? "yes" : "no");
            shown++;
        }

        if (shown == 0) {
            System.out.println("No packaged extensions found.");
        }

        if (!runtimeProviders.isEmpty()) {
            System.out.printf("%nRuntime providers: %s%n", String.join(", ", runtimeProviders));
        }
        System.out.println("Tip: enable cloud extensions at build time with -Pext-cloud-ollama,ext-cloud-gemini,ext-cloud-cerebras");
    }

    private Set<String> getRuntimeProviderIds() {
        Set<String> ids = new LinkedHashSet<>();
        try {
            List<ProviderInfo> providers = sdk.listAvailableProviders();
            for (ProviderInfo provider : providers) {
                if (provider.id() != null && !provider.id().isBlank()) {
                    ids.add(provider.id());
                }
            }
        } catch (Exception e) {
            // Keep output useful even if provider registry is unavailable.
        }
        return ids;
    }

    private boolean isClassPresent(String className) {
        try {
            Class.forName(className, false, Thread.currentThread().getContextClassLoader());
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }
}
