package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.kernel.KernelPlatform;
import tech.kayys.gollek.plugin.kernel.KernelPlatformDetector;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Show extension modules, runner plugins, and kernel platform info.
 * Provides a complete view of the multilevel plugin system.
 */
@Dependent
@Unremovable
@Command(name = "extensions", description = "Show packaged extension modules, runner plugins, and kernel info")
public class ExtensionsCommand implements Runnable {

    private record ExtensionDef(String type, String name, String profile, String markerClass, String providerId) {
    }

    private static final List<ExtensionDef> EXTENSIONS = List.of(
            // Local runtimes
            new ExtensionDef("runtime", "GGUF", "base", "tech.kayys.gollek.inference.llamacpp.LlamaCppProvider", "gguf"),
            new ExtensionDef("runtime", "ONNX", "base", "tech.kayys.gollek.onnx.runner.OnnxRuntimeRunner", "onnx"),
            new ExtensionDef("runtime", "SafeTensor", "base",
                    "tech.kayys.gollek.safetensor.engine.warmup.SafetensorProvider", "safetensor"),
            new ExtensionDef("runtime", "LibTorch", "experimental",
                    "tech.kayys.gollek.inference.libtorch.LibTorchProvider", "libtorch"),
            new ExtensionDef("runtime", "LiteRT", "optional",
                    "tech.kayys.gollek.inference.litertTFLiteProvider", "litert"),
            new ExtensionDef("runtime", "TensorRT", "optional",
                    "tech.kayys.gollek.inference.tensorrt.TensorRTProvider", "tensorrt"),

            // Cloud providers
            new ExtensionDef("cloud", "Gemini", "ext-cloud-gemini",
                    "tech.kayys.gollek.provider.gemini.GeminiProvider", "gemini"),
            new ExtensionDef("cloud", "Cerebras", "ext-cloud-cerebras",
                    "tech.kayys.gollek.provider.cerebras.CerebrasProvider", "cerebras"),
            new ExtensionDef("cloud", "Mistral", "ext-cloud-mistral",
                    "tech.kayys.gollek.provider.mistral.MistralProvider", "mistral"),
            new ExtensionDef("cloud", "OpenAI", "optional",
                    "tech.kayys.gollek.provider.openai.OpenAiProvider", "openai"),
            new ExtensionDef("cloud", "Anthropic", "optional",
                    "tech.kayys.gollek.provider.anthropic.AnthropicProvider", "anthropic"),

            // Tool/integration providers
            new ExtensionDef("tool", "MCP", "base",
                    "tech.kayys.gollek.provider.mcp.McpProvider", "mcp"));

    @Inject
    GollekSdk sdk;

    @Inject
    Instance<RunnerPlugin> runnerPluginInstances;

    @Option(names = { "-a", "--all" }, description = "Show missing extensions too")
    boolean showAll;

    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String CYAN = "\u001B[36m";
    private static final String DIM = "\u001B[2m";
    private static final String BOLD = "\u001B[1m";

    @Override
    public void run() {
        // === Section 1: Kernel Platform ===
        printKernelInfo();
        System.out.println();

        // === Section 2: Extension Modules ===
        printExtensions();
        System.out.println();

        // === Section 3: Runner Plugins ===
        printRunnerPlugins();
        System.out.println();

        // === Section 4: Dynamic Plugins ===
        printDynamicPlugins();
    }

    private void printKernelInfo() {
        System.out.println(BOLD + "=== Kernel Platform ===" + RESET);
        KernelPlatform platform = KernelPlatformDetector.detect();
        System.out.printf("  Platform:     %s%s%s%n", CYAN, platform.getDisplayName(), RESET);
        System.out.printf("  GPU Accel:    %s%n", platform.isCpu()
                ? YELLOW + "No (CPU only)" + RESET
                : GREEN + "Yes" + RESET);
        System.out.printf("  Architecture: %s%n", System.getProperty("os.arch", "unknown"));
        System.out.printf("  OS:           %s %s%n",
                System.getProperty("os.name", "unknown"),
                System.getProperty("os.version", ""));
        System.out.printf("  Java:         %s (%s)%n",
                System.getProperty("java.version", "unknown"),
                System.getProperty("java.vendor", "unknown"));
    }

    private void printExtensions() {
        Set<String> runtimeProviders = getRuntimeProviderIds();

        System.out.println(BOLD + "=== Extension Modules ===" + RESET);
        System.out.printf("  %-8s %-12s %-18s %-10s %-10s%n",
                "TYPE", "NAME", "PROFILE", "PACKAGED", "PROVIDER");
        System.out.println("  " + "-".repeat(62));

        int shown = 0;
        for (ExtensionDef ext : EXTENSIONS) {
            boolean packaged = isClassPresent(ext.markerClass());
            boolean providerAvailable = runtimeProviders.contains(ext.providerId());
            if (!showAll && !packaged) {
                continue;
            }
            String packagedStr = packaged ? GREEN + "yes" + RESET : DIM + "no" + RESET;
            String providerStr = providerAvailable ? GREEN + "yes" + RESET : DIM + "no" + RESET;
            System.out.printf("  %-8s %-12s %-18s %-19s %-19s%n",
                    ext.type(),
                    ext.name(),
                    ext.profile(),
                    packagedStr,
                    providerStr);
            shown++;
        }

        if (shown == 0) {
            System.out.println("  No packaged extensions found. Use --all to see all possible extensions.");
        }

        if (!runtimeProviders.isEmpty()) {
            System.out.printf("%n  " + DIM + "Active providers: %s" + RESET + "%n",
                    String.join(", ", runtimeProviders));
        }
    }

    private void printRunnerPlugins() {
        System.out.println(BOLD + "=== Runner Plugins ===" + RESET);
        try {
            if (runnerPluginInstances == null || runnerPluginInstances.isUnsatisfied()) {
                System.out.println("  No runner plugins discovered.");
                return;
            }

            System.out.printf("  %-20s %-12s %-10s%n", "ID", "FORMAT", "STATUS");
            System.out.println("  " + "-".repeat(44));

            int count = 0;
            for (RunnerPlugin plugin : runnerPluginInstances) {
                String id = plugin.id();
                String format = plugin.format() != null ? plugin.format() : "unknown";
                String status = GREEN + "active" + RESET;
                System.out.printf("  %-20s %-12s %-19s%n", id, format, status);
                count++;
            }

            if (count == 0) {
                System.out.println("  No runner plugins active.");
            } else {
                System.out.printf("%n  " + DIM + "Total runners: %d" + RESET + "%n", count);
            }
        } catch (Exception e) {
            System.out.println("  " + YELLOW + "Failed to enumerate runner plugins: " + e.getMessage() + RESET);
        }
    }

    private void printDynamicPlugins() {
        System.out.println(BOLD + "=== Dynamic Plugins ===" + RESET);
        try {
            List<tech.kayys.gollek.spi.plugin.GollekPlugin.PluginMetadata> plugins = sdk.listPlugins();
            if (plugins != null && !plugins.isEmpty()) {
                for (var plugin : plugins) {
                    System.out.printf("  - %s (%s) v%s%n",
                            plugin.implementationClass(), plugin.id(), plugin.version());
                }
            } else {
                System.out.println("  No dynamic plugins discovered.");
            }
        } catch (Exception e) {
            System.out.println("  " + YELLOW + "Failed to fetch plugins: " + e.getMessage() + RESET);
        }

        System.out.println();
        System.out.println(DIM + "Tip: enable cloud extensions at build time with "
                + "-Pext-cloud-gemini,ext-cloud-cerebras" + RESET);
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
